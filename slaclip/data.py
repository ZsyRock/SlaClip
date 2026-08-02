from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import urllib.request
import zipfile
from functools import partial
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from datasets import load_dataset, load_from_disk
from transformers import BertTokenizerFast

_NAMES_MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
_NAMES_MAX_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
_NAMES_MAX_ARCHIVE_MEMBERS = 2_000


def _mnist_transforms():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


def _fmnist_transforms():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]
    )


def _cifar_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


def padded_collate_imdb(batch, padding_idx=0):
    x = pad_sequence(
        [elem["input_ids"] for elem in batch],
        batch_first=True,
        padding_value=int(padding_idx),
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, y


def _clean_name_line(s: str) -> str:
    s = s.replace("\ufeff", "").strip()
    return re.sub(r"[\r\n\t]+", " ", s).strip()


def _build_char_vocab(samples: list[str]):
    charset = set()
    for text in samples:
        charset.update(text)
    id2char = ["<pad>", "<unk>"] + sorted(charset)
    return {ch: i for i, ch in enumerate(id2char)}, id2char


def _encode_name(name: str, char2id: dict, max_len: int | None):
    ids = [int(char2id.get(ch, 1)) for ch in name]
    if max_len is not None and max_len > 0:
        ids = ids[: int(max_len)]
    if not ids:
        ids = [1]
    return torch.tensor(ids, dtype=torch.long)


class NamesCharDataset(torch.utils.data.Dataset):
    def __init__(self, xs: list[torch.Tensor], ys: list[int]):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, idx):
        return self.xs[idx], torch.tensor(self.ys[idx], dtype=torch.long)


def collate_names_char(batch, pad_id: int = 0):
    xs = [item[0] for item in batch]
    ys = torch.stack([item[1] for item in batch]).long()
    return pad_sequence(xs, batch_first=True, padding_value=int(pad_id)), ys


def _indices_sha256(indices: Sequence[int]) -> str:
    values = np.asarray(list(indices), dtype=np.int64)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _json_sha256(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _named_file_hashes(directory: str, filenames: Sequence[str]) -> dict:
    per_file = {
        str(filename): _file_sha256(os.path.join(directory, str(filename)))
        for filename in sorted(filenames)
    }
    return {
        "algorithm": "sha256",
        "per_file": per_file,
        "manifest_sha256": _json_sha256(per_file),
    }


def deterministic_split_indices(
    size: int,
    holdout_fraction: float,
    seed: int,
    *,
    labels: Sequence[int] | torch.Tensor | np.ndarray | None = None,
) -> tuple[list[int], list[int]]:
    """Return deterministic train/holdout indices, stratified when labels exist."""

    size = int(size)
    holdout_fraction = float(holdout_fraction)
    if size < 2:
        raise ValueError("A train/holdout split requires at least two samples")
    if not 0.01 <= holdout_fraction <= 0.99:
        raise ValueError("holdout_fraction must be in [0.01, 0.99]")

    target = max(1, min(size - 1, int(round(size * holdout_fraction))))
    rng = np.random.default_rng(int(seed))

    if labels is None:
        shuffled = rng.permutation(size)
        holdout = shuffled[:target]
        train = shuffled[target:]
    else:
        labels_np = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(labels_np) != size:
            raise ValueError("labels length must match size")
        classes, counts = np.unique(labels_np, return_counts=True)
        raw = counts.astype(np.float64) * holdout_fraction
        allocations = np.floor(raw).astype(np.int64)
        # When possible, retain at least one member of each class in training.
        capacities = np.maximum(0, counts - 1)
        allocations = np.minimum(allocations, capacities)
        remaining = target - int(allocations.sum())
        order = np.argsort(-(raw - np.floor(raw)), kind="stable")
        while remaining > 0:
            changed = False
            for pos in order:
                if allocations[pos] < capacities[pos]:
                    allocations[pos] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break
        if remaining:
            raise ValueError("Unable to construct requested stratified holdout")

        holdout_parts = []
        train_parts = []
        for class_value, allocation in zip(classes, allocations):
            class_indices = np.flatnonzero(labels_np == class_value)
            class_indices = rng.permutation(class_indices)
            holdout_parts.append(class_indices[: int(allocation)])
            train_parts.append(class_indices[int(allocation) :])
        holdout = rng.permutation(np.concatenate(holdout_parts))
        train = rng.permutation(np.concatenate(train_parts))

    return train.astype(np.int64).tolist(), holdout.astype(np.int64).tolist()


def _labels_from_dataset(dataset: Dataset) -> Sequence[int] | None:
    if isinstance(dataset, Subset):
        parent_labels = _labels_from_dataset(dataset.dataset)
        if parent_labels is None:
            return None
        parent_array = np.asarray(parent_labels, dtype=np.int64)
        return parent_array[np.asarray(dataset.indices, dtype=np.int64)]
    labels = getattr(dataset, "targets", None)
    if labels is not None:
        return labels
    labels = getattr(dataset, "ys", None)
    if labels is not None:
        return labels
    try:
        return dataset["label"]  # Hugging Face Dataset
    except (KeyError, TypeError, AttributeError):
        return None


def deterministic_holdout(
    dataset: Dataset,
    *,
    fraction: float,
    seed: int,
) -> tuple[Subset, Subset, dict]:
    train_idx, holdout_idx = deterministic_split_indices(
        len(dataset),
        fraction,
        seed,
        labels=_labels_from_dataset(dataset),
    )
    metadata = {
        "source_size": int(len(dataset)),
        "train_size": int(len(train_idx)),
        "holdout_size": int(len(holdout_idx)),
        "holdout_fraction": float(fraction),
        "seed": int(seed),
        "train_indices_sha256": _indices_sha256(train_idx),
        "holdout_indices_sha256": _indices_sha256(holdout_idx),
        "stratified": _labels_from_dataset(dataset) is not None,
    }
    return Subset(dataset, train_idx), Subset(dataset, holdout_idx), metadata


def _maybe_download_names(data_root: str) -> None:
    names_url = os.environ.get(
        "NAMES_DATA_URL",
        "https://download.pytorch.org/tutorial/data.zip",
    )
    os.makedirs(data_root, exist_ok=True)
    names_dir = os.path.join(data_root, "data", "names")
    if os.path.isdir(names_dir) and any(
        fn.endswith(".txt") for fn in os.listdir(names_dir)
    ):
        return
    with urllib.request.urlopen(names_url) as response:
        content = response.read(_NAMES_MAX_DOWNLOAD_BYTES + 1)
    if len(content) > _NAMES_MAX_DOWNLOAD_BYTES:
        raise ValueError(
            "Names archive exceeds the 64 MiB download safety limit; refusing "
            "to consume unbounded memory or disk space"
        )
    root_real = os.path.realpath(data_root)
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        if len(members) > _NAMES_MAX_ARCHIVE_MEMBERS:
            raise ValueError("Names archive contains too many members")
        total_uncompressed = sum(int(member.file_size) for member in members)
        if total_uncompressed > _NAMES_MAX_UNCOMPRESSED_BYTES:
            raise ValueError(
                "Names archive exceeds the 128 MiB uncompressed safety limit"
            )
        for member in members:
            destination = os.path.realpath(os.path.join(data_root, member.filename))
            if os.path.commonpath([root_real, destination]) != root_real:
                raise ValueError(f"Unsafe path in Names archive: {member.filename}")
            unix_mode = int(member.external_attr) >> 16
            if stat.S_ISLNK(unix_mode):
                raise ValueError(
                    f"Symlink is not allowed in Names archive: {member.filename}"
                )
        archive.extractall(path=data_root)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def _make_loader(
    dataset: Dataset | None,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
    collate_fn=None,
) -> DataLoader | None:
    if dataset is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def make_dataloaders(
    args,
) -> Tuple[DataLoader, DataLoader | None, DataLoader | None, int, Dict]:
    """Build phase-aware train/validation/test loaders.

    Selection phase never returns a test loader. Retrain restores the full
    official training split and returns no validation loader. This prevents a
    validation holdout from silently changing the N used for final three-seed
    results while avoiding use of the official test split for tuning.
    """

    ds = str(args.dataset).lower().strip()
    data_root = str(args.data_root)
    selection = str(args.phase) == "selection"
    test_set: Dataset | None = None
    collate_fn = None
    meta: dict = {}

    if ds == "cifar10":
        num_classes = 10
        train_set = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=_cifar_transforms()
        )
        if not selection:
            test_set = datasets.CIFAR10(
                root=data_root,
                train=False,
                download=True,
                transform=_cifar_transforms(),
            )
    elif ds == "mnist":
        num_classes = 10
        train_set = datasets.MNIST(
            root=data_root, train=True, download=True, transform=_mnist_transforms()
        )
        if not selection:
            test_set = datasets.MNIST(
                root=data_root,
                train=False,
                download=True,
                transform=_mnist_transforms(),
            )
    elif ds == "fmnist":
        num_classes = 10
        train_set = datasets.FashionMNIST(
            root=data_root, train=True, download=True, transform=_fmnist_transforms()
        )
        if not selection:
            test_set = datasets.FashionMNIST(
                root=data_root,
                train=False,
                download=True,
                transform=_fmnist_transforms(),
            )
    elif ds == "imdb":
        num_classes = 2
        tokenizer_name = "bert-base-cased"
        max_len = int(args.max_sequence_length)
        tok_cache_dir = os.path.join(
            data_root,
            f"imdb_tokenized_{tokenizer_name.replace('/', '_')}_len{max_len}",
        )

        if os.path.isdir(tok_cache_dir):
            tokenized = load_from_disk(tok_cache_dir)
            tokenizer = BertTokenizerFast.from_pretrained(
                tokenizer_name, cache_dir=data_root, local_files_only=True
            )
        else:
            raw_dataset = load_dataset("imdb", cache_dir=data_root)
            tokenizer = BertTokenizerFast.from_pretrained(
                tokenizer_name, cache_dir=data_root
            )

            def _tok_map(example):
                return tokenizer(
                    example["text"],
                    truncation=True,
                    max_length=max_len,
                )

            tokenized = raw_dataset.map(_tok_map, batched=True)
            tokenized.save_to_disk(tok_cache_dir)
        tokenized_fingerprints = {
            split_name: str(tokenized[split_name]._fingerprint)
            for split_name in sorted(tokenized)
        }
        train_info = getattr(tokenized["train"], "info", None)
        tokenizer_commit = getattr(tokenizer, "_commit_hash", None)
        if tokenizer_commit is None:
            tokenizer_commit = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        dataset_version = getattr(train_info, "version", None)
        tokenized.set_format(type="torch", columns=["input_ids", "label"])
        train_set = tokenized["train"]
        if not selection:
            test_set = tokenized["test"]
        pad_token_id = int(tokenizer.pad_token_id or 0)
        collate_fn = partial(padded_collate_imdb, padding_idx=pad_token_id)
        meta.update(
            {
                "vocab_size": int(len(tokenizer)),
                "tokenizer_name": tokenizer_name,
                "tokenizer_commit_hash": tokenizer_commit,
                "tokenizer_vocab_sha256": _json_sha256(tokenizer.get_vocab()),
                "pad_token_id": pad_token_id,
                "pad_id": pad_token_id,
                "huggingface_dataset": {
                    "name": "imdb",
                    "builder_name": getattr(train_info, "builder_name", None),
                    "config_name": getattr(train_info, "config_name", None),
                    "version": (
                        str(dataset_version) if dataset_version is not None else None
                    ),
                    "tokenized_split_fingerprints": tokenized_fingerprints,
                    "remote_revision_pinned_by_paper": False,
                    "note": (
                        "The paper/released repository does not pin a Hugging Face "
                        "dataset or tokenizer revision; these fields record the "
                        "materialized cache used by this run."
                    ),
                },
            }
        )
    elif ds == "names":
        _maybe_download_names(data_root)
        names_dir = os.path.join(data_root, "data", "names")
        txt_files = sorted(fn for fn in os.listdir(names_dir) if fn.endswith(".txt"))
        if not txt_files:
            raise FileNotFoundError(
                f"Names dataset has no .txt files under: {names_dir}"
            )
        class_names = sorted(os.path.splitext(fn)[0] for fn in txt_files)
        class_to_id = {name: i for i, name in enumerate(class_names)}
        num_classes = len(class_names)

        all_names: list[str] = []
        all_labels: list[int] = []
        for filename in txt_files:
            class_name = os.path.splitext(filename)[0]
            path = os.path.join(names_dir, filename)
            with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    name = _clean_name_line(line)
                    if name:
                        all_names.append(name)
                        all_labels.append(int(class_to_id[class_name]))

        char2id, _ = _build_char_vocab(all_names)
        pad_id = 0
        max_len = int(args.max_sequence_length) if args.max_sequence_length else None
        full_names = NamesCharDataset(
            [_encode_name(name, char2id, max_len) for name in all_names],
            list(map(int, all_labels)),
        )
        train_idx, test_idx = deterministic_split_indices(
            len(full_names),
            0.1,
            int(args.data_split_seed),
            labels=all_labels,
        )
        train_set = Subset(full_names, train_idx)
        if not selection:
            test_set = Subset(full_names, test_idx)
        collate_fn = partial(collate_names_char, pad_id=pad_id)
        meta.update(
            {
                "vocab_size": int(len(char2id)),
                "pad_id": int(pad_id),
                "class_names": class_names,
                "names_source": {
                    "url": os.environ.get(
                        "NAMES_DATA_URL",
                        "https://download.pytorch.org/tutorial/data.zip",
                    ),
                    "raw_text_files": _named_file_hashes(names_dir, txt_files),
                    "remote_revision_pinned_by_paper": False,
                    "note": (
                        "The paper/released repository does not publish an archive "
                        "digest; this records the exact extracted text files used."
                    ),
                },
                "names_canonical_split": {
                    "paper_specified": False,
                    "seed": int(args.data_split_seed),
                    "train_fraction": 0.9,
                    "full_size": int(len(full_names)),
                    "train_size": int(len(train_idx)),
                    "test_size": int(len(test_idx)),
                    "train_indices_sha256": _indices_sha256(train_idx),
                    "test_indices_sha256": _indices_sha256(test_idx),
                },
            }
        )
    else:
        raise ValueError(f"Unsupported dataset: {ds}")

    official_train_size = int(len(train_set))
    validation_set: Dataset | None = None
    if selection:
        train_set, validation_set, split_meta = deterministic_holdout(
            train_set,
            fraction=float(args.validation_fraction),
            seed=int(args.selection_seed),
        )
        split_meta.update(
            {
                "paper_specified": False,
                "purpose": "single-seed hyperparameter selection only",
                "privacy_scope": (
                    "Validation is treated as a public tuning holdout. The selection-run "
                    "accountant protects only private_train_size samples, not this holdout. "
                    "It also does not compose multiple candidate runs."
                ),
            }
        )
        meta["validation_split"] = split_meta

    # Private loaders are converted to Poisson sampling by PrivacyEngine. The
    # deterministic generator still governs custom non-private runs.
    shuffle_train = str(args.method) == "nondp"
    train_loader = _make_loader(
        train_set,
        batch_size=int(args.batch_size),
        workers=int(args.workers),
        shuffle=shuffle_train,
        seed=int(args.seed),
        collate_fn=collate_fn,
    )
    validation_loader = _make_loader(
        validation_set,
        batch_size=int(args.batch_size_test),
        workers=int(args.workers),
        shuffle=False,
        seed=int(args.selection_seed),
        collate_fn=collate_fn,
    )
    test_loader = _make_loader(
        test_set,
        batch_size=int(args.batch_size_test),
        workers=int(args.workers),
        shuffle=False,
        seed=int(args.data_split_seed),
        collate_fn=collate_fn,
    )

    assert train_loader is not None
    meta.update(
        {
            "num_classes": int(num_classes),
            "official_train_size": official_train_size,
            "private_train_size": int(len(train_set)),
            "validation_size": (
                int(len(validation_set)) if validation_set is not None else 0
            ),
            "test_size": int(len(test_set)) if test_set is not None else 0,
            "selection_phase_test_loader_created": False if selection else None,
        }
    )
    return train_loader, validation_loader, test_loader, num_classes, meta
