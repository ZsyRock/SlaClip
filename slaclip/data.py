from __future__ import annotations

import io
import os
import re
import urllib.request
import zipfile
from typing import Dict, Tuple

import numpy as np
import torch
import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from datasets import load_dataset, load_from_disk
from transformers import BertTokenizerFast


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
        padding_value=padding_idx,
    )
    y = torch.stack([elem["label"] for elem in batch]).long()
    return x, y


def _clean_name_line(s: str) -> str:
    s = s.replace("\ufeff", "").strip()
    s = re.sub(r"[\r\n\t]+", " ", s).strip()
    return s


def _build_char_vocab(samples: list[str]):
    charset = set()
    for t in samples:
        charset.update(list(t))
    charset = sorted(list(charset))
    id2char = ["<pad>", "<unk>"] + charset
    char2id = {ch: i for i, ch in enumerate(id2char)}
    return char2id, id2char


def _encode_name(name: str, char2id: dict, max_len: int | None):
    ids = [int(char2id.get(ch, 1)) for ch in name]
    if max_len is not None and max_len > 0:
        ids = ids[: int(max_len)]
    if len(ids) == 0:
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
    xs = [b[0] for b in batch]
    ys = torch.stack([b[1] for b in batch]).long()
    x_pad = pad_sequence(xs, batch_first=True, padding_value=int(pad_id))
    return x_pad, ys


def _maybe_download_names(data_root: str) -> None:
    names_url = os.environ.get(
        "NAMES_DATA_URL",
        "https://download.pytorch.org/tutorial/data.zip",
    )
    os.makedirs(data_root, exist_ok=True)
    names_dir = os.path.join(data_root, "data", "names")
    if os.path.isdir(names_dir) and any(fn.endswith(".txt") for fn in os.listdir(names_dir)):
        return
    with urllib.request.urlopen(names_url) as resp:
        content = resp.read()
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(path=data_root)


def make_dataloaders(args) -> Tuple[DataLoader, DataLoader, int, Dict]:
    ds = str(args.dataset).lower().strip()
    data_root = str(args.data_root)

    if ds == "cifar10":
        num_classes = 10
        train_set = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=_cifar_transforms()
        )
        test_set = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=_cifar_transforms()
        )
    elif ds == "mnist":
        num_classes = 10
        train_set = datasets.MNIST(
            root=data_root, train=True, download=True, transform=_mnist_transforms()
        )
        test_set = datasets.MNIST(
            root=data_root, train=False, download=True, transform=_mnist_transforms()
        )
    elif ds == "fmnist":
        num_classes = 10
        train_set = datasets.FashionMNIST(
            root=data_root, train=True, download=True, transform=_fmnist_transforms()
        )
        test_set = datasets.FashionMNIST(
            root=data_root, train=False, download=True, transform=_fmnist_transforms()
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
                tokenizer_name,
                cache_dir=data_root,
                local_files_only=True,
            )
        else:
            raw_dataset = load_dataset("imdb", cache_dir=data_root)
            tokenizer = BertTokenizerFast.from_pretrained(
                tokenizer_name,
                cache_dir=data_root,
                local_files_only=True,
            )

            def _tok_map(ex):
                return tokenizer(
                    ex["text"],
                    truncation=True,
                    max_length=args.max_sequence_length,
                )

            tokenized = raw_dataset.map(_tok_map, batched=True)
            tokenized.save_to_disk(tok_cache_dir)
        tokenized.set_format(type="torch", columns=["input_ids", "label"])
        train_set = tokenized["train"]
        test_set = tokenized["test"]
    elif ds == "names":
        _maybe_download_names(data_root)
        names_dir = os.path.join(data_root, "data", "names")
        txt_files = sorted([fn for fn in os.listdir(names_dir) if fn.endswith(".txt")])
        if not txt_files:
            raise FileNotFoundError(f"Names dataset has no .txt files under: {names_dir}")
        class_names = [os.path.splitext(fn)[0] for fn in txt_files]
        class_names_sorted = sorted(class_names)
        cls2id = {c: i for i, c in enumerate(class_names_sorted)}
        num_classes = len(class_names_sorted)

        all_names: list[str] = []
        all_labels: list[int] = []
        for fn in txt_files:
            cls_name = os.path.splitext(fn)[0]
            y = int(cls2id[cls_name])
            path = os.path.join(names_dir, fn)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    nm = _clean_name_line(line)
                    if not nm:
                        continue
                    all_names.append(nm)
                    all_labels.append(y)

        char2id, _id2char = _build_char_vocab(all_names)
        pad_id = 0
        max_len = int(args.max_sequence_length) if args.max_sequence_length else None
        xs = [_encode_name(n, char2id, max_len) for n in all_names]
        ys = list(map(int, all_labels))

        rng = np.random.RandomState(int(args.seed))
        idx = np.arange(len(ys))
        rng.shuffle(idx)
        n_total = len(idx)
        n_train = max(1, min(n_total - 1, int(round(0.9 * n_total))))
        train_idx = idx[:n_train].tolist()
        test_idx = idx[n_train:].tolist()

        x_train = [xs[i] for i in train_idx]
        y_train = [ys[i] for i in train_idx]
        x_test = [xs[i] for i in test_idx]
        y_test = [ys[i] for i in test_idx]

        train_set = NamesCharDataset(x_train, y_train)
        test_set = NamesCharDataset(x_test, y_test)
    else:
        raise ValueError(f"Unsupported dataset: {ds}")

    if ds == "imdb":
        train_loader = DataLoader(
            train_set,
            num_workers=args.workers,
            batch_size=args.batch_size,
            collate_fn=padded_collate_imdb,
            pin_memory=torch.cuda.is_available(),
            shuffle=True,
            drop_last=False,
        )
        test_loader = DataLoader(
            test_set,
            num_workers=args.workers,
            batch_size=args.batch_size_test,
            collate_fn=padded_collate_imdb,
            pin_memory=torch.cuda.is_available(),
            shuffle=False,
            drop_last=False,
        )
        meta = {"vocab_size": len(tokenizer), "num_classes": num_classes}
        return train_loader, test_loader, num_classes, meta

    if ds == "names":
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=lambda b: collate_names_char(b, pad_id=pad_id),
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size_test,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            collate_fn=lambda b: collate_names_char(b, pad_id=pad_id),
        )
        meta = {
            "vocab_size": int(len(char2id)),
            "pad_id": int(pad_id),
            "num_classes": int(num_classes),
            "class_names": class_names_sorted,
        }
        return train_loader, test_loader, num_classes, meta

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    meta = {}
    return train_loader, test_loader, num_classes, meta
