import io
import zipfile

import numpy as np
import pytest

import slaclip.data as data_module
from slaclip.data import _named_file_hashes, deterministic_split_indices


def test_deterministic_split_is_stable_disjoint_and_stratified():
    labels = np.repeat(np.arange(4), 25)
    train_a, validation_a = deterministic_split_indices(100, 0.1, 2026, labels=labels)
    train_b, validation_b = deterministic_split_indices(100, 0.1, 2026, labels=labels)
    assert train_a == train_b
    assert validation_a == validation_b
    assert len(train_a) == 90
    assert len(validation_a) == 10
    assert set(train_a).isdisjoint(validation_a)
    assert set(train_a) | set(validation_a) == set(range(100))
    assert set(labels[validation_a]) == {0, 1, 2, 3}


def test_data_split_seed_is_independent_from_training_seed_by_construction():
    first = deterministic_split_indices(1000, 0.1, 2026)
    second = deterministic_split_indices(1000, 0.1, 2026)
    different_convention = deterministic_split_indices(1000, 0.1, 2027)
    assert first == second
    assert first != different_convention


def test_named_file_hash_manifest_is_stable_and_content_sensitive(tmp_path):
    first = tmp_path / "b.txt"
    second = tmp_path / "a.txt"
    first.write_bytes(b"beta\n")
    second.write_bytes(b"alpha\n")

    before = _named_file_hashes(str(tmp_path), [first.name, second.name])
    repeated = _named_file_hashes(str(tmp_path), [second.name, first.name])
    assert before == repeated
    assert list(before["per_file"]) == ["a.txt", "b.txt"]

    first.write_bytes(b"changed\n")
    after = _named_file_hashes(str(tmp_path), [first.name, second.name])
    assert before["manifest_sha256"] != after["manifest_sha256"]


def _zip_bytes(filename: str, payload: bytes) -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(result, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(filename, payload)
    return result.getvalue()


def test_names_download_refuses_archive_over_uncompressed_disk_cap(
    monkeypatch, tmp_path
):
    content = _zip_bytes("data/names/example.txt", b"x" * 128)
    monkeypatch.setattr(data_module, "_NAMES_MAX_UNCOMPRESSED_BYTES", 64)
    monkeypatch.setattr(
        data_module.urllib.request,
        "urlopen",
        lambda _url: io.BytesIO(content),
    )

    with pytest.raises(ValueError, match="uncompressed safety limit"):
        data_module._maybe_download_names(str(tmp_path))


def test_names_download_refuses_path_traversal(monkeypatch, tmp_path):
    content = _zip_bytes("../outside.txt", b"unsafe")
    monkeypatch.setattr(
        data_module.urllib.request,
        "urlopen",
        lambda _url: io.BytesIO(content),
    )

    with pytest.raises(ValueError, match="Unsafe path"):
        data_module._maybe_download_names(str(tmp_path))
    assert not (tmp_path.parent / "outside.txt").exists()
