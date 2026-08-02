import json

import pytest

from slaclip.logging_utils import (
    assert_output_paths_available,
    write_run_json,
)


def test_existing_run_is_not_silently_overwritten(tmp_path):
    result = tmp_path / "run.json"
    result.write_text("existing")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        assert_output_paths_available([result], overwrite=False)
    assert_output_paths_available([result], overwrite=True)


def test_run_json_contains_metadata_epochs_and_final(tmp_path):
    result = tmp_path / "run.json"
    write_run_json(
        result,
        metadata={"git": {"slaclip": {"commit": "abc"}}},
        records=[{"epoch": 1, "epsilon": 2.0, "validation_accuracy": float("nan")}],
        final={"epsilon": 2.0},
    )
    payload = json.loads(result.read_text())
    assert payload["schema_version"] == 2
    assert payload["metadata"]["git"]["slaclip"]["commit"] == "abc"
    assert payload["epochs"][0]["epsilon"] == 2.0
    assert payload["epochs"][0]["validation_accuracy"] is None
    assert payload["final"]["epsilon"] == 2.0
