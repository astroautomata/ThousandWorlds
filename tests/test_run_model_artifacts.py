from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import thousandworlds.run_model as run_model


@pytest.mark.parametrize("subset", ("multi-partial", "multi-complete"))
def test_multi_run_persists_shared_planet_predictions_in_requested_output(
    monkeypatch,
    tmp_path,
    subset,
):
    test_ids = np.array([101, 102, 103], dtype=np.int32)
    shared_ids = np.array([103, 101], dtype=np.int32)
    field_names = ["surface_temperature", "olr"]
    predictions = np.arange(12, dtype=np.float32).reshape(2, 3, 2, 1, 1)
    point_predictions = np.arange(6, dtype=np.float32).reshape(1, 3, 2, 1, 1) + 100
    data = SimpleNamespace(grid_bundle=SimpleNamespace(test_ids=test_ids, field_names=field_names))
    score_paths = {}

    monkeypatch.setattr(
        run_model,
        "_run_method",
        lambda args: {
            "data": data,
            "predictions": predictions,
            "point_predictions": point_predictions,
        },
    )
    monkeypatch.setattr(
        run_model.tw,
        "load",
        lambda **kwargs: SimpleNamespace(test_ids=shared_ids),
    )

    def fake_score(path, *, protocol, point_predictions_path, **kwargs):
        score_paths[protocol] = (Path(path), Path(point_predictions_path))
        return {"protocol": protocol}

    monkeypatch.setattr(run_model, "score_saved_submission", fake_score)
    out_dir = tmp_path / "requested"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_model",
            "train_mean",
            subset,
            "--data-dir",
            str(tmp_path / "dataset"),
            "--out-dir",
            str(out_dir),
        ],
    )

    run_model.main()

    shared_path = out_dir / "predictions_shared_planets.npz"
    shared_mean_path = out_dir / "predictions_mean_shared_planets.npz"
    assert score_paths["shared_planets"] == (shared_path, shared_mean_path)
    with np.load(shared_path, allow_pickle=False) as saved:
        assert saved.files == ["predictions", "simulation_id", "field_names"]
        np.testing.assert_array_equal(saved["predictions"], predictions[:, [2, 0]])
        np.testing.assert_array_equal(saved["simulation_id"], shared_ids)
        np.testing.assert_array_equal(saved["field_names"], field_names)
    with np.load(shared_mean_path, allow_pickle=False) as saved:
        assert saved.files == ["predictions", "simulation_id", "field_names"]
        np.testing.assert_array_equal(saved["predictions"], point_predictions[:, [2, 0]])
        np.testing.assert_array_equal(saved["simulation_id"], shared_ids)
        np.testing.assert_array_equal(saved["field_names"], field_names)
