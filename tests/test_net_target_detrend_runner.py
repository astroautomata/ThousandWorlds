import argparse
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import thousandworlds.run_model as run_model
from thousandworlds.models._target_detrend import TARGET_DETREND_CFG


METHODS = {
    "coord_mlp": (
        "CoordMLP",
        "_coord_mlp_hparams",
        {
            "coord_hidden_width": 8,
            "coord_num_layers": 1,
            "coord_activation": "tanh",
            "coord_batch_size": 2,
            "coord_predict_chunk_size": 16,
            "coord_num_steps": 1,
            "coord_lr": 1.0e-4,
            "coord_weight_decay": 0.0,
            "coord_hard_stop_step": 1,
        },
    ),
    "coord_deeponet": (
        "CoordDeepONet",
        "_coord_deeponet_hparams",
        {
            "deeponet_rank": 2,
            "deeponet_branch_hidden_width": 8,
            "deeponet_trunk_hidden_width": 8,
            "deeponet_branch_num_layers": 1,
            "deeponet_trunk_num_layers": 1,
            "deeponet_activation": "silu",
            "deeponet_batch_size": 2,
            "deeponet_predict_chunk_size": 16,
            "deeponet_num_steps": 1,
            "deeponet_lr": 1.0e-4,
            "deeponet_weight_decay": 0.0,
            "deeponet_hard_stop_step": 1,
        },
    ),
    "conv_decoder": (
        "ConvDecoder",
        "_convdec_hparams",
        {
            "convdec_seed_channels": 8,
            "convdec_convs_per_stage": 1,
            "convdec_batch_size": 2,
            "convdec_num_steps": 1,
            "convdec_lr": 1.0e-4,
            "convdec_weight_decay": 0.0,
            "convdec_hard_stop_step": 1,
        },
    ),
    "sfno": (
        "SFNO",
        "_sfno_hparams",
        {
            "sfno_embed_dim": 8,
            "sfno_num_layers": 1,
            "sfno_batch_size": 2,
            "sfno_num_steps": 1,
            "sfno_lr": 1.0e-4,
            "sfno_weight_decay": 0.0,
            "sfno_hard_stop_step": 1,
        },
    ),
}


@pytest.mark.parametrize("method", METHODS)
def test_field_net_runners_train_on_residuals_and_restore_trend(monkeypatch, method):
    class FakeModel:
        fit_stats_ = {"steps_run": 1}

        def __init__(self, **kwargs):
            pass

        def fit(self, X, s, Y, **kwargs):
            np.testing.assert_allclose(Y, 3.0)

        def predict(self, X, s):
            return torch.full((len(X), 2, 32, 64), 2.0)

    class FakeTransform:
        def apply_grid(self, residual, X, s):
            calls.append("restore")
            return np.asarray(residual) + 7.0

    calls = []
    class_name, hparams_name, hparams = METHODS[method]
    monkeypatch.setattr(importlib.import_module(f"thousandworlds.models.{method}"), class_name, FakeModel)
    monkeypatch.setattr(run_model, hparams_name, lambda *args: hparams)
    monkeypatch.setattr(
        run_model,
        "prepare_tw_data",
        lambda *args, **kwargs: SimpleNamespace(
            X_train_std=np.zeros((2, 8), dtype=np.float32),
            X_test_std=np.zeros((1, 8), dtype=np.float32),
            s_train=np.zeros(2, dtype=np.int64),
            s_test=np.zeros(1, dtype=np.int64),
            stats=object(),
            gcm_labels=["gcm"],
            grid_bundle=SimpleNamespace(
                X_train=np.zeros((2, 8), dtype=np.float32),
                X_test=np.zeros((1, 8), dtype=np.float32),
                Y_train=np.full((2, 2, 32, 64), 10.0, dtype=np.float32),
                field_mask_train=np.ones((2, 2), dtype=bool),
                raw_field_names=["surface_temperature", "v_0"],
            ),
        ),
    )
    monkeypatch.setattr(run_model, "average_space_grid", lambda Y, *args, **kwargs: Y)
    monkeypatch.setattr(
        run_model,
        "_prepare_net_target",
        lambda args, data, Y: (Y - 7.0, FakeTransform(), {"enabled": True}),
    )
    monkeypatch.setattr(run_model, "enforce_equatorial_symmetry_grid", lambda Y, names: Y)
    monkeypatch.setattr(run_model, "inverse_average_space_grid", lambda Y, *args, **kwargs: Y)
    args = argparse.Namespace(
        subset="multi-partial",
        data_dir=Path("dataset"),
        seed=0,
        dtype="float32",
        device="cpu",
        target_detrend=True,
    )

    result = getattr(run_model, f"_run_{method}")(args)

    np.testing.assert_allclose(result["predictions"], 9.0)
    assert calls == ["restore"]
    assert result["meta"]["target_detrend"] == {"enabled": True}


def test_resolved_config_records_the_only_transform_override(tmp_path):
    args = argparse.Namespace(
        method="coord_mlp",
        subset="multi-partial",
        seed=0,
        dtype="float32",
        device="cuda",
        target_detrend=True,
        **run_model.COORD_MLP_ARG_DEFAULTS,
    )

    config = run_model._resolved_config(args, out_dir=tmp_path / "out", data_dir=tmp_path)

    assert config["target_detrend"] == TARGET_DETREND_CFG
