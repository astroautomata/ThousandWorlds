from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import thousandworlds.make_model_tables as make_model_tables
import thousandworlds.models as models
import thousandworlds.rerun_public_models as rerun_public_models
import thousandworlds.run_model as run_model


METHODS = ("conv_decoder", "sfno")


def test_conv_decoder_presets_match_final_tuning_summaries():
    assert run_model.CONVDEC_PRESETS == {
        "multi-partial": {
            "seed_channels": 2048,
            "convs_per_stage": 1,
            "batch_size": 32,
            "num_steps": 3000,
            "lr": 3.0e-4,
            "weight_decay": 0.3,
            "hard_stop_step": 2900,
        },
        "multi-complete": {
            "seed_channels": 2048,
            "convs_per_stage": 1,
            "batch_size": 32,
            "num_steps": 3000,
            "lr": 3.0e-4,
            "weight_decay": 0.3,
            "hard_stop_step": 2700,
        },
        "single-complete": {
            "seed_channels": 1024,
            "convs_per_stage": 2,
            "batch_size": 32,
            "num_steps": 3000,
            "lr": 3.0e-4,
            "weight_decay": 0.3,
            "hard_stop_step": 2700,
        },
    }


def test_sfno_presets_match_final_tuning_summaries():
    assert run_model.SFNO_PRESETS == {
        "multi-partial": {
            "embed_dim": 768,
            "num_layers": 2,
            "batch_size": 32,
            "num_steps": 3000,
            "lr": 3.0e-4,
            "weight_decay": 0.3,
            "hard_stop_step": 3000,
        },
        "multi-complete": {
            "embed_dim": 768,
            "num_layers": 2,
            "batch_size": 32,
            "num_steps": 3000,
            "lr": 3.0e-4,
            "weight_decay": 0.3,
            "hard_stop_step": 2900,
        },
        "single-complete": {
            "embed_dim": 768,
            "num_layers": 2,
            "batch_size": 32,
            "num_steps": 3000,
            "lr": 3.0e-4,
            "weight_decay": 0.3,
            "hard_stop_step": 2600,
        },
    }


def test_baselines_are_public_and_cli_discoverable():
    assert models.ConvDecoder.__name__ == "ConvDecoder"
    assert models.SFNO.__name__ == "SFNO"
    assert set(METHODS) <= make_model_tables.PUBLIC_METHODS
    assert set(METHODS) <= set(rerun_public_models.METHODS)

    help_text = subprocess.run(
        [sys.executable, "-m", "thousandworlds.run_model", "--help"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for token in (*METHODS, "--convdec-seed-channels", "--sfno-embed-dim"):
        assert token in help_text


@pytest.mark.parametrize(
    ("method", "defaults", "overrides", "expected"),
    [
        (
            "conv_decoder",
            "CONVDEC_ARG_DEFAULTS",
            {"seed_channels": 64, "convs_per_stage": 2, "hard_stop_step": 17},
            {"convdec_seed_channels": 64, "convdec_convs_per_stage": 2, "convdec_hard_stop_step": 17},
        ),
        (
            "sfno",
            "SFNO_ARG_DEFAULTS",
            {"embed_dim": 64, "num_layers": 2, "hard_stop_step": 19},
            {"sfno_embed_dim": 64, "sfno_num_layers": 2, "sfno_hard_stop_step": 19},
        ),
    ],
)
def test_hparams_replay_bare_config_keys(method, defaults, overrides, expected):
    arg_defaults = getattr(run_model, defaults)
    args = argparse.Namespace(subset="multi-partial", **arg_defaults)
    resolver = getattr(run_model, f"_{'convdec' if method == 'conv_decoder' else 'sfno'}_hparams")

    values = resolver(args, {method: overrides})

    assert values == {**arg_defaults, **expected}


@pytest.mark.parametrize(
    ("method", "defaults", "block", "expected"),
    [
        (
            "conv_decoder",
            "CONVDEC_ARG_DEFAULTS",
            "conv_decoder",
            {"seed_channels": 2048, "convs_per_stage": 1, "batch_size": 32},
        ),
        (
            "sfno",
            "SFNO_ARG_DEFAULTS",
            "sfno",
            {"embed_dim": 768, "num_layers": 2, "batch_size": 32},
        ),
    ],
)
def test_resolved_config_uses_replayable_bare_keys(method, defaults, block, expected, tmp_path):
    args = argparse.Namespace(
        method=method,
        subset="multi-partial",
        seed=0,
        dtype="float32",
        device="cuda",
        **getattr(run_model, defaults),
    )

    cfg = run_model._resolved_config(args, out_dir=tmp_path / "results" / "models" / method, data_dir=tmp_path)

    assert expected.items() <= cfg[block].items()
    assert not any(key.startswith(("convdec_", "sfno_")) for key in cfg[block])
    assert cfg[block]["optimizer"] == "AdamW"
    assert cfg[block]["equatorial_symmetry"] is True


@pytest.mark.parametrize(
    ("method", "defaults", "arg_name", "config_value"),
    [
        ("conv_decoder", "CONVDEC_ARG_DEFAULTS", "convdec_seed_channels", 64),
        ("sfno", "SFNO_ARG_DEFAULTS", "sfno_embed_dim", 64),
    ],
)
def test_config_replay_preserves_explicit_default_valued_overrides(method, defaults, arg_name, config_value, tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("method")
    parser.add_argument("subset")
    parser.add_argument("--config", type=Path)
    arg_defaults = getattr(run_model, defaults)
    for key, default in arg_defaults.items():
        parser.add_argument(f"--{key.replace('_', '-')}", default=default)
    config = tmp_path / "config.json"
    config.write_text(
        f'{{"method": "{method}", "subset": "multi-partial", "{method}": {{"{arg_name.split("_", 1)[1]}": {config_value}}}}}',
        encoding="utf-8",
    )
    args = parser.parse_args([method, "multi-partial", "--config", str(config)])
    args._explicit_args = {arg_name}

    merged, _ = run_model._merge_config_args(args, parser)

    assert getattr(merged, arg_name) == arg_defaults[arg_name]


@pytest.mark.parametrize(
    ("method", "class_name", "hparams"),
    [
        (
            "conv_decoder",
            "ConvDecoder",
            {
                "convdec_seed_channels": 8,
                "convdec_convs_per_stage": 1,
                "convdec_batch_size": 2,
                "convdec_num_steps": 1,
                "convdec_lr": 3.0e-4,
                "convdec_weight_decay": 1.0e-4,
                "convdec_hard_stop_step": 1,
            },
        ),
        (
            "sfno",
            "SFNO",
            {
                "sfno_embed_dim": 8,
                "sfno_num_layers": 1,
                "sfno_batch_size": 2,
                "sfno_num_steps": 1,
                "sfno_lr": 3.0e-4,
                "sfno_weight_decay": 1.0e-4,
                "sfno_hard_stop_step": 1,
            },
        ),
    ],
)
@pytest.mark.parametrize("n_sim_types", [1, 2, 5])
def test_runner_uses_dynamic_gcm_slots_and_external_prediction_path(
    monkeypatch, method, class_name, hparams, n_sim_types
):
    calls = {}

    class FakeModel:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.fit_stats_ = {"steps_run": 1}

        def fit(self, *args, **kwargs):
            calls["fit"] = kwargs

        def predict(self, X, s):
            return torch.zeros((len(X), 2, 32, 64))

    monkeypatch.setattr(importlib.import_module(f"thousandworlds.models.{method}"), class_name, FakeModel)
    data = SimpleNamespace(
        X_train_std=np.zeros((2, 8), dtype=np.float32),
        X_test_std=np.zeros((1, 8), dtype=np.float32),
        s_train=np.zeros(2, dtype=np.int64),
        s_test=np.zeros(1, dtype=np.int64),
        stats=object(),
        gcm_labels=[f"gcm-{i}" for i in range(n_sim_types)],
        grid_bundle=SimpleNamespace(
            X_train=np.zeros((2, 8), dtype=np.float32),
            X_test=np.zeros((1, 8), dtype=np.float32),
            Y_train=np.zeros((2, 2, 32, 64), dtype=np.float32),
            field_mask_train=np.ones((2, 2), dtype=bool),
            raw_field_names=["surface_temperature", "v_0"],
        ),
    )
    monkeypatch.setattr(run_model, "prepare_tw_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(run_model, "average_space_grid", lambda Y, *args, **kwargs: Y)
    monkeypatch.setattr(run_model, "enforce_equatorial_symmetry_grid", lambda Y, names: calls.setdefault("path", []).append("symmetry") or Y * 2.0)
    monkeypatch.setattr(run_model, "inverse_average_space_grid", lambda Y, *args, **kwargs: calls.setdefault("path", []).append("inverse") or Y + 3.0)
    monkeypatch.setattr(run_model, f"_{'convdec' if method == 'conv_decoder' else 'sfno'}_hparams", lambda *args: hparams)
    args = argparse.Namespace(subset="multi-partial", data_dir=Path("dataset"), seed=4, dtype="float32", device="cpu")

    result = getattr(run_model, f"_run_{method}")(args)

    assert calls["fit"]["n_sim_types"] == n_sim_types
    assert result["predictions"].shape == (1, 1, 2, 32, 64)
    np.testing.assert_allclose(result["predictions"], 3.0)
    assert calls["path"] == ["symmetry", "inverse"]
    assert result["meta"]["method"] == method
    assert result["meta"]["equatorial_symmetry"] is True


def test_sfno_dependency_is_pinned_in_models_extra():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert '"torch-harmonics==0.8.0"' in pyproject.read_text(encoding="utf-8")
