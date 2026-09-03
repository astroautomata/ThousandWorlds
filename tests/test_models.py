import importlib
import inspect
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch

import thousandworlds.models as models
import thousandworlds.run_model as run_model
import thousandworlds.rerun_public_models as rerun_public_models
from thousandworlds.models._torch_kernels import build_design_matrix
from thousandworlds.models._gplfr_core import GPLFRCore
from thousandworlds.models._gplfr_weighting import retrieve_field_group_index
from thousandworlds.models._common import enforce_equatorial_symmetry_grid, equal_group_normalized_rmse_grid, masked_mean_grid
from thousandworlds.models.pca_mlp import PCAMLP, _ScoreMLP, _equal_group_mean
from thousandworlds.models.pca_ridge import PCARidge, fit_latent_ridge


def test_models_surface_smoke_and_knn_path():
    assert models.KNN is not None
    assert models.TrainMean is not None
    assert models.GPLFR is not None
    assert models.PPCAICM is not None
    assert models.PCAMLP is not None
    assert models.PCARidge is not None
    assert models.PCAGBT is not None
    assert models.CoordDeepONet is not None
    assert models.CoordMLP is not None

    model = models.KNN()
    X_train = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    Y_train = np.array([
        [[[10.0]], [[100.0]]],
        [[[20.0]], [[200.0]]],
        [[[30.0]], [[300.0]]],
    ], dtype=np.float32)
    field_mask = np.array([[True, False], [True, True], [True, True]])

    model.fit(X_train, Y_train, k_candidates=[2], field_mask=field_mask)
    pred = model.predict(np.array([[0.05, 0.05], [1.95, 2.05]], dtype=np.float32), k=2)

    assert pred.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(pred[:, :, 0, 0], [[15.0, 200.0], [25.0, 250.0]])


def test_knn_ignores_missing_neighbor_fields_and_falls_back_to_field_mean():
    model = models.KNN()
    X_train = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    Y_train = np.array([
        [[[10.0]], [[np.nan]]],
        [[[20.0]], [[np.nan]]],
        [[[30.0]], [[300.0]]],
    ], dtype=np.float32)
    field_mask = np.array([[True, False], [True, False], [True, True]])

    model.fit(X_train, Y_train, k_candidates=[2], field_mask=field_mask)
    pred = model.predict(np.array([[0.1]], dtype=np.float32), k=2)

    assert np.isfinite(pred).all()
    np.testing.assert_allclose(pred[0, :, 0, 0], [15.0, 300.0])


def test_knn_cv_objective_equal_weights_normalized_variable_groups():
    target = np.zeros((1, 3, 32, 64), dtype=np.float32)
    pred = np.zeros_like(target)
    pred[:, 0] = 2.0
    pred[:, 1] = 20.0
    pred[:, 2] = 4.0

    score = equal_group_normalized_rmse_grid(
        pred,
        target,
        ["temperature_0", "temperature_1", "u_0"],
        np.array([1.0, 10.0, 1.0], dtype=np.float32),
        np.ones((1, 3), dtype=bool),
    )

    np.testing.assert_allclose(score, 3.0, atol=1.0e-6)


def test_knn_plain_cv_fits_preprocessing_inside_each_fold(monkeypatch):
    n = 10
    grid = SimpleNamespace(
        Y_train=np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1),
        X_train=np.arange(n, dtype=np.float32).reshape(n, 1),
        X_test=np.array([[10.0]], dtype=np.float32),
        field_mask_train=np.ones((n, 1), dtype=bool),
        raw_field_names=["surface_temperature"],
    )
    data = SimpleNamespace(
        grid_bundle=grid,
        X_train_std=grid.X_train.copy(),
        X_test_std=grid.X_test.copy(),
        s_train=np.zeros(n, dtype=np.int64),
        s_test=np.zeros(1, dtype=np.int64),
        gcm_labels=["only"],
        stats=object(),
    )
    calls = []

    def prepare_fold(given, train_idx):
        calls.append(np.asarray(train_idx))
        return given

    monkeypatch.setattr(run_model, "prepare_tw_data", lambda *args, **kwargs: data)
    monkeypatch.setattr(run_model, "prepare_tw_fold", prepare_fold)
    monkeypatch.setattr(
        run_model,
        "average_space_grid",
        lambda values, *args, **kwargs: np.asarray(values),
    )
    monkeypatch.setattr(
        run_model.tw,
        "inverse_preprocess_outputs_grid",
        lambda values, *args, **kwargs: np.asarray(values),
    )
    monkeypatch.setattr(
        run_model,
        "field_rmse_scale_grid",
        lambda *args, **kwargs: np.ones(1, dtype=np.float32),
    )
    monkeypatch.setattr(
        run_model,
        "equal_group_normalized_rmse_grid",
        lambda *args, **kwargs: 0.0,
    )
    monkeypatch.setattr(
        run_model,
        "enforce_equatorial_symmetry_grid",
        lambda values, *args, **kwargs: values,
    )
    args = SimpleNamespace(
        subset="multi-partial",
        data_dir=Path("unused"),
        k="1",
        gcm_penalty="0",
        best_k=None,
        best_gcm_penalty=None,
        n_folds=5,
        seed=0,
    )

    run_model._run_knn(args)

    expected = run_model._kfold_indices(n, 5, 0)
    assert len(calls) == 5
    for actual, (train_idx, _) in zip(calls, expected, strict=True):
        np.testing.assert_array_equal(actual, train_idx)


def test_equatorial_symmetry_grid_enforces_scalar_symmetry_and_v_antisymmetry():
    Y = np.array([[
        [[1.0], [3.0], [5.0], [7.0]],
        [[2.0], [4.0], [6.0], [8.0]],
    ]], dtype=np.float32)

    out = enforce_equatorial_symmetry_grid(Y, ["temperature_0", "v_0"])

    np.testing.assert_allclose(out[0, 0, :, 0], [4.0, 4.0, 4.0, 4.0])
    np.testing.assert_allclose(out[0, 1, :, 0], [-3.0, -1.0, 1.0, 3.0])


def test_train_mean_masked_grid_mean_excludes_nans():
    Y_train = np.array([
        [[[10.0]], [[np.nan]]],
        [[[20.0]], [[np.nan]]],
        [[[30.0]], [[300.0]]],
    ], dtype=np.float32)
    field_mask = np.array([[True, False], [True, False], [True, True]])

    mean = masked_mean_grid(Y_train, field_mask)

    assert np.isfinite(mean).all()
    np.testing.assert_allclose(mean[:, 0, 0], [20.0, 300.0])


def test_run_model_help_lists_supported_methods():
    result = subprocess.run(
        [sys.executable, "-m", "thousandworlds.run_model", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "{train_mean,knn,pca_ridge,pca_mlp,pca_gbt,ppca_icm,gplfr,coord_mlp,coord_deeponet,conv_decoder,sfno}" in result.stdout
    assert "1,2,3,5,10" in result.stdout
    assert "0.0,0.3,1.0,3.0,10.0" in result.stdout
    assert "--lambda-reg" in result.stdout
    assert "--cv-latent-dim" in result.stdout
    assert "--cv-lambda" in result.stdout


def test_gplfr_resolver_defaults_and_config_replay():
    args = run_model.argparse.Namespace(
        subset="multi-partial",
        latent_dim=60,
        gplfr_num_training_steps=3000,
        gplfr_inverse_temperature=0.25,
        gplfr_latent_nugget=0.03,
        gplfr_n_samples=2,
        gplfr_kernel="rbf",
        _explicit_args=set(),
    )
    hparams = run_model._gplfr_hparams(args)
    assert hparams["latent_dim"] == 150
    assert hparams["num_training_steps"] == 115
    assert hparams["inverse_temperature"] == 0.1
    assert hparams["latent_nugget"] == 0.1
    assert hparams["lr_Z"] == 0.1
    assert hparams["lr_global"] == 0.3
    assert hparams["variable_weights"] == "fixed"
    assert hparams["output_coregionalization"] == "none"
    assert hparams["n_samples"] == 64
    assert hparams["kernel"] == "matern52"

    args._explicit_args = {"latent_dim", "gplfr_num_training_steps", "gplfr_inverse_temperature", "gplfr_latent_nugget", "gplfr_n_samples", "gplfr_kernel"}
    hparams = run_model._gplfr_hparams(args)
    assert hparams["latent_dim"] == 60
    assert hparams["num_training_steps"] == 3000
    assert hparams["inverse_temperature"] == 0.25
    assert hparams["latent_nugget"] == 0.03
    assert hparams["n_samples"] == 2
    assert hparams["kernel"] == "rbf"

    replay = run_model._gplfr_hparams(args, {"gplfr": {"kernel": "matern32", "latent_dim": 4, "num_training_steps": 29, "inverse_temperature": 0.5, "latent_nugget": 0.01, "n_samples": 3, "variable_weights": "fixed", "output_coregionalization": "none", "optimizer": {"lr_Z": 0.2, "lr_global": 0.4}}})
    assert replay["latent_dim"] == 4
    assert replay["num_training_steps"] == 29
    assert replay["inverse_temperature"] == 0.5
    assert replay["latent_nugget"] == 0.01
    assert replay["n_samples"] == 3
    assert replay["kernel"] == "matern32"
    assert replay["variable_weights"] == "fixed"
    assert replay["output_coregionalization"] == "none"
    assert replay["lr_Z"] == 0.2
    assert replay["lr_global"] == 0.4


@pytest.mark.parametrize(
    ("subset", "kernel", "steps", "beta", "nugget"),
    (
        ("multi-partial", "matern52", 115, 0.1, 0.1),
        ("multi-complete", "matern52", 120, 0.1, 0.03),
        ("single-complete", "matern32", 30, 0.03, 0.03),
    ),
)
def test_gplfr_subset_presets_match_published_configs(
    subset, kernel, steps, beta, nugget
):
    args = run_model.argparse.Namespace(subset=subset, _explicit_args=set())

    hparams = run_model._gplfr_hparams(args)

    assert (hparams["kernel"], hparams["num_training_steps"]) == (kernel, steps)
    assert hparams["inverse_temperature"] == beta
    assert hparams["latent_nugget"] == nugget
    assert hparams["n_samples"] == 64
    assert (hparams["variable_weights"], hparams["output_coregionalization"]) == (
        "fixed",
        "none",
    )


def test_gplfr_weighting_accepts_public_radiation_field_names():
    assert retrieve_field_group_index("asr") == retrieve_field_group_index("asr_cloudy")
    assert retrieve_field_group_index("olr") == retrieve_field_group_index("olr_cloudy")


def test_ppca_icm_tuned_preset_allows_explicit_cli_overrides():
    args = run_model.argparse.Namespace(
        subset="multi-partial",
        ppca_icm_preset="tuned",
        latent_dim=120,
        kernel="matern52",
        kernel_mode="shared",
        ppca_iters=50,
        gp_steps=4000,
        gp_lr=1.0e-3,
        n_samples=64,
        hard_stop_step=1300,
        ell_init="4.0",
        _explicit_args={"latent_dim", "gp_lr", "hard_stop_step", "ell_init"},
    )

    hparams = run_model._ppca_icm_hparams(args)

    assert hparams["latent_dim"] == 120
    assert hparams["gp_lr"] == 1.0e-3
    assert hparams["hard_stop_step"] == 1300
    assert hparams["ell_init"] == 4.0
    assert hparams["gp_steps"] == 4000


def test_ppca_icm_resolved_config_replays_frozen_config(tmp_path):
    frozen = {
        "ppca_icm": {
            "latent_dim": 4,
            "kernel": "matern32",
            "kernel_mode": "shared",
            "ppca_iters": 2,
            "gp_steps": 2,
            "gp_lr": 0.003,
            "n_samples": 2,
            "hard_stop_step": 2,
            "ell_init": 4.0,
        }
    }
    args = run_model.argparse.Namespace(
        method="ppca_icm",
        subset="multi-partial",
        seed=0,
        dtype="float32",
        device="cuda",
        ppca_icm_preset="tuned",
        _config=frozen,
    )

    assert frozen["ppca_icm"].items() <= run_model._resolved_config(
        args, out_dir=tmp_path / "out", data_dir=tmp_path
    )["ppca_icm"].items()


@pytest.mark.parametrize("kernel_mode", ["shared", "per_pc"])
def test_ppca_icm_zero_hard_stop_supports_finite_predictions(kernel_mode):
    torch.manual_seed(0)
    X = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    s = torch.tensor([0, 1, 0, 1])
    Y = torch.randn((4, 3, 2), dtype=torch.float64)
    field_mask = torch.ones((4, 2), dtype=torch.bool)
    sh_mask = torch.ones((3, 2), dtype=torch.bool)
    model = models.PPCAICM(
        latent_dim=1,
        kernel_mode=kernel_mode,
        dtype=torch.float64,
        device="cpu",
    )

    model.fit(
        X,
        s,
        Y,
        field_mask=field_mask,
        sh_mask=sh_mask,
        ppca_iters=2,
        gp_steps=2,
        hard_stop_step=0,
    )

    assert model.gp_fit_stats_["shared"]["steps_run"] == 0
    if kernel_mode == "per_pc":
        assert model.gp_fit_stats_["per_pc"]["steps_run"] == 0
    prediction = model.predict(X[:2], s[:2])
    samples = model.predict_samples(
        X[:2],
        s[:2],
        n_post_samples=3,
        seed=0,
    )
    assert prediction.shape == (2, 3, 2)
    assert samples.shape == (3, 2, 3, 2)
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(samples).all()


def test_ppca_icm_rejects_zero_schedule_even_with_zero_hard_stop():
    model = models.PPCAICM(
        latent_dim=1,
        kernel_mode="shared",
        dtype=torch.float64,
        device="cpu",
    )

    with pytest.raises(ValueError, match="gp_steps"):
        model.fit(
            torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            torch.zeros(2, dtype=torch.long),
            torch.randn((2, 2, 1), dtype=torch.float64),
            field_mask=torch.ones((2, 1), dtype=torch.bool),
            sh_mask=torch.ones((2, 1), dtype=torch.bool),
            ppca_iters=1,
            gp_steps=0,
            hard_stop_step=0,
        )


def test_pca_mlp_zero_hard_stop_supports_finite_prediction():
    torch.manual_seed(0)
    X = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    s = torch.tensor([0, 1, 0, 1])
    Y = torch.randn((4, 3, 2), dtype=torch.float64)
    field_mask = torch.ones((4, 2), dtype=torch.bool)
    sh_mask = torch.ones((3, 2), dtype=torch.bool)
    model = PCAMLP(
        latent_dim=1,
        hidden_width=4,
        dtype=torch.float64,
        device="cpu",
    )

    model.fit(
        X,
        s,
        Y,
        field_mask=field_mask,
        sh_mask=sh_mask,
        ppca_iters=2,
        num_steps=2,
        hard_stop_step=0,
    )

    assert model.mlp_fit_stats_["steps_run"] == 0
    prediction = model.predict(X[:2], s[:2])
    assert prediction.shape == (2, 3, 2)
    assert torch.isfinite(prediction).all()


def test_pca_mlp_rejects_zero_schedule_even_with_zero_hard_stop():
    model = PCAMLP(
        latent_dim=1,
        hidden_width=4,
        dtype=torch.float64,
        device="cpu",
    )

    with pytest.raises(ValueError, match="num_steps"):
        model.fit(
            torch.tensor([[0.0], [1.0]], dtype=torch.float64),
            torch.zeros(2, dtype=torch.long),
            torch.randn((2, 2, 1), dtype=torch.float64),
            field_mask=torch.ones((2, 1), dtype=torch.bool),
            sh_mask=torch.ones((2, 1), dtype=torch.bool),
            ppca_iters=1,
            num_steps=0,
            hard_stop_step=0,
        )


def test_gplfr_public_surface_uses_core_module():
    from thousandworlds.models.gplfr import GPLFR

    assert models.GPLFR.__name__ == GPLFR.__name__ == "GPLFR"
    assert models.GPLFR.__module__.endswith("models.gplfr")
    assert GPLFRCore.__name__ == "GPLFRCore"
    assert GPLFR(latent_dim=2, kernel="rbf", device="cpu").kernel == "rbf"


def test_gplfr_exposes_training_checkpoint_callback():
    from thousandworlds.models.gplfr import GPLFR

    assert "checkpoint_callback" in inspect.signature(GPLFR.fit).parameters
    assert "checkpoint_callback" in inspect.signature(GPLFRCore.fit).parameters
    calls = []
    model = GPLFR(
        latent_dim=1,
        num_training_steps=2,
        log_every=1,
        variable_weights="fixed",
        output_coregionalization="none",
        dtype=torch.float64,
        device="cpu",
    )
    model.fit(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        torch.zeros(3, dtype=torch.long),
        torch.randn(3, 4, 1),
        field_mask=torch.ones(3, 1, dtype=torch.bool),
        sh_mask=torch.ones(4, 1, dtype=torch.bool),
        field_names=["surface_temperature"],
        n_sim_types=1,
        verbose=False,
        checkpoint_callback=lambda core, guide, step: calls.append(step),
    )
    assert calls == [0, 1]


def test_gplfr_exponential_decay_uses_requested_rates_for_each_update(monkeypatch):
    import pyro
    from thousandworlds.models.gplfr import GPLFR

    rates: dict[int, list[float]] = {}
    captured_svi = []
    adam_step = torch.optim.Adam.step

    def capture_rates(optimizer, *args, **kwargs):
        rates.setdefault(id(optimizer), []).append(optimizer.param_groups[0]["lr"])
        return adam_step(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.Adam, "step", capture_rates)
    model = GPLFR(
        latent_dim=1,
        num_training_steps=3,
        lr_gamma=0.5,
        log_every=1,
        variable_weights="fixed",
        output_coregionalization="none",
        dtype=torch.float64,
        device="cpu",
    )
    model.fit(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        torch.zeros(3, dtype=torch.long),
        torch.tensor([[[0.1], [0.2], [0.3], [0.4]], [[0.2], [0.3], [0.4], [0.5]], [[0.3], [0.4], [0.5], [0.6]]]),
        field_mask=torch.ones(3, 1, dtype=torch.bool),
        sh_mask=torch.ones(4, 1, dtype=torch.bool),
        field_names=["surface_temperature"],
        n_sim_types=1,
        verbose=False,
        diagnostics_callback=lambda core, guide, svi, step, loss, Y, X, s: captured_svi.append(svi),
    )

    observed = {
        pyro.get_param_store().param_name(param).rsplit(".", 1)[-1]: rates[id(scheduler.optimizer)]
        for param, scheduler in captured_svi[-1].optim.optim_objs.items()
    }
    latent = [history for name, history in observed.items() if name in {"Z_T", "U_T"}]
    global_ = [history for name, history in observed.items() if name not in {"Z_T", "U_T"}]
    assert latent and global_
    assert all(history == [0.1, 0.05, 0.025] for history in latent)
    assert all(history == [0.3, 0.15, 0.075] for history in global_)
    assert model.fit_stats_["lr_gamma"] == 0.5


def test_gplfr_default_decay_matches_explicit_one():
    from thousandworlds.models.gplfr import GPLFR

    X = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    s = torch.zeros(3, dtype=torch.long)
    Y = torch.tensor([[[0.1], [0.2], [0.3], [0.4]], [[0.2], [0.3], [0.4], [0.5]], [[0.3], [0.4], [0.5], [0.6]]])
    fit_kwargs = {
        "field_mask": torch.ones(3, 1, dtype=torch.bool),
        "sh_mask": torch.ones(4, 1, dtype=torch.bool),
        "field_names": ["surface_temperature"],
        "n_sim_types": 1,
        "seed": 7,
        "verbose": False,
    }
    model_kwargs = {
        "latent_dim": 1,
        "num_training_steps": 3,
        "log_every": 1,
        "variable_weights": "fixed",
        "output_coregionalization": "none",
        "dtype": torch.float64,
        "device": "cpu",
    }
    default = GPLFR(**model_kwargs)
    explicit = GPLFR(**model_kwargs, lr_gamma=1.0)

    default.fit(X, s, Y, **fit_kwargs)
    explicit.fit(X, s, Y, **fit_kwargs)

    assert default.fit_stats_["loss_history"] == explicit.fit_stats_["loss_history"]
    for name in default.model_.posterior_samples_:
        torch.testing.assert_close(default.model_.posterior_samples_[name], explicit.model_.posterior_samples_[name])
    torch.testing.assert_close(default.predict(X, s), explicit.predict(X, s))


def test_gplfr_selects_direct_or_whitened_latent_parameterization():
    from thousandworlds.models.gplfr import GPLFR

    for parameterization, latent_name in (("direct_z", "Z_T"), ("whitened", "U_T")):
        model = GPLFR(
            latent_dim=1,
            latent_parameterization=parameterization,
            num_training_steps=1,
            variable_weights="fixed",
            output_coregionalization="none",
            dtype=torch.float64,
            device="cpu",
        )
        model.fit(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            torch.zeros(3, dtype=torch.long),
            torch.randn(3, 4, 1),
            field_mask=torch.ones(3, 1, dtype=torch.bool),
            sh_mask=torch.ones(4, 1, dtype=torch.bool),
            field_names=["surface_temperature"],
            n_sim_types=1,
            verbose=False,
        )
        assert model.model_._latent_param_mode == parameterization
        assert latent_name in model.model_.posterior_samples_

    with pytest.raises(ValueError, match="latent_parameterization"):
        GPLFR(latent_parameterization="invalid")


def test_gplfr_forwards_optimizer_diagnostics_callback():
    from thousandworlds.models.gplfr import GPLFR

    calls = []
    model = GPLFR(
        latent_dim=1,
        num_training_steps=2,
        variable_weights="fixed",
        output_coregionalization="none",
        dtype=torch.float64,
        device="cpu",
    )
    model.fit(
        torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        torch.zeros(3, dtype=torch.long),
        torch.randn(3, 4, 1),
        field_mask=torch.ones(3, 1, dtype=torch.bool),
        sh_mask=torch.ones(4, 1, dtype=torch.bool),
        field_names=["surface_temperature"],
        n_sim_types=1,
        verbose=False,
        diagnostics_callback=lambda core, guide, svi, step, loss, Y, X, s: calls.append(
            (step, loss, tuple(Y.shape), tuple(X.shape), tuple(s.shape))
        ),
    )
    assert [call[0] for call in calls] == [0, 1]
    assert all(np.isfinite(call[1]) for call in calls)
    assert calls[0][2:] == ((3, 4, 1), (3, 2), (3,))


def test_rerun_public_models_dry_run_exposes_gplfr():
    assert "gplfr" in rerun_public_models.METHODS
    result = subprocess.run(
        [sys.executable, "-m", "thousandworlds.rerun_public_models", "--dry-run", "--methods", "gplfr", "--subsets", "multi-partial"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "-m thousandworlds.run_model gplfr multi-partial" in result.stdout


def test_gplfr_reference_module_was_deleted():
    assert not Path(models.__file__).with_name("_gplfr_reference.py").exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("thousandworlds.models._gplfr_reference")


def test_gplfr_core_has_no_forbidden_reference_strings():
    source = Path(inspect.getsourcefile(GPLFRCore)).read_text()
    for token in ("MCMC", "NUTS", "wandb", "pca_init", "frozen_Z", "log_pred_density", "equicorr", "intel_extension_for_pytorch"):
        assert token not in source


def test_public_gplfr_run_does_not_fit_on_test_targets():
    source = inspect.getsource(run_model._run_gplfr)
    assert "test_Y=" not in source
    assert "test_field_mask=" not in source


def test_pca_ridge_latent_ridge_does_not_penalize_intercept():
    H = torch.ones((4, 1), dtype=torch.float64)
    Z = torch.full((4, 1), 3.0, dtype=torch.float64)
    B = fit_latent_ridge(H, Z, lambda_reg=1.0e6, intercept=True)
    torch.testing.assert_close(B, torch.tensor([[3.0]], dtype=torch.float64))

    B = fit_latent_ridge(H, torch.ones_like(Z), lambda_reg=1.0, intercept=False)
    torch.testing.assert_close(B, torch.tensor([[0.5]], dtype=torch.float64))


def test_pca_ridge_design_drops_reference_gcm_with_intercept():
    X = torch.zeros((3, 2), dtype=torch.float64)
    s = torch.tensor([0, 1, 2])
    H = build_design_matrix(X, s, n_sim_types=3, design_cfg={"intercept": True, "inputs": True, "sim_onehot": True})
    assert H.shape == (3, 5)
    torch.testing.assert_close(H[:, 3:], torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))


def test_pca_ridge_resolver_defaults_and_config_replay():
    args = run_model.argparse.Namespace(
        latent_dim=60,
        lambda_reg=1.0e-3,
        ppca_iters=50,
        n_folds=5,
        cv_latent_dim="20,50,100,150",
        cv_lambda="1.0e-6,1.0e-4,1.0e-3,1.0e-2,1.0e-1,1.0",
        best_latent_dim=None,
        best_lambda_reg=None,
        _explicit_args=set(),
    )
    hparams = run_model._pca_ridge_hparams(args)
    assert hparams["latent_dim"] == 50
    assert hparams["n_folds"] == 5

    cfg = {
        "pca_ridge": {"latent_dim": 100, "lambda_reg": 0.1, "ppca_iters": 7},
        "CV_sweep": {"n_folds": 2, "latent_dim": [50, 100], "lambda_reg": [0.1, 1.0], "scores": [[1.0, 0.5], [0.4, 0.6]]},
        "best": {"latent_dim": 100, "lambda_reg": 0.1, "cv_equal_group_normalized_rmse": 0.4},
    }
    replay = run_model._pca_ridge_hparams(args, cfg)
    assert replay["best_latent_dim"] == 100
    assert replay["best_lambda_reg"] == 0.1
    assert replay["cv_sweep_scores"] == [[1.0, 0.5], [0.4, 0.6]]

    args._explicit_args = {"latent_dim"}
    args.latent_dim = 50
    override = run_model._pca_ridge_hparams(args, cfg)
    assert override["best_latent_dim"] is None
    assert override["latent_dim"] == 50


def test_pca_ridge_fit_predict_smoke():
    model = PCARidge(latent_dim=1, lambda_reg=1.0e-3, dtype=torch.float64, device="cpu")
    X = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    s = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    Y = torch.randn((4, 3, 2), dtype=torch.float64)
    field_mask = torch.ones((4, 2), dtype=torch.bool)
    sh_mask = torch.ones((3, 2), dtype=torch.bool)
    model.fit(X, s, Y, field_mask=field_mask, sh_mask=sh_mask, ppca_iters=2, seed=0, n_sim_types=2)
    pred = model.predict(X[:2], s[:2])
    assert pred.shape == (2, 3, 2)
    assert torch.isfinite(pred).all()


def test_pca_gbt_resolver_defaults_and_config_replay():
    args = run_model.argparse.Namespace(
        latent_dim=60,
        gbt_learning_rate=0.05,
        gbt_max_iter=600,
        gbt_max_leaf_nodes=31,
        gbt_min_samples_leaf=10,
        gbt_l2=1.0,
        _explicit_args=set(),
    )
    hparams = run_model._pca_gbt_hparams(args)
    assert hparams["latent_dim"] == 150  # default ignores the generic --latent-dim unless explicit
    assert hparams["n_folds"] == 5  # CV-sweep default (mirrors pca_ridge)
    assert hparams["best_gbt_learning_rate"] is None  # no config -> fresh sweep

    args._explicit_args = {"latent_dim", "gbt_max_leaf_nodes"}
    args.latent_dim = 60
    args.gbt_max_leaf_nodes = 63
    override = run_model._pca_gbt_hparams(args)
    assert override["latent_dim"] == 60
    assert override["gbt_max_leaf_nodes"] == 63

    # A pure --config replay carries no explicit CLI overrides. config.json stores
    # the chosen hyperparameters with bare keys (learning_rate, max_leaf_nodes);
    # replay maps them back onto the gbt_* hparam names.
    args._explicit_args = set()
    replay = run_model._pca_gbt_hparams(args, {"pca_gbt": {"latent_dim": 8, "max_leaf_nodes": 15}})
    assert replay["latent_dim"] == 8
    assert replay["gbt_max_leaf_nodes"] == 15


def test_pca_gbt_cv_replay_uses_stored_best():
    cfg = {
        "pca_gbt": {"latent_dim": 150, "learning_rate": 0.1, "max_leaf_nodes": 63},
        "CV_sweep": {"n_folds": 3, "learning_rate": [0.03, 0.05, 0.1], "max_leaf_nodes": [15, 31, 63], "scores": [[0.5, 0.5, 0.5]] * 3},
        "best": {"learning_rate": 0.1, "max_leaf_nodes": 63, "cv_equal_group_normalized_rmse": 0.42},
    }
    args = run_model.argparse.Namespace(
        latent_dim=60,
        gbt_learning_rate=0.05,
        gbt_max_iter=600,
        gbt_max_leaf_nodes=31,
        gbt_min_samples_leaf=10,
        gbt_l2=1.0,
        _explicit_args=set(),
    )
    hp = run_model._pca_gbt_hparams(args, cfg)
    # stored best -> _run_pca_gbt skips the sweep and uses these directly
    assert hp["best_gbt_learning_rate"] == 0.1
    assert hp["best_gbt_max_leaf_nodes"] == 63
    assert hp["gbt_max_leaf_nodes"] == 63  # chosen value baked into the pca_gbt block
    assert hp["best_cv_equal_group_normalized_rmse"] == 0.42
    assert hp["cv_sweep_scores"] is not None


def test_pca_gbt_fit_predict_smoke():
    from thousandworlds.models.pca_gbt import PCAGBT

    rng = np.random.default_rng(0)
    n = 60
    model = PCAGBT(latent_dim=2, max_iter=30, early_stopping=False, n_jobs=1, dtype=torch.float64, device="cpu")
    X = torch.tensor(rng.normal(size=(n, 2)), dtype=torch.float64)
    s = torch.tensor(([0, 1] * n)[:n], dtype=torch.long)
    Y = torch.tensor(rng.normal(size=(n, 3, 2)), dtype=torch.float64)
    field_mask = torch.ones((n, 2), dtype=torch.bool)
    sh_mask = torch.ones((3, 2), dtype=torch.bool)
    model.fit(X, s, Y, field_mask=field_mask, sh_mask=sh_mask, ppca_iters=2, seed=0, n_sim_types=2)
    pred = model.predict(X[:5], s[:5])
    assert pred.shape == (5, 3, 2)
    assert torch.isfinite(pred).all()


def test_pca_mlp_equal_group_mean_weights_groups_not_field_counts():
    per_field = torch.tensor([10.0, 2.0, 4.0], dtype=torch.float32)
    field_names = ["surface_temperature", "temperature_0", "temperature_1"]
    out = _equal_group_mean(per_field, field_names)
    assert torch.isclose(out, torch.tensor(6.5))


def test_pca_mlp_depth_is_native_and_config_replayable():
    for depth in (1, 2):
        model = _ScoreMLP(3, 2, 4, num_layers=depth, activation="silu")
        assert sum(isinstance(layer, torch.nn.Linear) for layer in model.net) == depth + 1
    args = run_model.argparse.Namespace(
        subset="unused",
        latent_dim=60,
        hidden_width=128,
        depth=2,
        num_steps=3000,
        lr=1.0e-3,
        weight_decay=1.0e-3,
        hard_stop_step=None,
    )
    assert run_model._pca_mlp_hparams(args)["depth"] == 2
    assert run_model._pca_mlp_hparams(
        args, {"pca_mlp": {"depth": 1}}
    )["depth"] == 1


def test_pca_mlp_restores_best_validation_step(monkeypatch):
    saved = {"call": 0, "best_state": None}
    metric_seq = [5.0, 4.0, 3.0, 4.0, 5.0]

    def fake_eval(self, *args, **kwargs):
        value = metric_seq[saved["call"]]
        if saved["call"] == 2:
            saved["best_state"] = deepcopy(self._net.state_dict())
        saved["call"] += 1
        return value

    monkeypatch.setattr(PCAMLP, "_eval_norm_srmse_val_1e3", fake_eval)

    model = PCAMLP(latent_dim=1, hidden_width=4, dtype=torch.float64, device="cpu")
    X = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [1.5, 1.5]], dtype=torch.float64)
    s = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    Y = torch.randn((4, 2, 1), dtype=torch.float64)
    field_mask = torch.ones((4, 1), dtype=torch.bool)
    sh_mask = torch.ones((2, 1), dtype=torch.bool)

    model.fit(
        X[:2],
        s[:2],
        Y[:2],
        field_mask=field_mask[:2],
        sh_mask=sh_mask,
        field_names=["surface_temperature"],
        num_steps=4,
        ppca_iters=2,
        lr=1.0e-2,
        weight_decay=0.0,
        seed=0,
        val_X=X[2:],
        val_s=s[2:],
        val_Y=Y[2:],
        val_field_mask=field_mask[2:],
        log_every=1,
        early_stop_patience_evals=2,
    )

    assert model.mlp_fit_stats_["best_step"] == 2
    assert model.mlp_fit_stats_["steps_run"] == 4
    assert model.mlp_fit_stats_["early_stop_triggered"] is True
    for key, value in model._net.state_dict().items():
        torch.testing.assert_close(value, saved["best_state"][key])
