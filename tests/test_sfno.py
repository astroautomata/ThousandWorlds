from __future__ import annotations

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_harmonics")

import thousandworlds as tw
import thousandworlds.models.sfno as sfno_module
from thousandworlds.models._common import enforce_equatorial_symmetry_grid
from thousandworlds.models._coordinate import field_norm_stats, latitude_weights
from thousandworlds.models.sfno import SFNO
from thousandworlds.models.sfno import _masked_latitude_mse

sfno_smoke = pytest.mark.skipif(
    os.environ.get("TW_RUN_SFNO_SMOKE") != "1",
    reason="set TW_RUN_SFNO_SMOKE=1 to run the expensive SFNO smoke checks",
)


@pytest.mark.parametrize("n_fields", [48, 53])
def test_sfno_forward_backward_shapes(n_fields: int):
    model = SFNO(embed_dim=4, num_layers=1, batch_size=2, device="cpu")
    net = model._build_network(n_fields)
    x = torch.randn(2, 13, 32, 64, requires_grad=True)

    pred = net(x)
    pred.square().mean().backward()

    assert pred.shape == (2, n_fields, 32, 64)
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in net.parameters() if parameter.requires_grad)


def test_sfno_native_position_embedding_and_retained_modes():
    net = SFNO(embed_dim=6, num_layers=2, device="cpu")._build_network(3)

    assert tuple(net.pos_embed.position_embeddings.shape) == (1, 6, 32, 64)
    assert net.trans_down.lmax == net.trans_down.mmax == 22
    assert net.trans.lmax == net.trans.mmax == 22
    assert net.normalization_layer == "none"
    assert not net.residual_prediction


def test_sfno_passes_every_frozen_torch_harmonics_constructor_argument(monkeypatch):
    captured = {}

    def build(**kwargs):
        captured.update(kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(sfno_module, "SphericalFourierNeuralOperator", build)
    SFNO(embed_dim=6, num_layers=2, device="cpu")._build_network(53)

    assert captured == {
        "img_size": (32, 64),
        "grid": "legendre-gauss",
        "grid_internal": "legendre-gauss",
        "scale_factor": 1,
        "in_chans": 13,
        "out_chans": 53,
        "embed_dim": 6,
        "num_layers": 2,
        "activation_function": "gelu",
        "encoder_layers": 1,
        "use_mlp": True,
        "mlp_ratio": 2.0,
        "drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "normalization_layer": "none",
        "hard_thresholding_fraction": 22 / 32,
        "residual_prediction": False,
        "pos_embed": "learnable latlon",
        "bias": False,
    }


def _loss_and_grad(target: torch.Tensor, field_mask: torch.Tensor):
    torch.manual_seed(4)
    net = torch.nn.Conv2d(3, 2, 1, bias=False)
    pred = net(torch.randn(3, 3, 32, 64, generator=torch.Generator().manual_seed(5)))
    loss = _masked_latitude_mse(pred, target, field_mask, torch.as_tensor(latitude_weights(32)))
    loss.backward()
    return loss.detach(), [parameter.grad.detach().clone() for parameter in net.parameters()]


def test_sfno_masked_loss_and_gradients_ignore_nan_and_large_values():
    target = torch.randn(3, 2, 32, 64, generator=torch.Generator().manual_seed(2))
    field_mask = torch.tensor([[True, False], [True, True], [False, True]])
    target_nan = target.clone()
    target_large = target.clone()
    target_nan[~field_mask] = torch.nan
    target_large[~field_mask] = 1.0e9

    loss_nan, grads_nan = _loss_and_grad(target_nan, field_mask)
    loss_large, grads_large = _loss_and_grad(target_large, field_mask)

    torch.testing.assert_close(loss_nan, loss_large, rtol=1.0e-6, atol=1.0e-7)
    for grad_nan, grad_large in zip(grads_nan, grads_large, strict=True):
        torch.testing.assert_close(grad_nan, grad_large, rtol=1.0e-6, atol=1.0e-7)


def test_sfno_external_prediction_path_enforces_equatorial_symmetry():
    model = SFNO(embed_dim=4, num_layers=1, device="cpu")
    model._build_network(2)
    model.field_names_ = ["surface_temperature", "v_0"]
    model._field_mean = torch.zeros(2)
    model._field_std = torch.ones(2)
    pred = model.predict(np.zeros((2, 8), dtype=np.float32), np.array([0, 4])).numpy()

    sym = enforce_equatorial_symmetry_grid(pred, model.field_names_)

    np.testing.assert_allclose(sym[:, 0], np.flip(sym[:, 0], axis=-2), atol=1.0e-6)
    np.testing.assert_allclose(sym[:, 1], -np.flip(sym[:, 1], axis=-2), atol=1.0e-6)


def test_sfno_fit_validation_and_hard_stop_statistics():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(4, 8)).astype(np.float32)
    s = np.arange(4, dtype=np.int64)
    Y = rng.normal(size=(4, 1, 32, 64)).astype(np.float32)
    mask = np.ones((4, 1), dtype=bool)
    model = SFNO(embed_dim=4, num_layers=1, batch_size=2, device="cpu")

    model.fit(
        X,
        s,
        Y,
        field_mask=mask,
        field_names=["surface_temperature"],
        num_steps=5,
        hard_stop_step=2,
        val_X=X[:2],
        val_s=s[:2],
        val_Y=Y[:2],
        val_field_mask=mask[:2],
        log_every=1,
    )

    assert model.predict(X[:2], s[:2]).shape == (2, 1, 32, 64)
    assert model.fit_stats_["steps_requested"] == 5
    assert model.fit_stats_["steps_run"] == 2
    assert model.fit_stats_["best_step"] in {1, 2}
    assert np.isfinite(model.fit_stats_["best_val_equal_base_normalized_rmse_grid"])
    assert model.fit_stats_["loss"] == "masked_latitude_weighted_grid_mse"


@pytest.mark.parametrize(
    ("n_sim_types", "labels"),
    [(1, [0, 0]), (2, [0, 1]), (5, [0, 4])],
)
def test_sfno_real_fit_supports_dynamic_simulation_types(
    n_sim_types, labels
):
    rng = np.random.default_rng(n_sim_types)
    X = rng.normal(size=(2, 8)).astype(np.float32)
    s = np.asarray(labels, dtype=np.int64)
    Y = rng.normal(size=(2, 1, 32, 64)).astype(np.float32)
    mask = np.ones((2, 1), dtype=bool)
    model = SFNO(embed_dim=4, num_layers=1, batch_size=2, device="cpu")

    model.fit(
        X,
        s,
        Y,
        field_mask=mask,
        field_names=["surface_temperature"],
        n_sim_types=n_sim_types,
        num_steps=1,
        val_X=X,
        val_s=s,
        val_Y=Y,
        val_field_mask=mask,
        log_every=1,
    )

    assert model.n_sim_types_ == n_sim_types
    assert model._input_grid(torch.as_tensor(X), torch.as_tensor(s)).shape == (
        2,
        8 + n_sim_types,
        32,
        64,
    )
    assert torch.isfinite(model.predict(X, s)).all()


@pytest.mark.parametrize("labels", [[-1, 0], [0, 2]])
def test_sfno_rejects_out_of_range_simulation_type_labels(labels):
    model = SFNO(device="cpu")
    model.n_sim_types_ = 2

    with pytest.raises(ValueError, match="simulation-type labels"):
        model._input_grid(torch.zeros(2, 8), torch.tensor(labels))


def test_sfno_training_statistics_keep_complete_examples_batched():
    calls = []

    class CountingNet(torch.nn.Module):
        def forward(self, x):
            calls.append(len(x))
            return x.new_zeros((len(x), 1, 32, 64))

    model = SFNO(embed_dim=4, num_layers=1, batch_size=2, device="cpu")
    model._net = CountingNet()
    model._lat_weights = torch.as_tensor(latitude_weights(32))
    loss = model._dataset_loss(
        torch.zeros(5, 8),
        torch.zeros(5, dtype=torch.long),
        torch.zeros(5, 1, 32, 64),
        torch.ones(5, 1, 32, 64, dtype=torch.bool),
    )

    assert calls == [2, 2, 1]
    assert loss == 0.0


@pytest.mark.requires_dataset
@sfno_smoke
def test_sfno_real_example_transform_round_trip_and_plausible_prediction(data_dir):
    bundle = tw.load("single-complete", data_dir=data_dir, space="grid")
    stats = tw.load_stats("single-complete", data_dir)
    names = bundle.field_names[:2]
    X = bundle.X_train[:3]
    Y = bundle.Y_train[:3, :2]
    transformed = tw.preprocess_outputs_grid(Y, names, stats, X=X)
    np.testing.assert_allclose(
        tw.inverse_preprocess_outputs_grid(transformed, names, stats, X=X),
        Y,
        rtol=1.0e-5,
        atol=1.0e-5,
        equal_nan=True,
    )

    class ZeroNet(torch.nn.Module):
        def forward(self, x):
            return x.new_zeros((len(x), 2, 32, 64))

    model = SFNO(embed_dim=4, num_layers=1, device="cpu")
    model._net = ZeroNet()
    model.field_names_ = names
    mask = torch.as_tensor(bundle.field_mask_train[:3, :2])
    model._field_mean, model._field_std = field_norm_stats(
        torch.as_tensor(transformed),
        mask,
        torch.as_tensor(latitude_weights(32)),
    )
    s = np.zeros(3, dtype=np.int64)
    physical = tw.inverse_preprocess_outputs_grid(model.predict(tw.transform_inputs(X, stats), s).numpy(), names, stats, X=X)

    assert np.isfinite(physical[np.broadcast_to(bundle.field_mask_train[:3, :2, None, None], physical.shape)]).all()
    for field in range(2):
        train = Y[:, field][np.isfinite(Y[:, field])]
        span = train.max() - train.min()
        assert train.min() - span <= np.median(physical[:, field]) <= train.max() + span


@sfno_smoke
def test_sfno_conditioning_changes_anomaly_pattern_and_has_spatial_variance():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(8, 8)).astype(np.float32)
    s = np.arange(8, dtype=np.int64) % 5
    lat = np.linspace(-1.0, 1.0, 32, dtype=np.float32)[:, None]
    lon = np.linspace(-np.pi, np.pi, 64, endpoint=False, dtype=np.float32)[None]
    p1, p2 = lat * np.cos(lon), (1.0 - lat**2) * np.sin(2.0 * lon)
    Y = np.stack([X[i, 0] * p1 + X[i, 1] * p2 for i in range(len(X))], axis=0)[:, None]
    model = SFNO(embed_dim=8, num_layers=1, batch_size=8, device="cpu")
    model.fit(
        X,
        s,
        Y,
        field_mask=np.ones((8, 1), dtype=bool),
        field_names=["surface_temperature"],
        num_steps=40,
        lr=1.0e-2,
        seed=2,
    )

    pred = model.predict(X[:2], s[:2]).numpy()[:, 0]
    weights = latitude_weights(32)
    anomaly = pred - (pred.mean(axis=-1) * weights[None]).sum(axis=-1)[:, None, None]
    relative_difference = np.linalg.norm(anomaly[0] - anomaly[1]) / max(np.linalg.norm(anomaly[0]), 1.0e-12)

    assert relative_difference > 1.0e-3
    assert np.var(pred, axis=(-2, -1)).min() > 1.0e-8


@sfno_smoke
def test_sfno_overfits_twenty_deterministic_examples():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(20, 8)).astype(np.float32)
    s = np.arange(20, dtype=np.int64) % 5
    Y = np.full((20, 1, 32, 64), 2.0, dtype=np.float32)
    model = SFNO(embed_dim=4, num_layers=1, batch_size=20, device="cpu")
    model.fit(
        X,
        s,
        Y,
        field_mask=np.ones((20, 1), dtype=bool),
        field_names=["surface_temperature"],
        num_steps=500,
        lr=1.0e-2,
        seed=3,
    )

    initial = model.fit_stats_["initial_train_normalized_latitude_weighted_mse"]
    final = model.fit_stats_["final_train_normalized_latitude_weighted_mse"]
    assert final < 1.0e-3
    assert final < 0.01 * initial
