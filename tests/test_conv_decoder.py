from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

import thousandworlds as tw
import thousandworlds.models.conv_decoder as conv_decoder
from thousandworlds.models._common import (
    average_space_grid,
    enforce_equatorial_symmetry_grid,
    inverse_average_space_grid,
)
from thousandworlds.models._coordinate import latitude_weights


DATA_DIR = Path(__file__).resolve().parent.parent / "dataset"


def test_conv_decoder_forward_backward_48_fields():
    spec = importlib.util.find_spec("thousandworlds.models.conv_decoder")
    assert spec is not None, "Conv-Decoder module is not implemented"
    net = importlib.import_module("thousandworlds.models.conv_decoder")._ConvDecoderNet(
        n_fields=48,
        seed_channels=8,
        convs_per_stage=1,
    )

    pred = net(torch.randn(2, 13))
    pred.square().mean().backward()

    assert pred.shape == (2, 48, 32, 64)
    assert all(parameter.grad is not None for parameter in net.parameters())


def test_conv_decoder_forward_backward_53_fields_with_two_convs_per_stage():
    net = conv_decoder._ConvDecoderNet(n_fields=53, seed_channels=8, convs_per_stage=2)

    pred = net(torch.randn(2, 13))
    pred.mean().backward()

    assert pred.shape == (2, 53, 32, 64)
    assert all(parameter.grad is not None for parameter in net.parameters())
    assert not any(isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.Dropout)) for module in net.modules())


def test_masked_nan_and_large_targets_leave_loss_and_gradients_unchanged():
    torch.manual_seed(0)
    net = conv_decoder._ConvDecoderNet(n_fields=3, seed_channels=8, convs_per_stage=1)
    inputs = torch.randn(2, 13)
    target = torch.randn(2, 3, 32, 64)
    field_mask = torch.tensor([[True, False, True], [True, True, False]])
    masked = ~field_mask[:, :, None, None].expand_as(target)

    def loss_and_grad(corrupted: torch.Tensor):
        net.zero_grad(set_to_none=True)
        loss = conv_decoder._masked_latitude_weighted_mse(
            net(inputs),
            corrupted,
            field_mask,
            torch.as_tensor(latitude_weights(32)),
        )
        loss.backward()
        return loss.detach(), [parameter.grad.detach().clone() for parameter in net.parameters()]

    baseline_loss, baseline_grad = loss_and_grad(target)
    for fill in (float("nan"), 1.0e9):
        corrupted = target.clone()
        corrupted[masked] = fill
        loss, gradients = loss_and_grad(corrupted)
        torch.testing.assert_close(loss, baseline_loss, rtol=1.0e-6, atol=1.0e-7)
        for gradient, expected in zip(gradients, baseline_grad, strict=True):
            torch.testing.assert_close(gradient, expected, rtol=1.0e-6, atol=1.0e-7)


def test_external_prediction_path_enforces_equatorial_symmetry():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(4, 8)).astype(np.float32)
    s = np.array([0, 1, 2, 3])
    Y = rng.normal(size=(4, 2, 32, 64)).astype(np.float32)
    model = conv_decoder.ConvDecoder(seed_channels=8, convs_per_stage=1, batch_size=4, device="cpu")
    model.fit(
        X,
        s,
        Y,
        field_mask=np.ones((4, 2), dtype=bool),
        field_names=["surface_temperature", "v_0"],
        n_sim_types=5,
        num_steps=1,
        val_X=X[:2],
        val_s=s[:2],
        val_Y=Y[:2],
        val_field_mask=np.ones((2, 2), dtype=bool),
        log_every=1,
    )

    pred = enforce_equatorial_symmetry_grid(model.predict(X[:2], s[:2]).numpy(), model.field_names_)

    np.testing.assert_allclose(pred[:, 0], np.flip(pred[:, 0], axis=-2), atol=1.0e-6)
    np.testing.assert_allclose(pred[:, 1], -np.flip(pred[:, 1], axis=-2), atol=1.0e-6)
    assert model.fit_stats_["best_step"] == 1
    assert np.isfinite(model.fit_stats_["best_val_equal_base_normalized_rmse_grid"])


@pytest.mark.parametrize(
    ("n_sim_types", "labels"),
    [(1, [0, 0]), (2, [0, 1]), (5, [0, 4])],
)
def test_conv_decoder_real_fit_supports_dynamic_simulation_types(
    n_sim_types, labels
):
    rng = np.random.default_rng(n_sim_types)
    X = rng.normal(size=(2, 8)).astype(np.float32)
    s = np.asarray(labels, dtype=np.int64)
    Y = rng.normal(size=(2, 1, 32, 64)).astype(np.float32)
    mask = np.ones((2, 1), dtype=bool)
    model = conv_decoder.ConvDecoder(
        seed_channels=8, convs_per_stage=1, batch_size=2, device="cpu"
    )

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
    assert model._net.project.in_features == 8 + n_sim_types
    assert torch.isfinite(model.predict(X, s)).all()


@pytest.mark.parametrize("labels", [[-1, 0], [0, 2]])
def test_conv_decoder_rejects_out_of_range_simulation_type_labels(labels):
    model = conv_decoder.ConvDecoder(device="cpu")
    model.n_sim_types_ = 2

    with pytest.raises(ValueError, match="simulation-type labels"):
        model._features(torch.zeros(2, 8), torch.tensor(labels))


@pytest.mark.requires_dataset
def test_real_examples_round_trip_and_prediction_is_finite_and_plausible():
    bundle = tw.load(subset="single-complete", data_dir=DATA_DIR, space="grid")
    stats = tw.load_stats("single-complete", DATA_DIR)
    field_names = ["surface_temperature", "temperature_0"]
    fields = [bundle.raw_field_names.index(name) for name in field_names]
    X = bundle.X_train[:4]
    Y = bundle.Y_train[:4, fields]
    Y_transformed = average_space_grid(Y, field_names, stats, X=X)
    np.testing.assert_allclose(
        inverse_average_space_grid(Y_transformed, field_names, stats, X=X),
        Y,
        rtol=1.0e-5,
        atol=1.0e-5,
    )

    model = conv_decoder.ConvDecoder(seed_channels=8, convs_per_stage=1, batch_size=4, device="cpu")
    model.fit(
        tw.transform_inputs(X, stats),
        np.zeros(4, dtype=np.int64),
        Y_transformed,
        field_mask=np.ones((4, 2), dtype=bool),
        field_names=field_names,
        n_sim_types=5,
        num_steps=5,
        lr=3.0e-3,
    )
    pred = inverse_average_space_grid(
        enforce_equatorial_symmetry_grid(model.predict(tw.transform_inputs(X, stats), np.zeros(4, dtype=np.int64)).numpy(), field_names),
        field_names,
        stats,
        X=X,
    )

    assert np.isfinite(pred).all()
    for field in range(len(field_names)):
        train = Y[:, field][np.isfinite(Y[:, field])]
        width = train.max() - train.min()
        assert train.min() - width <= np.median(pred[:, field]) <= train.max() + width


def test_parameter_conditioning_changes_anomaly_patterns_and_preserves_spatial_variance():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(4, 8)).astype(np.float32)
    X[1] += 3.0
    s = np.zeros(4, dtype=np.int64)
    lon = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    pattern = np.broadcast_to(np.sin(lon)[None], (32, 64)).astype(np.float32)
    Y = np.stack([(1.0 + x[0]) * pattern for x in X], axis=0)[:, None]
    model = conv_decoder.ConvDecoder(seed_channels=8, convs_per_stage=1, batch_size=4, device="cpu")
    model.fit(
        X,
        s,
        Y,
        field_mask=np.ones((4, 1), dtype=bool),
        field_names=["surface_temperature"],
        n_sim_types=5,
        num_steps=2,
        lr=3.0e-3,
    )

    pred = model.predict(X[:2], s[:2])
    weights = torch.as_tensor(latitude_weights(32))[None, None, :, None]
    anomaly = pred - (pred * weights).sum(dim=(-2, -1), keepdim=True) / pred.shape[-1]
    relative_difference = torch.linalg.vector_norm(anomaly[0] - anomaly[1]) / torch.linalg.vector_norm(anomaly[0]).clamp_min(1.0e-12)

    assert relative_difference > 1.0e-3
    assert torch.var(pred, dim=(-2, -1)).min() > 1.0e-8


def test_twenty_example_overfit_reaches_contract_threshold_within_500_steps():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 8)).astype(np.float32)
    s = np.arange(20) % 5
    Y = np.broadcast_to(X[:, :1, None, None], (20, 1, 32, 64)).copy()
    model = conv_decoder.ConvDecoder(seed_channels=8, convs_per_stage=1, batch_size=32, device="cpu")

    model.fit(
        X,
        s,
        Y,
        field_mask=np.ones((20, 1), dtype=bool),
        field_names=["surface_temperature"],
        n_sim_types=5,
        num_steps=500,
        lr=1.0e-2,
        weight_decay=0.0,
    )

    assert model.fit_stats_["final_train_normalized_loss"] < 1.0e-3
    assert model.fit_stats_["final_train_normalized_loss"] < 0.01 * model.fit_stats_["initial_train_normalized_loss"]
