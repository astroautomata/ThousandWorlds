import numpy as np
import pytest

from thousandworlds.evaluate import _ramse_per_example_field, _weighted_mse_per_example_field
from thousandworlds.spectral import GRID_SHAPE, forward_sht_operator, to_grid, to_spectral


@pytest.fixture(scope="module")
def band_limited_fields() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal((3, 2, 484)).astype(np.float32)
    target = to_grid(coeffs)
    pred = to_grid(coeffs + 0.3 * rng.standard_normal((3, 2, 484)).astype(np.float32))
    return pred, target


def test_forward_operator_is_left_inverse() -> None:
    analysis, _ = forward_sht_operator()
    synthesis = np.load(__import__("thousandworlds").spectral.ASSET_DIR / "inverse_sht.npy")
    err = np.abs(analysis.astype(np.float64) @ synthesis.astype(np.float64) - np.eye(484)).max()
    assert err < 1.0e-5


def test_round_trip_on_band_limited_fields(band_limited_fields) -> None:
    _, target = band_limited_fields
    assert np.allclose(to_grid(to_spectral(target)), target, atol=1.0e-3)


def test_ramse_zero_on_identical_fields(band_limited_fields) -> None:
    _, target = band_limited_fields
    assert np.abs(_ramse_per_example_field(target, target)).max() < 1.0e-2


def test_zero_prediction_identity(band_limited_fields) -> None:
    # AMSE(0, y) = 3 * band power of y; band power equals the area-weighted
    # mean square for band-limited fields (Parseval).
    _, target = band_limited_fields
    amse = np.square(_ramse_per_example_field(np.zeros_like(target), target))
    ms = _weighted_mse_per_example_field(target, np.zeros_like(target))
    assert np.allclose(amse, 3.0 * ms, rtol=2.0e-3)


def test_amse_at_least_mse_on_band_limited_fields(band_limited_fields) -> None:
    pred, target = band_limited_fields
    amse = np.square(_ramse_per_example_field(pred, target))
    mse = _weighted_mse_per_example_field(pred, target)
    assert (amse >= mse * (1.0 - 1.0e-4)).all()
