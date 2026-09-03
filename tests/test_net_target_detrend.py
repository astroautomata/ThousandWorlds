from types import SimpleNamespace

import numpy as np

import thousandworlds as tw
from thousandworlds.models._target_detrend import fit_net_target_detrender


def _case(field_mask: np.ndarray):
    rng = np.random.default_rng(27)
    n, n_fields, n_coeffs = 6, 2, tw.spectral.N_COEFFS
    X = rng.normal(size=(n, 8)).astype(np.float32)
    s = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    coeffs = rng.normal(size=(n, n_fields, n_coeffs)).astype(np.float32)
    masks = np.ones((n_coeffs, n_fields), dtype=bool)
    stats = SimpleNamespace(
        spectral={
            name: SimpleNamespace(
                mean=rng.normal(size=n_coeffs).astype(np.float32),
                sigma=float(scale),
                mask=np.ones(n_coeffs, dtype=bool),
            )
            for name, scale in zip(("surface_temperature", "v_0"), (2.0, 0.5), strict=True)
        }
    )
    raw_coeffs = np.stack(
        [
            coeffs[:, j] * stats.spectral[name].sigma + stats.spectral[name].mean
            for j, name in enumerate(stats.spectral)
        ],
        axis=1,
    )
    Y = tw.to_grid(raw_coeffs)
    return X, s, coeffs, field_mask, list(stats.spectral), stats, masks, Y


def test_target_detrend_round_trip_covers_complete_and_partial_fields():
    for field_mask in (
        np.ones((6, 2), dtype=bool),
        np.array([[1, 1], [1, 0], [1, 1], [0, 1], [1, 1], [1, 1]], dtype=bool),
    ):
        X, s, coeffs, field_mask, names, stats, masks, Y = _case(field_mask)
        transform = fit_net_target_detrender(
            X,
            s,
            coeffs,
            field_mask=field_mask,
            field_names=names,
            stats=stats,
            sh_mask=masks,
            inverse_sht=tw.load_inverse_sht_matrix(),
        )

        restored = transform.apply_grid(transform.remove_grid(Y, X, s), X, s)

        np.testing.assert_allclose(restored, Y, rtol=0.0, atol=5.0e-6)
        assert transform.trend.design_cfg == {
            "intercept": False,
            "inputs": True,
            "sim_onehot": True,
        }
        assert transform.trend.lambda_reg == 1.0e-3


def test_zero_coefficient_residual_equals_bare_trend_prediction():
    field_mask = np.ones((6, 2), dtype=bool)
    X, s, coeffs, field_mask, names, stats, masks, _ = _case(field_mask)
    transform = fit_net_target_detrender(
        X,
        s,
        coeffs,
        field_mask=field_mask,
        field_names=names,
        stats=stats,
        sh_mask=masks,
        inverse_sht=tw.load_inverse_sht_matrix(),
    )

    restored = transform.apply_grid(transform.zero_residual_grid(len(X)), X, s)

    np.testing.assert_allclose(restored, transform.bare_trend_grid(X, s), rtol=0.0, atol=2.0e-6)
