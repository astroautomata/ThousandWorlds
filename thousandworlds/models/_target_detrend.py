from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import thousandworlds as tw
from thousandworlds.preprocessing import LinearTrend, Stats


TARGET_DETREND_CFG = {
    "enabled": True,
    "lambda": 1.0e-3,
    "design": {"intercept": False, "inputs": True, "sim_onehot": True},
}


@dataclass(frozen=True)
class NetTargetDetrender:
    trend: LinearTrend
    field_names: list[str]
    stats: Stats
    sh_mask: np.ndarray
    inverse_sht: np.ndarray

    def _grid(self, coeffs_norm: np.ndarray, *, include_mean: bool) -> np.ndarray:
        coeffs_norm = np.asarray(coeffs_norm, dtype=np.float32)
        coeffs = np.zeros_like(coeffs_norm)
        for j, name in enumerate(self.field_names):
            record = self.stats.spectral[name]
            coeffs[:, j, record.mask] = coeffs_norm[:, j, record.mask] * record.sigma
            if include_mean:
                coeffs[:, j, record.mask] += record.mean
        return tw.to_grid(
            tw.apply_symmetry_mask(coeffs, self.field_names, self.sh_mask),
            self.inverse_sht,
        ).astype(np.float32)

    def trend_delta_grid(self, X_std: np.ndarray, gcm_idx: np.ndarray) -> np.ndarray:
        return self._grid(tw.apply_linear_trend(X_std, gcm_idx, self.trend), include_mean=False)

    def zero_residual_grid(self, n: int) -> np.ndarray:
        return self._grid(
            np.zeros((int(n), len(self.field_names), self.sh_mask.shape[-1]), dtype=np.float32),
            include_mean=True,
        )

    def bare_trend_grid(self, X_std: np.ndarray, gcm_idx: np.ndarray) -> np.ndarray:
        return self.zero_residual_grid(len(X_std)) + self.trend_delta_grid(X_std, gcm_idx)

    def remove_grid(self, Y: np.ndarray, X_std: np.ndarray, gcm_idx: np.ndarray) -> np.ndarray:
        return np.asarray(Y, dtype=np.float32) - self.trend_delta_grid(X_std, gcm_idx)

    def apply_grid(self, residual: np.ndarray, X_std: np.ndarray, gcm_idx: np.ndarray) -> np.ndarray:
        return np.asarray(residual, dtype=np.float32) + self.trend_delta_grid(X_std, gcm_idx)


def fit_net_target_detrender(
    X_std: np.ndarray,
    gcm_idx: np.ndarray,
    coeffs_norm: np.ndarray,
    *,
    field_mask: np.ndarray,
    field_names: list[str],
    stats: Stats,
    sh_mask: np.ndarray,
    inverse_sht: np.ndarray,
) -> NetTargetDetrender:
    coeffs_norm = np.asarray(coeffs_norm, dtype=np.float32)
    sh_mask = np.asarray(sh_mask, dtype=bool)
    field_coeff_mask = sh_mask.T if sh_mask.shape == (coeffs_norm.shape[2], coeffs_norm.shape[1]) else sh_mask
    trend = tw.fit_linear_trend(
        X_std,
        gcm_idx,
        coeffs_norm,
        lambda_reg=TARGET_DETREND_CFG["lambda"],
        field_mask=field_mask,
        sh_mask=field_coeff_mask,
        design_cfg=TARGET_DETREND_CFG["design"],
    )
    return NetTargetDetrender(trend, list(field_names), stats, field_coeff_mask, np.asarray(inverse_sht))
