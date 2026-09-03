from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from thousandworlds.models._common import PreparedTWData, prepare_tw_fold
from thousandworlds.preprocessing import SpectralFieldStats, Stats, normalise_spectral


def _data() -> PreparedTWData:
    masks = {
        "surface_temperature": np.array([True, False, True]),
        "v_0": np.array([True, True, False]),
    }
    spectral = {
        name: SpectralFieldStats(
            mean=np.full(mask.sum(), 99.0, dtype=np.float32),
            sigma=99.0,
            count=99,
            energy_mean=99.0,
            mask=mask,
        )
        for name, mask in masks.items()
    }
    stats = Stats(
        stats_dir=Path("accepted-full-stats"),
        input_names=["T_star", "F_star"],
        field_names=list(masks),
        per_level=True,
        asr_olr_normalize_by_f_star=False,
        strategies={
            "T_star": ("Z-scaling", {"stat_key_pattern": "T_star"}),
            "F_star": ("log_Z-scaling", {"stat_key_pattern": "F_star"}),
            "surface_temperature": ("Z-scaling", {"stat_key_pattern": "surface_temperature"}),
            "v": ("Z-scaling", {"stat_key_pattern": "v"}),
        },
        normalize_mean={"T_star": np.array([99.0]), "F_star": np.array([99.0])},
        normalize_std={"T_star": np.array([99.0]), "F_star": np.array([99.0])},
        spectral=spectral,
        spectral_meta={"fields": {name: {"count": 99, "sigma": 99.0} for name in masks}},
        transforms_meta={},
    )
    X_train = np.array([[1.0, 1.0], [3.0, np.e**2], [10.0, np.e**4], [20.0, np.e**6]], dtype=np.float32)
    coeffs = np.array(
        [
            [[1.0, 0.0, 3.0], [2.0, 4.0, 0.0]],
            [[3.0, 0.0, 5.0], [0.0, 0.0, 0.0]],
            [[100.0, 0.0, 300.0], [200.0, 400.0, 0.0]],
            [[500.0, 0.0, 700.0], [600.0, 800.0, 0.0]],
        ],
        dtype=np.float32,
    )
    field_mask = np.array([[True, True], [True, False], [True, True], [True, True]])
    grid = SimpleNamespace(X_train=X_train, X_test=X_train[:1], Y_train=np.zeros((4, 2, 2, 2), dtype=np.float32))
    spectral_bundle = SimpleNamespace(
        Y_train=coeffs,
        field_mask_train=field_mask,
        raw_field_names=list(masks),
    )
    return PreparedTWData(
        grid_bundle=grid,
        spectral_bundle=spectral_bundle,
        stats=stats,
        X_train_std=np.empty_like(X_train),
        X_test_std=np.empty_like(X_train[:1]),
        s_train=np.zeros(4, dtype=np.int64),
        s_test=np.zeros(1, dtype=np.int64),
        sh_mask=np.stack(list(masks.values()), axis=1),
        inverse_sht=np.empty((0, 0), dtype=np.float32),
        gcm_labels=["gcm"],
    )


def test_fold_statistics_ignore_validation_only_perturbations():
    data = _data()
    perturbed = deepcopy(data)
    perturbed.grid_bundle.X_train[2:] *= 1_000.0
    perturbed.spectral_bundle.Y_train[2:] *= 1_000.0

    fold = prepare_tw_fold(data, np.array([0, 1]))
    fold_perturbed = prepare_tw_fold(perturbed, np.array([0, 1]))

    for key in fold.stats.normalize_mean:
        np.testing.assert_array_equal(fold.stats.normalize_mean[key], fold_perturbed.stats.normalize_mean[key])
        np.testing.assert_array_equal(fold.stats.normalize_std[key], fold_perturbed.stats.normalize_std[key])
    for name in fold.stats.spectral:
        np.testing.assert_array_equal(fold.stats.spectral[name].mean, fold_perturbed.stats.spectral[name].mean)
        assert fold.stats.spectral[name].sigma == fold_perturbed.stats.spectral[name].sigma

    np.testing.assert_allclose(fold.X_train_std[:2].mean(axis=0), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(fold.X_train_std[:2].std(axis=0), 1.0, atol=1.0e-6)
    assert data.stats.normalize_mean["T_star"].item() == 99.0


def test_fold_statistics_preserve_masks_and_spectral_shapes():
    data = _data()
    fold = prepare_tw_fold(data, np.array([0, 1]))
    normalized = normalise_spectral(
        data.spectral_bundle.Y_train,
        data.spectral_bundle.raw_field_names,
        fold.stats,
    )

    assert normalized.shape == data.spectral_bundle.Y_train.shape
    assert fold.sh_mask.shape == (3, 2)
    np.testing.assert_array_equal(fold.spectral_bundle.field_mask_train, data.spectral_bundle.field_mask_train)
    for j, name in enumerate(data.spectral_bundle.raw_field_names):
        np.testing.assert_array_equal(fold.stats.spectral[name].mask, data.stats.spectral[name].mask)
        np.testing.assert_array_equal(normalized[:, j, ~fold.stats.spectral[name].mask], 0.0)
