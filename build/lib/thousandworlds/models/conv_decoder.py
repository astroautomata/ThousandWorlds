from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch.nn import functional as F

from ._common import enforce_equatorial_symmetry_grid, resolve_torch_device
from ._coordinate import (
    area_weighted_equal_base_variable_normalized_rmse_grid,
    field_norm_stats,
    latitude_weights,
)


def _tensor(x, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.as_tensor(np.array(x, copy=True), device=device, dtype=dtype)


def _masked_latitude_weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    field_mask: torch.Tensor,
    lat_weights: torch.Tensor,
) -> torch.Tensor:
    valid = field_mask[:, :, None, None] & torch.isfinite(target)
    safe_target = torch.where(valid, target, target.new_zeros(()))
    weights = lat_weights.to(device=pred.device, dtype=pred.dtype)[None, None, :, None]
    return (torch.where(valid, torch.square(pred - safe_target), pred.new_zeros(())) * weights).sum() / (
        valid.to(pred.dtype) * weights
    ).sum().clamp_min(1.0)


class _PaddedConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.conv(F.pad(F.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1))))


class _ConvDecoderNet(torch.nn.Module):
    def __init__(
        self,
        *,
        n_fields: int,
        seed_channels: int,
        convs_per_stage: int,
        n_inputs: int = 13,
    ) -> None:
        super().__init__()
        self.seed_channels = int(seed_channels)
        self.project = torch.nn.Linear(int(n_inputs), self.seed_channels * 4 * 8)
        widths = [self.seed_channels, self.seed_channels // 2, self.seed_channels // 4]
        layers: list[torch.nn.Module] = []
        in_channels = self.seed_channels
        for width in widths:
            layers.append(torch.nn.Upsample(scale_factor=2, mode="nearest"))
            for _ in range(int(convs_per_stage)):
                layers.append(_PaddedConv(in_channels, width))
                in_channels = width
        self.decoder = torch.nn.Sequential(*layers)
        self.output = torch.nn.Conv2d(in_channels, int(n_fields), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seed = self.project(x).reshape(x.shape[0], self.seed_channels, 4, 8)
        return self.output(self.decoder(seed))


class ConvDecoder:
    def __init__(
        self,
        *,
        seed_channels: int = 128,
        convs_per_stage: int = 1,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "auto",
    ) -> None:
        self.seed_channels = int(seed_channels)
        self.convs_per_stage = int(convs_per_stage)
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.device = resolve_torch_device(device)
        self._net: _ConvDecoderNet | None = None
        self.fit_stats_: dict | None = None
        self.n_sim_types_ = 5

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        s: np.ndarray | torch.Tensor,
        Y: np.ndarray | torch.Tensor,
        *,
        field_mask: np.ndarray | torch.Tensor,
        field_names: list[str],
        n_sim_types: int = 5,
        num_steps: int = 3000,
        lr: float = 3.0e-4,
        weight_decay: float = 1.0e-4,
        seed: int = 0,
        val_X: np.ndarray | torch.Tensor | None = None,
        val_s: np.ndarray | torch.Tensor | None = None,
        val_Y: np.ndarray | torch.Tensor | None = None,
        val_field_mask: np.ndarray | torch.Tensor | None = None,
        log_every: int = 100,
        early_stop_patience_evals: int | None = None,
        early_stop_min_delta: float = 0.0,
        hard_stop_step: int | None = None,
    ) -> None:
        n_sim_types = int(n_sim_types)
        if n_sim_types < 1:
            raise ValueError("ConvDecoder requires at least one simulation type.")
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        self.field_names_ = list(field_names)
        self.n_sim_types_ = n_sim_types
        X_t = _tensor(X, device=self.device, dtype=self.dtype)
        s_t = _tensor(s, device=self.device, dtype=torch.long)
        Y_t = _tensor(Y, device=self.device, dtype=self.dtype)
        field_mask_t = _tensor(field_mask, device=self.device, dtype=torch.bool)
        self._train_Y_np = np.asarray(Y, dtype=np.float32)
        self._train_field_mask_np = np.asarray(field_mask, dtype=bool)
        self._lat_weights = torch.as_tensor(latitude_weights(Y_t.shape[-2]), device=self.device, dtype=self.dtype)
        self._field_mean, self._field_std = field_norm_stats(Y_t, field_mask_t, self._lat_weights)
        target = (Y_t - self._field_mean[None, :, None, None]) / self._field_std[None, :, None, None]
        features = self._features(X_t, s_t)
        self._net = _ConvDecoderNet(
            n_fields=len(self.field_names_),
            seed_channels=self.seed_channels,
            convs_per_stage=self.convs_per_stage,
            n_inputs=8 + self.n_sim_types_,
        ).to(device=self.device, dtype=self.dtype)
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        total_steps = min(int(num_steps), int(hard_stop_step)) if hard_stop_step is not None else int(num_steps)
        stats_idx = (
            torch.arange(X_t.shape[0], device=self.device)
            if X_t.shape[0] <= self.batch_size
            else torch.randint(X_t.shape[0], (self.batch_size,), generator=generator, device=self.device)
        )
        with torch.no_grad():
            initial_loss = self._loss(features[stats_idx], target[stats_idx], field_mask_t[stats_idx])
        validation = None
        if val_X is not None and val_s is not None and val_Y is not None and val_field_mask is not None:
            validation = (val_X, val_s, np.asarray(val_Y, dtype=np.float32), np.asarray(val_field_mask, dtype=bool))
        best_metric, best_state, best_step = float("inf"), None, None
        bad_evals = n_evals = steps_run = 0
        early_stopped = False

        for step in range(total_steps):
            idx = torch.randint(X_t.shape[0], (self.batch_size,), generator=generator, device=self.device)
            self._net.train()
            optimizer.zero_grad(set_to_none=True)
            loss = self._loss(features[idx], target[idx], field_mask_t[idx])
            loss.backward()
            optimizer.step()
            steps_run = step + 1
            if validation is None or (steps_run % int(log_every) and steps_run != total_steps):
                continue
            metric = self._validation_metric(*validation)
            n_evals += 1
            if metric < best_metric - float(early_stop_min_delta):
                best_metric, best_step, best_state, bad_evals = metric, steps_run, deepcopy(self._net.state_dict()), 0
            else:
                bad_evals += 1
            if early_stop_patience_evals is not None and bad_evals >= int(early_stop_patience_evals):
                early_stopped = True
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._net.eval()
        with torch.no_grad():
            final_loss = self._loss(features[stats_idx], target[stats_idx], field_mask_t[stats_idx])
        self.fit_stats_ = {
            "steps_requested": int(num_steps),
            "hard_stop_step": None if hard_stop_step is None else int(hard_stop_step),
            "steps_run": int(steps_run),
            "early_stop_triggered": bool(early_stopped),
            "best_step": None if best_step is None else int(best_step),
            "best_val_equal_base_normalized_rmse_grid": None if best_step is None else float(best_metric),
            "n_evals": int(n_evals),
            "initial_train_normalized_loss": float(initial_loss.detach().item()),
            "final_train_normalized_loss": float(final_loss.detach().item()),
            "final_training_loss": float(final_loss.detach().item()),
            "seed_channels": self.seed_channels,
            "convs_per_stage": self.convs_per_stage,
            "stage_channels": [self.seed_channels, self.seed_channels // 2, self.seed_channels // 4],
            "seed_grid": [4, 8],
            "batch_size": self.batch_size,
            "optimizer": "AdamW",
            "optimizer_lr": float(lr),
            "optimizer_weight_decay": float(weight_decay),
            "normalization": "observed_training_latitude_weighted_per_field",
            "loss": "latitude_weighted_masked_mse_divided_by_weighted_valid_mass",
        }

    @torch.no_grad()
    def predict(self, X: np.ndarray | torch.Tensor, s: np.ndarray | torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Model not fitted.")
        self._net.eval()
        features = self._features(
            _tensor(X, device=self.device, dtype=self.dtype),
            _tensor(s, device=self.device, dtype=torch.long),
        )
        out = self._net(features) * self._field_std[None, :, None, None] + self._field_mean[None, :, None, None]
        return out.detach().cpu().to(torch.float32)

    def _features(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if X.ndim != 2 or X.shape[1] != 8:
            raise ValueError("ConvDecoder expects X with shape (n, 8).")
        if s.ndim != 1 or len(s) != len(X):
            raise ValueError("ConvDecoder simulation-type labels must have shape (n,).")
        if torch.any((s < 0) | (s >= self.n_sim_types_)):
            raise ValueError(
                f"ConvDecoder simulation-type labels must be in [0, {self.n_sim_types_})."
            )
        return torch.cat(
            [X, F.one_hot(s, self.n_sim_types_).to(dtype=self.dtype)], dim=1
        )

    def _loss(self, features: torch.Tensor, target: torch.Tensor, field_mask: torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Model not fitted.")
        return _masked_latitude_weighted_mse(self._net(features), target, field_mask, self._lat_weights)

    @torch.no_grad()
    def _validation_metric(
        self,
        X: np.ndarray | torch.Tensor,
        s: np.ndarray | torch.Tensor,
        Y: np.ndarray,
        field_mask: np.ndarray,
    ) -> float:
        pred = enforce_equatorial_symmetry_grid(self.predict(X, s).numpy(), self.field_names_)
        return area_weighted_equal_base_variable_normalized_rmse_grid(
            pred,
            Y,
            self.field_names_,
            self._train_Y_np,
            self._train_field_mask_np,
            field_mask,
        )
