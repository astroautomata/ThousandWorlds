from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch_harmonics.examples.models.sfno import SphericalFourierNeuralOperator

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


def _masked_latitude_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    field_mask: torch.Tensor,
    lat_weights: torch.Tensor,
) -> torch.Tensor:
    error, mass = _masked_latitude_sums(pred, target, field_mask, lat_weights)
    return error / mass.clamp_min(1.0)


def _masked_latitude_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    field_mask: torch.Tensor,
    lat_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = field_mask[:, :, None, None] if field_mask.ndim == 2 else field_mask
    valid = valid & torch.isfinite(target)
    safe_target = torch.where(valid, target, target.new_zeros(()))
    weights = lat_weights.to(device=pred.device, dtype=pred.dtype)[None, None, :, None]
    return (
        (torch.where(valid, torch.square(pred - safe_target), pred.new_zeros(())) * weights).sum(),
        (valid.to(pred.dtype) * weights).sum(),
    )


class SFNO:
    def __init__(
        self,
        *,
        embed_dim: int = 64,
        num_layers: int = 4,
        batch_size: int = 32,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "auto",
    ) -> None:
        self.embed_dim = int(embed_dim)
        self.num_layers = int(num_layers)
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.device = resolve_torch_device(device)
        self._net: SphericalFourierNeuralOperator | None = None
        self.fit_stats_: dict | None = None
        self.n_sim_types_ = 5

    def _build_network(
        self, n_fields: int, n_sim_types: int = 5
    ) -> SphericalFourierNeuralOperator:
        self.n_sim_types_ = int(n_sim_types)
        if self.n_sim_types_ < 1:
            raise ValueError("SFNO requires at least one simulation type.")
        self._net = SphericalFourierNeuralOperator(
            img_size=(32, 64),
            grid="legendre-gauss",
            grid_internal="legendre-gauss",
            scale_factor=1,
            in_chans=8 + self.n_sim_types_,
            out_chans=int(n_fields),
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            activation_function="gelu",
            encoder_layers=1,
            use_mlp=True,
            mlp_ratio=2.0,
            drop_rate=0.0,
            drop_path_rate=0.0,
            normalization_layer="none",
            hard_thresholding_fraction=22 / 32,
            residual_prediction=False,
            pos_embed="learnable latlon",
            bias=False,
        )
        complex_dtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
        self._net._apply(
            lambda tensor: tensor.to(
                device=self.device,
                dtype=complex_dtype if tensor.is_complex() else self.dtype if tensor.is_floating_point() else tensor.dtype,
            )
        )
        return self._net

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
            raise ValueError("SFNO requires at least one simulation type.")
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        self.field_names_ = list(field_names)
        self.n_sim_types_ = n_sim_types
        X_t = _tensor(X, device=self.device, dtype=self.dtype)
        s_t = _tensor(s, device=self.device, dtype=torch.long)
        Y_t = _tensor(Y, device=self.device, dtype=self.dtype)
        fm_t = _tensor(field_mask, device=self.device, dtype=torch.bool)
        self._features(X_t, s_t)
        if Y_t.shape[-2:] != (32, 64):
            raise ValueError("SFNO expects 32x64 target grids.")
        self._train_Y_np = np.asarray(Y, dtype=np.float32)
        self._train_field_mask_np = np.asarray(field_mask, dtype=bool)
        self._lat_weights = torch.as_tensor(latitude_weights(32), device=self.device, dtype=self.dtype)
        self._field_mean, self._field_std = field_norm_stats(Y_t, fm_t, self._lat_weights)
        target, valid = self._normalise_targets(Y_t, fm_t)
        net = self._build_network(len(self.field_names_), self.n_sim_types_)
        optimizer = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        total_steps = min(int(num_steps), int(hard_stop_step)) if hard_stop_step is not None else int(num_steps)
        if total_steps < 1:
            raise ValueError("num_steps / hard_stop_step must leave at least one training step.")

        net.train()
        initial_loss = self._dataset_loss(X_t, s_t, target, valid)
        val_pack = None
        if all(value is not None for value in (val_X, val_s, val_Y, val_field_mask)):
            val_pack = (
                _tensor(val_X, device=self.device, dtype=self.dtype),
                _tensor(val_s, device=self.device, dtype=torch.long),
                np.asarray(val_Y, dtype=np.float32),
                np.asarray(val_field_mask, dtype=bool),
            )
        best_metric, best_state, best_step = float("inf"), None, None
        bad_evals = steps_run = n_evals = 0
        early_stop = False

        for step in range(total_steps):
            idx = torch.randint(len(X_t), (min(self.batch_size, len(X_t)),), generator=generator, device=self.device)
            net.train()
            optimizer.zero_grad(set_to_none=True)
            loss = _masked_latitude_mse(net(self._input_grid(X_t[idx], s_t[idx])), target[idx], valid[idx], self._lat_weights)
            loss.backward()
            optimizer.step()
            steps_run = step + 1
            if val_pack is None or (steps_run % int(log_every) and steps_run != total_steps):
                continue
            metric = self._validation_metric(*val_pack)
            n_evals += 1
            if metric < best_metric - float(early_stop_min_delta):
                best_metric, best_step, best_state, bad_evals = metric, steps_run, deepcopy(net.state_dict()), 0
            else:
                bad_evals += 1
            print(
                f"[sfno] iter {steps_run:>5}/{total_steps} | "
                f"train_mse={float(loss.detach()):.6e} | val_eq_base_nrmse={metric:.6f} | best={best_metric:.6f}"
            )
            if early_stop_patience_evals is not None and bad_evals >= int(early_stop_patience_evals):
                early_stop = True
                break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        final_loss = self._dataset_loss(X_t, s_t, target, valid)
        self.fit_stats_ = {
            "steps_requested": int(num_steps),
            "hard_stop_step": None if hard_stop_step is None else int(hard_stop_step),
            "steps_run": int(steps_run),
            "early_stop_triggered": bool(early_stop),
            "best_step": None if best_step is None else int(best_step),
            "best_val_equal_base_normalized_rmse_grid": None if best_step is None else float(best_metric),
            "n_evals": int(n_evals),
            "initial_train_normalized_latitude_weighted_mse": initial_loss,
            "final_train_normalized_latitude_weighted_mse": final_loss,
            "embed_dim": self.embed_dim,
            "num_layers": self.num_layers,
            "lmax_exclusive": 22,
            "mmax_exclusive": 22,
            "position_embedding": "learnable latlon",
            "batch_size": self.batch_size,
            "optimizer": "AdamW",
            "optimizer_lr": float(lr),
            "optimizer_weight_decay": float(weight_decay),
            "target_normalization": "per_field_training_grid_latitude_weighted",
            "spatial_normalization": "none",
            "loss": "masked_latitude_weighted_grid_mse",
        }

    @torch.no_grad()
    def predict(self, X: np.ndarray | torch.Tensor, s: np.ndarray | torch.Tensor) -> torch.Tensor:
        if self._net is None:
            raise RuntimeError("Model not fitted.")
        self._net.eval()
        X_t = _tensor(X, device=self.device, dtype=self.dtype)
        s_t = _tensor(s, device=self.device, dtype=torch.long)
        out = []
        for start in range(0, len(X_t), self.batch_size):
            pred = self._net(self._input_grid(X_t[start : start + self.batch_size], s_t[start : start + self.batch_size]))
            out.append((pred * self._field_std[None, :, None, None] + self._field_mean[None, :, None, None]).cpu())
        return torch.cat(out).to(torch.float32)

    def _input_grid(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        features = self._features(X, s)
        # Constant channels excite only degree zero. The native learned lat-lon
        # embedding is the sole spatial symmetry breaker after the encoder.
        return features[:, :, None, None].expand(-1, -1, 32, 64)

    def _features(self, X: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        if X.ndim != 2 or X.shape[1] != 8:
            raise ValueError("SFNO expects X with shape (n, 8).")
        if s.ndim != 1 or len(s) != len(X):
            raise ValueError("SFNO simulation-type labels must have shape (n,).")
        if torch.any((s < 0) | (s >= self.n_sim_types_)):
            raise ValueError(
                f"SFNO simulation-type labels must be in [0, {self.n_sim_types_})."
            )
        return torch.cat(
            [
                X,
                torch.nn.functional.one_hot(
                    s, self.n_sim_types_
                ).to(dtype=self.dtype),
            ],
            dim=1,
        )

    def _normalise_targets(self, Y: torch.Tensor, field_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = field_mask[:, :, None, None] & torch.isfinite(Y)
        safe = torch.where(valid, Y, Y.new_zeros(()))
        target = (safe - self._field_mean[None, :, None, None]) / self._field_std[None, :, None, None]
        return torch.where(valid, target, target.new_zeros(())), valid

    @torch.no_grad()
    def _dataset_loss(self, X: torch.Tensor, s: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> float:
        error = target.new_zeros(())
        mass = target.new_zeros(())
        for start in range(0, len(X), self.batch_size):
            batch = slice(start, start + self.batch_size)
            batch_error, batch_mass = _masked_latitude_sums(
                self._net(self._input_grid(X[batch], s[batch])),
                target[batch],
                valid[batch],
                self._lat_weights,
            )
            error += batch_error
            mass += batch_mass
        return float((error / mass.clamp_min(1.0)).item())

    def _validation_metric(
        self,
        X: torch.Tensor,
        s: torch.Tensor,
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
