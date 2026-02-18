"""Checkpoint management for TCFP model training."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from tcfp.nn import TCFPLinear


class TCFPCheckpointer:
    """4-tier checkpoint manager for TCFP training.

    Tiers (by step frequency):
      0 — metadata only (every meta_interval steps)
      1 — + EF state    (every ef_interval steps)
      2 — + full model  (every full_interval steps)
      best — separate directory updated whenever loss improves

    ErrorFeedbackState is serialized completely (all 5 dicts) to support
    correct resume of delayed scaling.

    All ``torch.load`` calls use ``weights_only=False`` because the EF state
    dict contains Python ints and floats alongside tensors. This is required
    for PyTorch ≥ 2.6 where ``weights_only=True`` is the default.

    Checkpoint directory layout::

        checkpoints/
          checkpoint_step_00001000/
            metadata.json
            ef_state.pt
            model.pt          (only at full_interval)
          checkpoint_step_00002000/
            ...
          best/
            metadata.json
            ef_state.pt
            model.pt

    Args:
        checkpoint_dir: Root directory for checkpoints.
        meta_interval: Save metadata every N steps.
        ef_interval: Save EF state every N steps (must be a multiple of
            meta_interval for sensible behavior).
        full_interval: Save full model weights every N steps.
        keep_last_n: Number of full checkpoints to retain; older ones deleted.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        meta_interval: int = 10,
        ef_interval: int = 100,
        full_interval: int = 1000,
        keep_last_n: int = 3,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.meta_interval = meta_interval
        self.ef_interval = ef_interval
        self.full_interval = full_interval
        self.keep_last_n = keep_last_n
        self._best_loss: float = float("inf")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model: nn.Module,
        step: int,
        loss: float,
        optimizer: torch.optim.Optimizer | None = None,
        extra_meta: dict[str, object] | None = None,
    ) -> Path:
        """Save a checkpoint for the given training step.

        The tier (0–2) is determined automatically from step modulo intervals.
        A best-loss copy is maintained separately.

        Args:
            model: The model to checkpoint.
            step: Current training step.
            loss: Current loss value (used for best-loss tracking).
            optimizer: If provided, optimizer state is included in full tiers.
            extra_meta: Additional key-value pairs to store in metadata.json.

        Returns:
            Path to the checkpoint directory that was written.
        """
        tier = self._tier_for_step(step)
        ckpt_dir = self.checkpoint_dir / f"checkpoint_step_{step:08d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Metadata (always written)
        meta: dict[str, object] = {"step": step, "loss": loss, "tier": tier}
        if extra_meta:
            meta.update(extra_meta)
        with open(ckpt_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        # EF state (tier >= 1)
        if tier >= 1:
            self._save_ef_state(ckpt_dir, model)

        # Full model (tier >= 2)
        if tier >= 2:
            save_dict: dict[str, object] = {"model": model.state_dict()}
            if optimizer is not None:
                save_dict["optimizer"] = optimizer.state_dict()
            torch.save(save_dict, ckpt_dir / "model.pt")

        # Best-loss checkpoint
        if loss < self._best_loss:
            self._best_loss = loss
            best_dir = self.checkpoint_dir / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(ckpt_dir, best_dir)
            # Ensure best always has EF state and full model weights
            if tier < 1:
                self._save_ef_state(best_dir, model)
            if tier < 2:
                save_dict = {"model": model.state_dict()}
                if optimizer is not None:
                    save_dict["optimizer"] = optimizer.state_dict()
                torch.save(save_dict, best_dir / "model.pt")

        self._cleanup_old()
        return ckpt_dir

    def load_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str | Path | None = None,
        tier: str = "full",
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict[str, object]:
        """Load a checkpoint into the model and optionally the optimizer.

        Args:
            model: Target model to restore.
            checkpoint_path: Explicit checkpoint directory. If None, finds the
                latest checkpoint satisfying the requested tier.
            tier: Required tier — "full" needs model.pt, "ef" needs ef_state.pt,
                "meta" needs only metadata.json.
            optimizer: If provided and model.pt exists, restore optimizer state.

        Returns:
            Metadata dict loaded from metadata.json.
        """
        if checkpoint_path is None:
            found = self._find_latest(tier)
            if found is None:
                raise FileNotFoundError(
                    f"No checkpoint with tier '{tier}' found in {self.checkpoint_dir}"
                )
            ckpt_dir = found
        else:
            ckpt_dir = Path(checkpoint_path)

        with open(ckpt_dir / "metadata.json") as f:
            meta: dict[str, object] = json.load(f)

        if (ckpt_dir / "ef_state.pt").exists():
            self._load_ef_state(ckpt_dir, model)

        model_pt = ckpt_dir / "model.pt"
        if model_pt.exists():
            saved = torch.load(model_pt, weights_only=False)
            model.load_state_dict(saved["model"])
            if optimizer is not None and "optimizer" in saved:
                optimizer.load_state_dict(saved["optimizer"])

        return meta

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _tier_for_step(self, step: int) -> int:
        """Return tier 0–2 based on step modulo the configured intervals.

        Note: step=0 always returns tier 2 (full save) because ``0 % N == 0``
        for any N. This is intentional — a full checkpoint at initialization
        provides a clean baseline for resuming from scratch.
        """
        if step % self.full_interval == 0:
            return 2
        if step % self.ef_interval == 0:
            return 1
        return 0

    def _save_ef_state(self, ckpt_dir: Path, model: nn.Module) -> None:
        """Serialize all ErrorFeedbackState dicts for each TCFPLinear in model.

        The outer dict key is ``_param_name`` when non-empty, otherwise the
        module's ``named_modules()`` path is used as a fallback to avoid key
        collisions between standalone layers that all have ``_param_name == ""``.
        """
        ef_snapshot: dict[str, object] = {}
        for mod_name, module in model.named_modules():
            if not isinstance(module, TCFPLinear):
                continue
            ef = module._error_state
            if ef is None:
                continue
            # Use mod_name as fallback key to avoid collisions when _param_name == "".
            outer_key = module._param_name if module._param_name else mod_name
            ef_snapshot[outer_key] = {
                "_buffers": {k: v.cpu() for k, v in ef._buffers.items()},
                "_amax_ema": {k: v.cpu() for k, v in ef._amax_ema.items()},
                "_delayed_amax": {k: v.cpu() for k, v in ef._delayed_amax.items()},
                "_residual_ratio": {k: v.cpu() for k, v in ef._residual_ratio.items()},
                "_step_count": dict(ef._step_count),
                "_ema_decay": ef._ema_decay,
            }
        torch.save(ef_snapshot, ckpt_dir / "ef_state.pt")

    def _load_ef_state(self, ckpt_dir: Path, model: nn.Module) -> None:
        """Restore ErrorFeedbackState from saved ef_state.pt.

        Uses ``weights_only=False`` because the snapshot contains Python ints
        and floats alongside tensors.

        After restore, all tensors are moved to the device of the module's
        weight parameter so that GPU models are not left with CPU-resident EF
        state that would crash on the next forward pass.
        """
        ef_snapshot: dict[str, object] = torch.load(
            ckpt_dir / "ef_state.pt",
            weights_only=False,
        )
        # Build O(1) lookup using the same outer-key logic as _save_ef_state.
        tc_layers: dict[str, TCFPLinear] = {
            (module._param_name if module._param_name else mod_name): module
            for mod_name, module in model.named_modules()
            if isinstance(module, TCFPLinear) and module._error_state is not None
        }
        for outer_key, state in ef_snapshot.items():
            module = tc_layers.get(str(outer_key))
            if module is None or module._error_state is None:
                continue
            ef = module._error_state
            assert isinstance(state, dict)
            buffers = state.get("_buffers", {})
            amax_ema = state.get("_amax_ema", {})
            delayed_amax = state.get("_delayed_amax", {})
            residual_ratio = state.get("_residual_ratio", {})
            step_count = state.get("_step_count", {})
            assert isinstance(buffers, dict)
            assert isinstance(amax_ema, dict)
            assert isinstance(delayed_amax, dict)
            assert isinstance(residual_ratio, dict)
            assert isinstance(step_count, dict)
            # Move tensors to the module's device (buffers were saved as CPU).
            device = module.weight.device
            ef._buffers.update({k: v.to(device) for k, v in buffers.items()})
            ef._amax_ema.update({k: v.to(device) for k, v in amax_ema.items()})
            ef._delayed_amax.update({k: v.to(device) for k, v in delayed_amax.items()})
            ef._residual_ratio.update({k: v.to(device) for k, v in residual_ratio.items()})
            ef._step_count.update(step_count)
            ema_decay = state.get("_ema_decay")
            if ema_decay is not None:
                ef._ema_decay = float(ema_decay)

    def _cleanup_old(self) -> None:
        """Delete oldest full checkpoints beyond keep_last_n."""
        full_ckpts = sorted(
            [
                d
                for d in self.checkpoint_dir.iterdir()
                if d.is_dir()
                and d.name.startswith("checkpoint_step_")
                and (d / "model.pt").exists()
            ],
            key=lambda d: int(d.name.split("_")[-1]),
        )
        for old in full_ckpts[: -self.keep_last_n]:
            shutil.rmtree(old)

    def _find_latest(self, tier: str) -> Path | None:
        """Find the most recent checkpoint directory satisfying the tier."""
        required_files = {
            "full": "model.pt",
            "ef": "ef_state.pt",
            "meta": "metadata.json",
        }
        required = required_files.get(tier, "metadata.json")
        candidates = sorted(
            [
                d
                for d in self.checkpoint_dir.iterdir()
                if d.is_dir()
                and d.name.startswith("checkpoint_step_")
                and (d / required).exists()
            ],
            key=lambda d: int(d.name.split("_")[-1]),
        )
        return candidates[-1] if candidates else None


class ShieldWorldArchiver:
    """Checkpoint-time SVD compression for ErrorFeedbackState buffers.

    Applied at save/load time only — not during training. Reduces checkpoint
    size by storing a low-rank approximation of EF buffers.

    Only the ``_buffers`` sub-dict is compressed; scalar dicts (``_step_count``,
    ``_ema_decay``) are stored verbatim.

    Integrate with ``TCFPCheckpointer`` by passing ``compress_ef=True`` or by
    calling ``compress_ef_state`` / ``decompress_ef_state`` manually.

    Example::

        ef_state = checkpointer._save_ef_state(...)
        compressed = ShieldWorldArchiver.compress_ef_state(ef_state)
        torch.save(compressed, path)

        raw = torch.load(path, weights_only=False)
        ef_state = ShieldWorldArchiver.decompress_ef_state(raw)
        checkpointer._load_ef_state_dict(ef_state, model)
    """

    @staticmethod
    def compress_ef_state(
        ef_state: dict[str, object],
        rank_fraction: float = 0.1,
    ) -> dict[str, object]:
        """Return a new ef_state with EF buffers replaced by SVD factors.

        Args:
            ef_state: Dict as produced by ``TCFPCheckpointer._save_ef_state``.
            rank_fraction: Fraction of min(rows, cols) to retain. Minimum rank 1.

        Returns:
            New ef_state dict with compressed ``_buffers`` sub-dicts.
        """
        compressed: dict[str, object] = {}
        for param_name, state in ef_state.items():
            assert isinstance(state, dict)
            new_state = dict(state)
            buffers = state.get("_buffers", {})
            assert isinstance(buffers, dict)
            compressed_buffers: dict[str, object] = {}
            for key, tensor in buffers.items():
                assert isinstance(tensor, torch.Tensor)
                if tensor.ndim < 2:
                    # 1-D tensors cannot be SVD-compressed
                    compressed_buffers[key] = tensor
                    continue
                t = tensor.float()
                rank = max(1, int(min(t.shape) * rank_fraction))
                try:
                    U, S, Vt = torch.linalg.svd(t, full_matrices=False)
                    compressed_buffers[f"{key}.U"] = U[:, :rank]
                    compressed_buffers[f"{key}.S"] = S[:rank]
                    compressed_buffers[f"{key}.Vt"] = Vt[:rank, :]
                except Exception:
                    # Fall back to uncompressed on SVD failure
                    compressed_buffers[key] = tensor
            new_state["_buffers"] = compressed_buffers
            compressed[param_name] = new_state
        return compressed

    @staticmethod
    def decompress_ef_state(
        compressed: dict[str, object],
    ) -> dict[str, object]:
        """Reconstruct EF buffers from SVD factors.

        Detects compressed entries via the ``.endswith(".U")`` pattern.

        Args:
            compressed: Dict produced by ``compress_ef_state``.

        Returns:
            Decompressed ef_state with full-rank float32 buffers.
        """
        decompressed: dict[str, object] = {}
        for param_name, state in compressed.items():
            assert isinstance(state, dict)
            new_state = dict(state)
            raw_buffers = state.get("_buffers", {})
            assert isinstance(raw_buffers, dict)

            # Identify base keys for SVD triplets via .endswith(".U")
            svd_base_keys: set[str] = {
                k[:-2] for k in raw_buffers if k.endswith(".U")
            }

            restored: dict[str, torch.Tensor] = {}
            for key in raw_buffers:
                if key.endswith((".U", ".S", ".Vt")):
                    continue  # handled as part of an SVD triplet
                restored[key] = raw_buffers[key]  # type: ignore[assignment]

            for base_key in svd_base_keys:
                U_val = raw_buffers.get(f"{base_key}.U")
                S_val = raw_buffers.get(f"{base_key}.S")
                Vt_val = raw_buffers.get(f"{base_key}.Vt")
                if U_val is None or S_val is None or Vt_val is None:
                    continue
                assert isinstance(U_val, torch.Tensor)
                assert isinstance(S_val, torch.Tensor)
                assert isinstance(Vt_val, torch.Tensor)
                # Reconstruct: U * diag(S) * Vt
                restored[base_key] = (
                    U_val.float() * S_val.float().unsqueeze(0)
                ) @ Vt_val.float()

            new_state["_buffers"] = restored
            decompressed[param_name] = new_state
        return decompressed
