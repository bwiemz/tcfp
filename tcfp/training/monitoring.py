"""Training health monitoring for TCFP models."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from tcfp.nn import TCFPLinear


@dataclass
class TrainingAlert:
    """A single health alert emitted by TCFPMonitor."""

    level: str  # "CRITICAL" | "WARNING" | "INFO"
    layer: str  # module path or "model" for global
    reason: str
    value: float | None
    step: int


@dataclass
class GradientStatus:
    """Per-parameter gradient health status from GradientCorruptionDetector."""

    name: str
    is_corrupt: bool
    is_quarantined: bool
    spike_ratio: float
    quarantine_steps: int


class TCFPMonitor:
    """Three-tier health monitor for TCFP model training.

    Checks are organized by frequency:
      - L1 (every step):              NaN/Inf in loss and gradients → CRITICAL
      - L2 (every l2_interval steps): global grad norm > threshold → WARNING
      - L3 (every l3_interval steps): per-layer EF buffer magnitude → WARNING/CRITICAL

    Args:
        grad_norm_threshold: Global grad norm above this → WARNING alert.
        ef_buffer_threshold: EF buffer mean |x| above this → WARNING; above
            10× this → CRITICAL.
        l2_interval: Steps between global grad norm checks.
        l3_interval: Steps between per-layer EF buffer checks.
    """

    def __init__(
        self,
        grad_norm_threshold: float = 10.0,
        ef_buffer_threshold: float = 1.0,
        l2_interval: int = 10,
        l3_interval: int = 100,
    ) -> None:
        self.grad_norm_threshold = grad_norm_threshold
        self.ef_buffer_threshold = ef_buffer_threshold
        self.l2_interval = l2_interval
        self.l3_interval = l3_interval

    def check(
        self,
        model: nn.Module,
        loss: float | torch.Tensor,
        step: int,
    ) -> list[TrainingAlert]:
        """Run health checks and return any alerts.

        Args:
            model: The model being trained.
            loss: Current step loss (scalar).
            step: Current training step (0-indexed).

        Returns:
            List of TrainingAlert objects (empty if healthy).
        """
        alerts: list[TrainingAlert] = []

        # ── L1: NaN/Inf — every step ──────────────────────────────────────────
        # type: ignore comment: basedpyright can't narrow union through isinstance
        # when torch is unresolvable; the ternary is correct at runtime.
        loss_val = (
            float(loss.item())  # type: ignore[union-attr]
            if isinstance(loss, torch.Tensor)
            else float(loss)
        )
        if not math.isfinite(loss_val):
            alerts.append(
                TrainingAlert(
                    level="CRITICAL",
                    layer="model",
                    reason=f"Loss is non-finite: {loss_val}",
                    value=loss_val,
                    step=step,
                )
            )

        for name, param in model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                alerts.append(
                    TrainingAlert(
                        level="CRITICAL",
                        layer=name,
                        reason="Gradient contains NaN or Inf",
                        value=None,
                        step=step,
                    )
                )

        # ── L2: global grad norm — every l2_interval steps ───────────────────
        if step % self.l2_interval == 0:
            grads = [
                p.grad.float()
                for p in model.parameters()
                if p.grad is not None and torch.isfinite(p.grad).all()
            ]
            if grads:
                total_norm = torch.stack([g.norm() for g in grads]).norm().item()
                if total_norm > self.grad_norm_threshold:
                    alerts.append(
                        TrainingAlert(
                            level="WARNING",
                            layer="model",
                            reason=(
                                f"Global grad norm {total_norm:.2f} exceeds "
                                f"threshold {self.grad_norm_threshold}"
                            ),
                            value=total_norm,
                            step=step,
                        )
                    )

        # ── L3: per-layer EF buffer magnitude — every l3_interval steps ──────
        if step % self.l3_interval == 0:
            for mod_name, module in model.named_modules():
                if not isinstance(module, TCFPLinear):
                    continue
                ef = module._error_state
                if ef is None:
                    continue
                buf = ef._buffers.get(module._param_name)
                if buf is None:
                    continue
                mean_abs = buf.float().abs().mean().item()
                if mean_abs > 10.0 * self.ef_buffer_threshold:
                    alerts.append(
                        TrainingAlert(
                            level="CRITICAL",
                            layer=mod_name,
                            reason=(
                                f"EF buffer mean |x| {mean_abs:.4f} >> "
                                f"threshold {self.ef_buffer_threshold}"
                            ),
                            value=mean_abs,
                            step=step,
                        )
                    )
                elif mean_abs > self.ef_buffer_threshold:
                    alerts.append(
                        TrainingAlert(
                            level="WARNING",
                            layer=mod_name,
                            reason=(
                                f"EF buffer mean |x| {mean_abs:.4f} > "
                                f"threshold {self.ef_buffer_threshold}"
                            ),
                            value=mean_abs,
                            step=step,
                        )
                    )

        return alerts


class GradientCorruptionDetector:
    """Detects gradient spikes and quarantines layers using adaptive duration.

    Quarantine duration is computed as:
        tau = max(3, ceil(log2(spike_ratio) / log2(1 + snr_normal) * safety_factor))

    No quarantine is triggered during the first ``warmup_steps`` steps because
    the EMA baseline is unreliable.

    Args:
        spike_threshold: spike_ratio above this triggers corruption detection.
        ema_decay: EMA decay for baseline norm tracking.
        snr_normal: Expected relative grad noise in normal training (used in
            tau formula).
        safety_factor: Multiplier on quarantine duration.
        warmup_steps: Steps before quarantine can trigger.
    """

    def __init__(
        self,
        spike_threshold: float = 5.0,
        ema_decay: float = 0.99,
        snr_normal: float = 0.1,
        safety_factor: float = 2.0,
        warmup_steps: int = 50,
    ) -> None:
        self.spike_threshold = spike_threshold
        self.ema_decay = ema_decay
        self.snr_normal = snr_normal
        self.safety_factor = safety_factor
        self.warmup_steps = warmup_steps

        self._ema_norm: dict[str, float] = {}
        self._step_count: dict[str, int] = {}
        self._quarantine: dict[str, int] = {}  # remaining quarantine steps

    def check(self, name: str, grad: torch.Tensor) -> GradientStatus:
        """Check gradient health for a named parameter.

        Updates EMA baseline and quarantine state. Returns current status.
        Call ``tick(name)`` at the end of each step to decrement counters.

        Args:
            name: Unique identifier for this parameter (e.g., module path).
            grad: Current gradient tensor.

        Returns:
            GradientStatus with corruption and quarantine state.
        """
        norm: float = float(grad.float().norm().item())
        step = self._step_count.get(name, 0)
        self._step_count[name] = step + 1

        baseline: float = self._ema_norm.get(name, norm)
        safe_baseline = max(baseline, 1e-12)
        spike_ratio = norm / safe_baseline

        is_corrupt = spike_ratio > self.spike_threshold
        # Update EMA only when the gradient is not a spike; absorbing spike norms
        # into the baseline would elevate future thresholds and mask real events.
        if not is_corrupt:
            self._ema_norm[name] = self.ema_decay * baseline + (1.0 - self.ema_decay) * norm

        quarantine_steps = self._quarantine.get(name, 0)

        if is_corrupt and step >= self.warmup_steps and quarantine_steps == 0:
            log_ratio = math.log2(max(spike_ratio, 1.001))
            log_snr = math.log2(1.0 + self.snr_normal)
            tau = max(3, math.ceil(log_ratio / log_snr * self.safety_factor))
            self._quarantine[name] = tau
            quarantine_steps = tau

        return GradientStatus(
            name=name,
            is_corrupt=is_corrupt,
            is_quarantined=quarantine_steps > 0,
            spike_ratio=spike_ratio,
            quarantine_steps=quarantine_steps,
        )

    def is_quarantined(self, name: str) -> bool:
        """Return True if the named parameter is currently quarantined."""
        return self._quarantine.get(name, 0) > 0

    def tick(self, name: str) -> None:
        """Decrement quarantine counter for a parameter (call once per step)."""
        if self._quarantine.get(name, 0) > 0:
            self._quarantine[name] -= 1
