"""Curriculum and phase detection utilities for progressive TCFP quantization."""

from __future__ import annotations

import statistics
from dataclasses import dataclass

import torch.nn as nn

from tcfp.core import ErrorFeedbackState
from tcfp.nn import TCFPLinear


@dataclass
class PhaseConfig:
    """Configuration for one phase in a ProgressiveQuantizer schedule.

    Args:
        phase_id: Phase index (0-based, monotonically increasing).
        abd: Whether to enable Asymmetric Backward Decomposition.
        srr: Whether to enable Stochastic Residual Rounding. When advancing
            to a phase with srr=True, the error feedback state is cleared
            (SRR and EF are mutually exclusive).
        hp_grad_weight: Whether to compute weight gradient in high precision.
        min_stability: Stability score [0, 1] required to advance from this
            phase to the next.
    """

    phase_id: int
    abd: bool
    srr: bool
    hp_grad_weight: bool
    min_stability: float


@dataclass
class WaveConfig:
    """Quantization config produced by QuantizationCurriculum for a given step.

    Args:
        aggression: Quantization aggressiveness in [0, 1]. Ramps linearly
            within each wave.
        abd_enabled: Whether ABD should be active.
        srr_enabled: Whether SRR should be active.
        hp_grad_weight: Whether to compute weight gradient in high precision.
    """

    aggression: float
    abd_enabled: bool
    srr_enabled: bool
    hp_grad_weight: bool


_DEFAULT_PHASES: list[PhaseConfig] = [
    PhaseConfig(phase_id=0, abd=False, srr=False, hp_grad_weight=True, min_stability=0.0),
    PhaseConfig(phase_id=1, abd=False, srr=False, hp_grad_weight=True, min_stability=0.70),
    PhaseConfig(phase_id=2, abd=True, srr=False, hp_grad_weight=True, min_stability=0.85),
    PhaseConfig(phase_id=3, abd=True, srr=True, hp_grad_weight=True, min_stability=0.92),
]

_PHASE_NAMES: list[str] = ["base", "error_feedback", "abd", "srr"]


class CUSUMPhaseDetector:
    """CUSUM-based change-point detector for training phase transitions.

    Based on the Lorden-Moustakides CUSUM statistic, this detector identifies
    when the gradient norm distribution shifts from "defensive" (high variance,
    unstable training) to "offensive" (low variance, stable training).

    The CUSUM statistic accumulates evidence of a distribution shift using:
        S_t = max(0, S_{t-1} + drift - deviation)
    where:
        deviation = (grad_norm - mu_off)^2
        drift = (sigma_def^2 - sigma_off^2) / 2

    Args:
        sigma_defensive: Expected std of grad norm in defensive phase.
        sigma_offensive: Expected std of grad norm in offensive phase.
        cusum_threshold: Forward CUSUM value that triggers "offensive" transition.
        allow_reset: If True, a backward CUSUM can re-trigger "defensive".
        reset_threshold: Backward CUSUM value for re-entry to defensive.
        mu_off: Expected mean of grad norm in the offensive (stable) phase.
    """

    def __init__(
        self,
        sigma_defensive: float = 2.0,
        sigma_offensive: float = 0.5,
        cusum_threshold: float = 10.0,
        allow_reset: bool = False,
        reset_threshold: float = 5.0,
        mu_off: float = 1.0,
    ) -> None:
        self.sigma_defensive = sigma_defensive
        self.sigma_offensive = sigma_offensive
        self.cusum_threshold = cusum_threshold
        self.allow_reset = allow_reset
        self.reset_threshold = reset_threshold
        self.mu_off = mu_off

        self._S: float = 0.0
        self._S_back: float = 0.0
        self._phase: str = "defensive"

    @property
    def phase(self) -> str:
        """Current phase: ``'defensive'`` or ``'offensive'``."""
        return self._phase

    @property
    def cusum_statistic(self) -> float:
        """Current value of the forward CUSUM statistic."""
        return self._S

    def update(self, grad_norm: float) -> str:
        """Update CUSUM with the given grad norm and return the new phase.

        Args:
            grad_norm: L2 norm of gradients for this step.

        Returns:
            Current phase string: ``'defensive'`` or ``'offensive'``.
        """
        deviation = (grad_norm - self.mu_off) ** 2
        drift = (self.sigma_defensive**2 - self.sigma_offensive**2) / 2.0
        self._S = max(0.0, self._S + drift - deviation)

        if self._phase == "defensive" and self.cusum_threshold < self._S:
            self._phase = "offensive"
            self._S_back = 0.0

        if self.allow_reset and self._phase == "offensive":
            # Backward CUSUM detects re-entry to defensive
            self._S_back = max(0.0, self._S_back - drift + deviation)
            if self._S_back > self.reset_threshold:
                self._phase = "defensive"
                self._S = 0.0

        return self._phase

    def reset(self) -> None:
        """Reset both CUSUM statistics and return to defensive phase."""
        self._S = 0.0
        self._S_back = 0.0
        self._phase = "defensive"


class ProgressiveQuantizer:
    """Staged enablement of TCFP quantization features gated on loss stability.

    Advances monotonically through phases. A phase advances when:
        stability >= next_phase.min_stability

    where ``stability = 1 / (1 + CV)`` and ``CV = std / mean`` over a rolling
    window of recent losses.

    Phases never revert. Highway layers (``_is_highway=True``) are skipped.

    Args:
        phases: Ordered list of PhaseConfig. Defaults to 4 built-in phases:
            0 (base), 1 (error feedback), 2 (+ABD), 3 (+SRR).
        loss_window: Rolling window size for stability computation.
        verbose: If True, print a message when a phase advances.
    """

    def __init__(
        self,
        phases: list[PhaseConfig] | None = None,
        loss_window: int = 50,
        verbose: bool = True,
    ) -> None:
        self.phases = phases if phases is not None else list(_DEFAULT_PHASES)
        self.loss_window = loss_window
        self.verbose = verbose
        self._current_phase: int = 0
        self._loss_history: list[float] = []

    @property
    def current_phase(self) -> int:
        """Index of the current phase (0-based)."""
        return self._current_phase

    @property
    def phase_name(self) -> str:
        """Human-readable name of the current phase."""
        if self._current_phase < len(_PHASE_NAMES):
            return _PHASE_NAMES[self._current_phase]
        return f"phase_{self._current_phase}"

    def step(self, model: nn.Module, loss: float) -> bool:
        """Record loss, check stability, and advance phase if ready.

        Args:
            model: The model whose TCFPLinear layers will be updated.
            loss: Current step loss value.

        Returns:
            True if the phase advanced this step, False otherwise.
        """
        self._loss_history.append(loss)
        if len(self._loss_history) > self.loss_window:
            self._loss_history = self._loss_history[-self.loss_window :]

        if self._current_phase >= len(self.phases) - 1:
            return False  # Already at final phase

        stability = self._compute_stability()
        next_phase = self.phases[self._current_phase + 1]

        if stability >= next_phase.min_stability:
            self._current_phase += 1
            self._advance_phase(model)
            if self.verbose:
                print(
                    f"[ProgressiveQuantizer] Phase {self._current_phase - 1} "
                    f"→ {self._current_phase} ({self.phase_name}), "
                    f"stability={stability:.3f}"
                )
            return True
        return False

    def _compute_stability(self) -> float:
        """Compute stability score from recent losses: 1 / (1 + CV)."""
        if len(self._loss_history) < 2:
            return 0.0
        losses = self._loss_history[-self.loss_window :]
        mean = statistics.mean(losses)
        if abs(mean) < 1e-12:
            return 1.0
        std = statistics.stdev(losses)
        cv = std / abs(mean)
        return 1.0 / (1.0 + cv)

    def _advance_phase(self, model: nn.Module) -> None:
        """Apply the current phase config to all TCFPLinear modules."""
        config = self.phases[self._current_phase]
        for _, module in model.named_modules():
            if isinstance(module, TCFPLinear):
                self._apply_phase_config(module, config)

    def _apply_phase_config(self, module: TCFPLinear, config: PhaseConfig) -> None:
        """Apply a PhaseConfig to a single TCFPLinear module.

        Highway layers (``_is_highway=True``) are skipped.
        """
        if getattr(module, "_is_highway", False):
            return
        module.abd = config.abd
        module.hp_grad_weight = config.hp_grad_weight
        if config.srr and not module.srr:
            module.srr = True
            module._error_state = None
        elif not config.srr and module.srr:
            module.srr = False
            if module.use_tensor_cores:
                module._error_state = ErrorFeedbackState()


class QuantizationCurriculum:
    """Wave-based quantization curriculum with periodic recovery periods.

    Within each wave, quantization aggression ramps linearly from 0 to 1.
    After each wave, a recovery period resets aggression to 0.

    Args:
        wave_steps: Number of steps per wave (aggressiveness ramp).
        recovery_steps: Number of steps between waves (recovery period).
        max_waves: Maximum number of waves before staying at peak aggression.
    """

    def __init__(
        self,
        wave_steps: int = 500,
        recovery_steps: int = 100,
        max_waves: int = 5,
    ) -> None:
        self.wave_steps = wave_steps
        self.recovery_steps = recovery_steps
        self.max_waves = max_waves

    def get_config(self, step: int) -> WaveConfig:
        """Return the WaveConfig for the given training step.

        Args:
            step: Current training step (0-indexed).

        Returns:
            WaveConfig with aggression and feature flags for this step.
        """
        cycle_len = self.wave_steps + self.recovery_steps
        wave_index = step // cycle_len
        pos_in_cycle = step % cycle_len

        if wave_index >= self.max_waves:
            # Past final wave — maximum aggression, all features enabled
            return WaveConfig(
                aggression=1.0,
                abd_enabled=True,
                srr_enabled=True,
                hp_grad_weight=True,
            )

        in_recovery = pos_in_cycle >= self.wave_steps
        if in_recovery:
            return WaveConfig(
                aggression=0.0,
                abd_enabled=False,
                srr_enabled=False,
                hp_grad_weight=True,
            )

        aggression = pos_in_cycle / max(self.wave_steps - 1, 1)
        return WaveConfig(
            aggression=aggression,
            abd_enabled=aggression >= 0.5,
            srr_enabled=aggression >= 0.8,
            hp_grad_weight=True,
        )

    def apply_to_model(self, model: nn.Module, config: WaveConfig) -> None:
        """Apply a WaveConfig to all TCFPLinear modules in the model.

        Highway layers (``_is_highway=True``) are skipped. When enabling SRR,
        error feedback state is cleared. When disabling SRR on TC layers, a
        fresh ErrorFeedbackState is created.

        Args:
            model: The model to update.
            config: WaveConfig produced by ``get_config``.
        """
        for _, module in model.named_modules():
            if not isinstance(module, TCFPLinear):
                continue
            if getattr(module, "_is_highway", False):
                continue
            module.abd = config.abd_enabled
            module.hp_grad_weight = config.hp_grad_weight
            if config.srr_enabled and not module.srr:
                module.srr = True
                module._error_state = None
            elif not config.srr_enabled and module.srr:
                module.srr = False
                if module.use_tensor_cores:
                    module._error_state = ErrorFeedbackState()
