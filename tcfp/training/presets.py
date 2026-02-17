"""Training configuration presets for TCFP benchmarking and experimentation."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn

from tcfp.nn import TCFPLinear


@dataclass
class TrainingPreset:
    """Named combination of TCFPLinear configuration overrides for benchmarking.

    Six working skulls are implemented. Two additional skulls (mythic, tough_luck)
    are deferred pending deeper infrastructure changes:
      - mythic: Requires ``ErrorFeedbackState`` constructor support for
        ``buffer_dtype=torch.float64``.
      - tough_luck: Requires paired pre/post hook architecture to avoid
        corrupting the FP32 master weight.

    Skulls applied by ``apply()`` (model-level):
      - **iron** (``disable_error_feedback``): Clear ``_error_state`` on all
        TC layers. Trains without cumulative error correction.
      - **catch** (``force_global_scale``): Set ``scale_block_size=None`` on
        all layers, falling back to per-tensor scaling.

    Skulls that are flags only (checked by external training loop code):
      - **famine** (``halve_abd_hysteresis``): Signal to halve hysteresis
        window in ``AdaptiveABDController``.
      - **blind** (``disable_monitoring``): Signal to skip ``TCFPMonitor``
        checks.
      - **iwhbyd** (``verbose_gemm``): Signal to enable verbose GEMM logging.
      - **thunderstorm** (``promote_sensitivity_tier``): Signal to lower ABD
        enable/disable thresholds in ``FisherSensitivityMap``.

    Args:
        name: Human-readable preset name.
        disable_error_feedback: iron — Remove error feedback from all TC layers.
        halve_abd_hysteresis: famine — Halve controller hysteresis window.
        force_global_scale: catch — Disable per-block scaling on all layers.
        disable_monitoring: blind — Skip health monitoring checks.
        verbose_gemm: iwhbyd — Enable verbose GEMM operation logging.
        promote_sensitivity_tier: thunderstorm — Tighten sensitivity thresholds.
    """

    name: str
    disable_error_feedback: bool = field(default=False)  # iron
    halve_abd_hysteresis: bool = field(default=False)  # famine
    force_global_scale: bool = field(default=False)  # catch
    disable_monitoring: bool = field(default=False)  # blind
    verbose_gemm: bool = field(default=False)  # iwhbyd
    promote_sensitivity_tier: bool = field(default=False)  # thunderstorm

    # Modifier weights for difficulty scoring
    _MODIFIER_WEIGHTS: dict[str, float] = field(
        default_factory=lambda: {
            "disable_error_feedback": 0.40,
            "force_global_scale": 0.20,
            "promote_sensitivity_tier": 0.20,
            "halve_abd_hysteresis": 0.10,
            "verbose_gemm": 0.05,
            "disable_monitoring": 0.05,
        },
        repr=False,
        compare=False,
    )

    def apply(self, model: nn.Module) -> None:
        """Apply model-level configuration overrides.

        Only ``iron`` (disable_error_feedback) and ``catch`` (force_global_scale)
        modify model attributes directly. The remaining flags are signals for
        external training loop components to query via the preset attributes.

        Args:
            model: The model to modify.
        """
        for _, module in model.named_modules():
            if not isinstance(module, TCFPLinear):
                continue
            if self.disable_error_feedback:
                module._error_state = None
            if self.force_global_scale:
                module.scale_block_size = None

    def difficulty_score(self) -> float:
        """Weighted sum of active modifiers.

        Higher scores indicate more aggressive / destabilizing configurations.
        Range is approximately 0.0 (default) to 1.0 (all skulls active).

        Returns:
            Difficulty score in [0, 1].
        """
        total = 0.0
        for attr, weight in self._MODIFIER_WEIGHTS.items():
            if getattr(self, attr, False):
                total += weight
        return total

    # ------------------------------------------------------------------
    # Preset factories
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> TrainingPreset:
        """Baseline preset — no modifiers active."""
        return cls(name="default")

    @classmethod
    def aggressive(cls) -> TrainingPreset:
        """famine + thunderstorm: tighter hysteresis and sensitivity promotion."""
        return cls(
            name="aggressive",
            halve_abd_hysteresis=True,
            promote_sensitivity_tier=True,
        )

    @classmethod
    def maximum(cls) -> TrainingPreset:
        """iron + famine + catch: remove EF, halve hysteresis, global scale only."""
        return cls(
            name="maximum",
            disable_error_feedback=True,
            halve_abd_hysteresis=True,
            force_global_scale=True,
        )
