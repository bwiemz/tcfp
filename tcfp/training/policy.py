"""Layer routing policies and analytics for TCFP model training."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from tcfp.core import ErrorFeedbackState
from tcfp.nn import TCFPLinear


def apply_highway_routing(
    model: nn.Module,
    interval: int = 4,
) -> list[str]:
    """Designate every interval-th TCFPLinear as a highway (full-precision anchor).

    Highway layers receive ``abd=False``, ``srr=False``, ``hp_grad_weight=True``,
    and ``_is_highway=True``. All other TCFPLinear layers are unaffected.

    Enumeration is depth-first (``named_modules()`` order), which is deterministic
    for a given model architecture.

    Args:
        model: The model to configure.
        interval: Every interval-th TCFPLinear (0-indexed) becomes a highway.

    Returns:
        List of module names that were designated as highway layers.
    """
    highway_names: list[str] = []
    tc_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, TCFPLinear)
    ]
    for i, (name, module) in enumerate(tc_layers):
        if i % interval == 0:
            module.abd = False
            module.srr = False
            module.hp_grad_weight = True
            module._is_highway = True  # type: ignore[attr-defined]
            highway_names.append(name)
    return highway_names


@dataclass
class HysteresisState:
    """Per-layer hysteresis state for AdaptiveABDController.

    Attributes:
        layer_name: Name of the module this state tracks.
        ef_norm_ema: Exponential moving average of the EF buffer L2 norm.
        abd_enabled: Whether ABD is currently enabled for this layer.
        shield_steps: Remaining steps before ABD state can change again.
        step_count: Total steps seen for this layer.
    """

    layer_name: str
    ef_norm_ema: float = field(default=0.0)
    abd_enabled: bool = field(default=False)
    shield_steps: int = field(default=0)
    step_count: int = field(default=0)


class AdaptiveABDController:
    """Dynamically enables/disables ABD per layer based on EF buffer norm.

    ABD is safe when EF buffer norms are small (quantization residuals are
    negligible). A hysteresis deadband prevents rapid oscillation between
    enabled and disabled states.

    Falls back gracefully when:
      - ``_error_state is None`` (SRR mode) → EF norm = 0.0 → ABD safe to enable
      - ``_param_name == ""`` (standalone layer) → ``_buffers.get("") is None``
        → EF norm = 0.0 → ABD safe to enable

    Highway layers (``_is_highway=True``) are read-only — their state is
    tracked but not modified.

    Args:
        enable_threshold: EF norm EMA below this → enable ABD.
        disable_threshold: EF norm EMA above this → disable ABD.
        shield_steps: Steps to lock ABD state after any change.
        ema_decay: EMA decay for EF norm smoothing.
    """

    def __init__(
        self,
        enable_threshold: float = 0.05,
        disable_threshold: float = 0.15,
        shield_steps: int = 50,
        ema_decay: float = 0.95,
    ) -> None:
        self.enable_threshold = enable_threshold
        self.disable_threshold = disable_threshold
        self.shield_steps = shield_steps
        self.ema_decay = ema_decay
        self._states: dict[str, HysteresisState] = {}

    def step(self, model: nn.Module) -> dict[str, bool]:
        """Update all per-layer states and apply ABD changes.

        Args:
            model: The model to inspect and update.

        Returns:
            Dict mapping each TCFPLinear module name to its current
            ``abd_enabled`` value.
        """
        result: dict[str, bool] = {}
        for name, module in model.named_modules():
            if not isinstance(module, TCFPLinear):
                continue
            if name not in self._states:
                self._states[name] = HysteresisState(layer_name=name)
            state = self._states[name]
            state.step_count += 1

            # Compute EF buffer norm (0.0 when EF is absent / SRR mode)
            ef_norm = 0.0
            ef = module._error_state
            if ef is not None:
                buf = ef._buffers.get(module._param_name)
                if buf is not None:
                    ef_norm = buf.float().norm().item()

            # Update EMA
            state.ef_norm_ema = (
                self.ema_decay * state.ef_norm_ema + (1.0 - self.ema_decay) * ef_norm
            )

            is_highway = getattr(module, "_is_highway", False)
            if state.shield_steps > 0:
                state.shield_steps -= 1
            elif not is_highway:
                if not state.abd_enabled and state.ef_norm_ema < self.enable_threshold:
                    state.abd_enabled = True
                    module.abd = True
                    state.shield_steps = self.shield_steps
                elif state.abd_enabled and state.ef_norm_ema > self.disable_threshold:
                    state.abd_enabled = False
                    module.abd = False
                    state.shield_steps = self.shield_steps

            result[name] = state.abd_enabled
        return result

    def get_state(self, layer_name: str) -> HysteresisState | None:
        """Return the HysteresisState for the named layer, or None if unseen."""
        return self._states.get(layer_name)


class FisherSensitivityMap:
    """Per-layer sensitivity map using empirical diagonal Fisher information.

    Accumulates ``||grad||^2`` averages in a rolling window via backward hooks
    on each TCFPLinear weight tensor. Periodically updates the sensitivity map
    and can push ABD/SRR configuration changes based on percentile thresholds.

    Critical implementation note: hook closures use a factory function to avoid
    the classic Python loop-variable capture bug. Without the factory, all hooks
    would capture the final value of the loop variable.

    Args:
        model: Model to register hooks on.
        grad_window: Rolling window size for gradient statistics.
        update_interval: Steps between sensitivity map refreshes.
        high_threshold: Percentile above which ABD is disabled (sensitive layer).
        low_threshold: Percentile below which ABD is enabled (insensitive layer).
        respect_highway: If True, skip highway layers in ``push_configs``.
    """

    def __init__(
        self,
        model: nn.Module,
        grad_window: int = 100,
        update_interval: int = 100,
        high_threshold: float = 0.8,
        low_threshold: float = 0.3,
        respect_highway: bool = True,
    ) -> None:
        self.grad_window = grad_window
        self.update_interval = update_interval
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.respect_highway = respect_highway

        self._model = model
        self._grad_history: dict[str, deque[float]] = {}
        self._sensitivity_map: dict[str, float] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._step: int = 0

        self._register_hooks(model)

    def _register_hooks(self, model: nn.Module) -> None:
        """Register gradient accumulation hooks on all TCFPLinear weight tensors."""
        for name, module in model.named_modules():
            if not isinstance(module, TCFPLinear):
                continue
            self._grad_history[name] = deque(maxlen=self.grad_window)
            hook = self._make_hook(name)
            handle = module.weight.register_hook(hook)
            self._hooks.append(handle)

    def _make_hook(self, layer_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """Factory for gradient accumulation closure.

        Each call creates a new scope that captures ``layer_name`` by value,
        avoiding the loop-variable capture bug.
        """

        def hook(grad: torch.Tensor) -> torch.Tensor:
            val = grad.float().pow(2).mean().item()
            self._grad_history[layer_name].append(val)
            return grad

        return hook

    def step(self) -> dict[str, float]:
        """Advance step counter and refresh sensitivity map at update_interval.

        Returns:
            Current sensitivity map (layer name → mean ``||grad||^2``).
        """
        self._step += 1
        if self._step % self.update_interval == 0:
            self._sensitivity_map = {
                name: sum(hist) / len(hist) if hist else 0.0
                for name, hist in self._grad_history.items()
            }
        return dict(self._sensitivity_map)

    def push_configs(self, model: nn.Module | None = None) -> None:
        """Update ABD flags on model layers based on current sensitivity map.

        Layers above the high_threshold percentile → disable ABD and SRR.
        Layers below the low_threshold percentile → enable ABD.

        Args:
            model: Target model. Defaults to the model passed at construction.
        """
        target = model if model is not None else self._model
        if not self._sensitivity_map:
            return

        values = sorted(self._sensitivity_map.values())
        n = len(values)

        def percentile_cut(p: float) -> float:
            idx = min(int(p * n), n - 1)
            return values[idx]

        high_cut = percentile_cut(self.high_threshold)
        low_cut = percentile_cut(self.low_threshold)

        for name, module in target.named_modules():
            if not isinstance(module, TCFPLinear):
                continue
            if self.respect_highway and getattr(module, "_is_highway", False):
                continue
            sens = self._sensitivity_map.get(name)
            if sens is None:
                continue
            if sens >= high_cut:
                # High sensitivity — disable ABD and SRR for safety.
                # When clearing SRR on a TC layer, reinstate error feedback
                # (SRR and EF are mutually exclusive; leaving both off silently
                # loses quantization accuracy).
                module.abd = False
                if module.srr:
                    module.srr = False
                    if module.use_tensor_cores:
                        module._error_state = ErrorFeedbackState()
            elif sens <= low_cut:
                # Low sensitivity — safe to enable ABD
                module.abd = True

    def remove_hooks(self) -> None:
        """Remove all registered gradient hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @property
    def sensitivity_map(self) -> dict[str, float]:
        """Current sensitivity map: layer name → mean ``||grad||^2``."""
        return dict(self._sensitivity_map)


class GradientBiasCorrector:
    """Corrects ABD gradient bias using the control variates method.

    On calibration steps, both the ABD gradient (g_hi, from W_hi only) and
    the full gradient (g_full, from W_hi + W_lo contribution) are computed.
    The per-step difference ``g_full - g_hi`` is tracked as a per-layer EMA:

        bias_buffer[layer] = EMA(g_full - g_hi)

    On non-calibration steps, the stored bias is added back:

        g_corrected = g_hi + bias_buffer[layer]

    Note: The bias buffer shape matches the input gradient shape, which varies
    per batch. The EMA tracks a noisy signal; this corrector is most effective
    during stable training phases.

    Args:
        calibration_interval: Steps between full-gradient calibration runs.
        momentum: EMA momentum for bias smoothing (higher = slower adaptation).
    """

    def __init__(
        self,
        calibration_interval: int = 100,
        momentum: float = 0.9,
    ) -> None:
        self.calibration_interval = calibration_interval
        self.momentum = momentum
        self._bias: dict[str, torch.Tensor] = {}

    def should_calibrate(self, step: int) -> bool:
        """Return True if this step should compute the full gradient.

        Args:
            step: Current training step.
        """
        return step % self.calibration_interval == 0

    def update_bias(
        self,
        layer_name: str,
        grad_hi: torch.Tensor,
        grad_full: torch.Tensor,
    ) -> None:
        """Update the EMA bias estimate for a layer from a calibration step.

        Args:
            layer_name: Identifier for the layer.
            grad_hi: ABD gradient (dX from W_hi only).
            grad_full: Full gradient (dX from W_hi + W_lo).
        """
        diff = (grad_full - grad_hi).detach()
        if layer_name not in self._bias:
            self._bias[layer_name] = diff.clone()
        else:
            self._bias[layer_name] = (
                self.momentum * self._bias[layer_name] + (1.0 - self.momentum) * diff
            )

    def correct(
        self,
        layer_name: str,
        grad_hi: torch.Tensor,
    ) -> torch.Tensor:
        """Apply stored bias correction to an ABD gradient.

        If no bias has been estimated for this layer, returns ``grad_hi``
        unchanged.

        Args:
            layer_name: Identifier for the layer.
            grad_hi: ABD gradient to correct.

        Returns:
            Bias-corrected gradient.
        """
        bias = self._bias.get(layer_name)
        if bias is None:
            return grad_hi
        return grad_hi + bias


@dataclass
class MomentumAlignmentState:
    """Per-layer alignment state for MomentumAlignmentTracker.

    Attributes:
        layer_name: Name of the module this state tracks.
        alignment_ema: EMA of cosine similarity between momentum and EF buffer.
            Reset to 0.0 whenever a corrective action fires, so the EMA
            reflects post-intervention data rather than historical drag.
        drag_steps: Consecutive check-intervals where alignment_ema < drag_threshold.
            Resets to 0 after any corrective action.
        checks_since_reset: Check-intervals elapsed since the last corrective action.
            Measured in units of check_interval (incremented only on alignment checks)
            so the guard ``checks_since_reset >= reset_after`` is commensurable with
            ``drag_steps >= reset_after``.
        total_checks: Total alignment checks performed for this layer.
        resets_applied: Number of EF buffer resets applied.
        promoted_to_srr: Whether this layer has been promoted to SRR.
        promoted_to_highway: Whether this layer has been promoted to highway.
    """

    layer_name: str
    alignment_ema: float = field(default=0.0)
    drag_steps: int = field(default=0)
    checks_since_reset: int = field(default=0)
    total_checks: int = field(default=0)
    resets_applied: int = field(default=0)
    promoted_to_srr: bool = field(default=False)
    promoted_to_highway: bool = field(default=False)


class MomentumAlignmentTracker:
    """Detect quantization drag via optimizer–EF geometric alignment.

    Monitors the cosine similarity between Adam's first-moment (exp_avg) buffer
    and the TCFP error feedback buffer for each layer. Persistent negative
    alignment indicates the EF correction is opposing the optimizer's preferred
    direction — "quantization drag" — a failure mode unique to error-feedback
    quantized training with no analog in full-precision or standard
    mixed-precision training.

    Graduated corrective actions triggered by sustained drag:
      Level 1 — EF Reset (drag for reset_after checks): Zero the EF buffer.
          Loses accumulated correction but immediately removes the opposing
          force. The drag counter resets; new observations must accumulate
          before higher-level promotions can trigger.
      Level 2 — SRR Fallback (drag for srr_after checks post-reset):
          Switch to stochastic rounding (srr=True, _error_state=None).
          Eliminates directional EF bias entirely. The layer exits alignment
          tracking once promoted (no EF buffer → no alignment to compute).
      Level 3 — Highway Promotion (direct escalation when drag_steps reaches
          highway_after, bypassing SRR). Fires only when highway_after <
          srr_after + reset_after; with defaults the SRR path fires first.
          Mark as _is_highway=True; grant full-precision anchor protection.

    Falls back gracefully when:
      - Optimizer has no state for a weight (e.g., step 0, non-Adam optimizers)
        → alignment = 0.0 (neutral, no action).
      - EF buffer norm is below magnitude_gate * momentum norm → alignment = 0.0
        (drag force too small to matter).
      - ``_error_state is None`` (SRR mode) → skipped (no EF buffer).
      - ``_is_highway=True`` → skipped (highway layers are externally managed).

    Args:
        optimizer: The model optimizer (expected to be Adam/AdamW with exp_avg).
        drag_threshold: alignment_ema below this triggers drag detection.
        ema_decay: EMA decay for alignment smoothing.
        check_interval: Steps between alignment computations.
        reset_after: Drag checks (in check_interval units) before EF buffer reset.
        srr_after: Drag checks post-reset before SRR promotion.
        highway_after: Total drag checks before direct highway promotion
            (bypasses SRR if reached first).
        magnitude_gate: Skip alignment if e_norm < magnitude_gate * m_norm.
            Prevents false triggers on near-zero EF buffers.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        drag_threshold: float = -0.3,
        ema_decay: float = 0.95,
        check_interval: int = 50,
        reset_after: int = 100,
        srr_after: int = 200,
        highway_after: int = 500,
        magnitude_gate: float = 0.05,
    ) -> None:
        self.optimizer = optimizer
        self.drag_threshold = drag_threshold
        self.ema_decay = ema_decay
        self.check_interval = check_interval
        self.reset_after = reset_after
        self.srr_after = srr_after
        self.highway_after = highway_after
        self.magnitude_gate = magnitude_gate
        self._states: dict[str, MomentumAlignmentState] = {}
        self._step: int = 0

    def step(self, model: nn.Module) -> dict[str, MomentumAlignmentState]:
        """Run one alignment check step and apply corrective actions if needed.

        Should be called once per training step. Alignment is only computed
        every ``check_interval`` steps to minimize overhead.

        Args:
            model: The model being trained.

        Returns:
            Dict mapping each TCFPLinear module name to its current
            ``MomentumAlignmentState``.
        """
        self._step += 1
        should_check = self._step % self.check_interval == 0

        # Build optimizer param → exp_avg lookup once per call.
        # Uses object identity (id) to avoid tensor equality overhead.
        opt_exp_avg: dict[int, torch.Tensor] = {}
        if should_check:
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    state = self.optimizer.state.get(param)
                    if state is not None:
                        exp_avg = state.get("exp_avg")
                        if exp_avg is not None:
                            opt_exp_avg[id(param)] = exp_avg

        result: dict[str, MomentumAlignmentState] = {}
        for name, module in model.named_modules():
            if not isinstance(module, TCFPLinear):
                continue
            if getattr(module, "_is_highway", False):
                continue
            if module._error_state is None:
                # SRR mode — no EF buffer, no drag possible.
                continue

            if name not in self._states:
                self._states[name] = MomentumAlignmentState(layer_name=name)
            state = self._states[name]

            if should_check:
                state.total_checks += 1
                state.checks_since_reset += 1
                alignment = self._get_alignment(module, opt_exp_avg)
                state.alignment_ema = (
                    self.ema_decay * state.alignment_ema
                    + (1.0 - self.ema_decay) * alignment
                )

                if state.alignment_ema < self.drag_threshold:
                    state.drag_steps += 1
                    self._apply_correction(module, state)
                else:
                    state.drag_steps = 0

            result[name] = state

        return result

    def get_state(self, layer_name: str) -> MomentumAlignmentState | None:
        """Return the MomentumAlignmentState for the named layer, or None if unseen."""
        return self._states.get(layer_name)

    @property
    def drag_layers(self) -> list[str]:
        """Layer names currently in the drag regime (alignment_ema < drag_threshold).

        Excludes layers that have been promoted to SRR or highway, even if their
        stale ``alignment_ema`` is still negative (those layers no longer receive
        EMA updates and are no longer managed by this tracker).
        """
        return [
            name
            for name, state in self._states.items()
            if state.alignment_ema < self.drag_threshold
            and not state.promoted_to_srr
            and not state.promoted_to_highway
        ]

    def _get_alignment(
        self,
        module: TCFPLinear,
        opt_exp_avg: dict[int, torch.Tensor],
    ) -> float:
        """Compute cosine similarity between momentum and EF buffer for one layer."""
        momentum = opt_exp_avg.get(id(module.weight))
        if momentum is None:
            return 0.0  # No optimizer state yet (step 0 or non-Adam)

        ef = module._error_state
        if ef is None:
            return 0.0
        buf = ef._buffers.get(module._param_name)
        if buf is None:
            return 0.0

        return self._compute_alignment(momentum, buf)

    def _compute_alignment(
        self,
        momentum: torch.Tensor,
        ef_buffer: torch.Tensor,
    ) -> float:
        """Cosine similarity between two tensors, with degenerate-case guards."""
        m_flat = momentum.flatten().float()
        e_flat = ef_buffer.flatten().float()
        m_norm = m_flat.norm()
        e_norm = e_flat.norm()
        # Gate: if EF is tiny relative to momentum, drag force is negligible.
        if m_norm < 1e-12 or e_norm < self.magnitude_gate * m_norm:
            return 0.0
        return (torch.dot(m_flat, e_flat) / (m_norm * e_norm)).item()

    def _apply_correction(
        self,
        module: TCFPLinear,
        state: MomentumAlignmentState,
    ) -> None:
        """Apply the graduated corrective response based on drag duration.

        Escalation order (first matching condition fires):
          1. Highway: drag_steps >= highway_after → direct highway promotion.
             Bypasses SRR; fires first only when highway_after < srr_after.
          2. SRR: drag_steps >= srr_after AND resets_applied > 0 → SRR fallback.
          3. EF Reset: drag_steps >= reset_after AND checks_since_reset >= reset_after.
        All actions reset drag_steps, checks_since_reset, and alignment_ema to 0.
        """
        # Level 3 — direct highway promotion (bypasses SRR when highway_after < srr_after)
        if not state.promoted_to_highway and state.drag_steps >= self.highway_after:
            module.abd = False
            module.hp_grad_weight = True
            module._is_highway = True  # type: ignore[attr-defined]
            state.promoted_to_highway = True
            state.drag_steps = 0
            state.checks_since_reset = 0
            state.alignment_ema = 0.0
            return

        if state.promoted_to_srr:
            # SRR-promoted layers have no EF buffer; _apply_correction is unreachable
            # for them because step() skips _error_state=None layers. Guard defensively.
            return

        # Level 2 — SRR fallback (requires a prior EF reset to confirm persistent drag)
        if state.drag_steps >= self.srr_after and state.resets_applied > 0:
            module.srr = True
            module._error_state = None
            state.promoted_to_srr = True
            state.drag_steps = 0
            state.checks_since_reset = 0
            state.alignment_ema = 0.0
            return

        # Level 1 — EF buffer reset
        if (
            state.drag_steps >= self.reset_after
            and state.checks_since_reset >= self.reset_after
        ):
            ef = module._error_state
            if ef is not None:
                buf = ef._buffers.get(module._param_name)
                if buf is not None:
                    buf.zero_()
            state.resets_applied += 1
            state.drag_steps = 0
            state.checks_since_reset = 0
            state.alignment_ema = 0.0
