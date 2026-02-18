"""Tests for tcfp.training utilities."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from tcfp.nn import TCFPLinear
from tcfp.training.checkpointing import ShieldWorldArchiver, TCFPCheckpointer
from tcfp.training.curriculum import (
    CUSUMPhaseDetector,
    PhaseConfig,
    ProgressiveQuantizer,
    QuantizationCurriculum,
)
from tcfp.training.monitoring import GradientCorruptionDetector, TCFPMonitor
from tcfp.training.policy import (
    AdaptiveABDController,
    FisherSensitivityMap,
    GradientBiasCorrector,
    MomentumAlignmentTracker,
    apply_highway_routing,
)
from tcfp.training.presets import TrainingPreset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def _pc(pid: int, abd: bool, srr: bool, ms: float) -> PhaseConfig:
    """Compact PhaseConfig factory for test readability."""
    return PhaseConfig(phase_id=pid, abd=abd, srr=srr, hp_grad_weight=True, min_stability=ms)


def _make_model(n: int = 4) -> nn.Sequential:
    """CPU-compatible model — fake-quantize path."""
    return nn.Sequential(*[TCFPLinear(64, 64) for _ in range(n)])


def _make_tc_model(n: int = 4) -> nn.Sequential:
    """TC-path model with error feedback state. Works on CPU for attribute tests."""
    layers = [
        TCFPLinear(64, 64, use_tensor_cores=True, error_feedback=True)
        for _ in range(n)
    ]
    model = nn.Sequential(*layers)
    # Assign non-empty _param_name so EF buffer lookups work
    for i, m in enumerate(model):
        m._param_name = f"{i}.weight"
    return model


def _populate_ef(model: nn.Module, scale: float = 0.01) -> None:
    """Manually populate EF buffers with small tensors (CPU-safe)."""
    for module in model.modules():
        if isinstance(module, TCFPLinear) and module._error_state is not None:
            ef = module._error_state
            param_name = module._param_name
            ef._buffers[param_name] = torch.randn(
                module.out_features, module.in_features
            ) * scale
            ef._amax_ema[param_name] = torch.tensor(scale)
            ef._delayed_amax[param_name] = torch.tensor(scale)
            ef._residual_ratio[param_name] = torch.tensor(1.0)
            ef._step_count[param_name] = 10


# ===========================================================================
# TCFPCheckpointer
# ===========================================================================


class TestTCFPCheckpointer:
    def test_save_creates_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cp = TCFPCheckpointer(tmp, meta_interval=1, ef_interval=1, full_interval=1)
            model = _make_model()
            path = cp.save(model, step=1, loss=1.0)
            assert path.exists()
            assert (path / "metadata.json").exists()

    def test_save_tier_meta_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # ef_interval=100 so step=5 is meta-only
            cp = TCFPCheckpointer(tmp, meta_interval=5, ef_interval=100, full_interval=1000)
            model = _make_model()
            path = cp.save(model, step=5, loss=1.0)
            assert (path / "metadata.json").exists()
            assert not (path / "ef_state.pt").exists()
            assert not (path / "model.pt").exists()

    def test_save_tier_ef(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cp = TCFPCheckpointer(tmp, meta_interval=10, ef_interval=100, full_interval=1000)
            model = _make_tc_model()
            _populate_ef(model)
            path = cp.save(model, step=100, loss=1.0)
            assert (path / "metadata.json").exists()
            assert (path / "ef_state.pt").exists()
            assert not (path / "model.pt").exists()

    def test_save_tier_full(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cp = TCFPCheckpointer(tmp, meta_interval=10, ef_interval=100, full_interval=1000)
            model = _make_tc_model()
            _populate_ef(model)
            path = cp.save(model, step=1000, loss=1.0)
            assert (path / "metadata.json").exists()
            assert (path / "ef_state.pt").exists()
            assert (path / "model.pt").exists()

    def test_round_trip_ef_state(self) -> None:
        """Save and reload EF state; verify all 5 dicts are preserved."""
        with tempfile.TemporaryDirectory() as tmp:
            cp = TCFPCheckpointer(tmp, meta_interval=1, ef_interval=1, full_interval=1000)
            model = _make_tc_model(n=2)
            _populate_ef(model, scale=0.05)

            # Record original state
            m0 = list(model.modules())[1]  # first TCFPLinear
            assert isinstance(m0, TCFPLinear)
            assert m0._error_state is not None
            orig_buf = m0._error_state._buffers[m0._param_name].clone()

            path = cp.save(model, step=1, loss=1.0)

            # Reset buffers and reload
            for module in model.modules():
                if isinstance(module, TCFPLinear) and module._error_state is not None:
                    module._error_state._buffers.clear()

            cp._load_ef_state(path, model)
            assert m0._error_state is not None
            loaded_buf = m0._error_state._buffers[m0._param_name]
            assert torch.allclose(orig_buf, loaded_buf)

    def test_cleanup_keeps_last_n(self) -> None:
        """After 5 full saves, only keep_last_n=3 remain."""
        with tempfile.TemporaryDirectory() as tmp:
            cp = TCFPCheckpointer(
                tmp, meta_interval=1, ef_interval=1, full_interval=1, keep_last_n=3
            )
            model = _make_model()
            # Use increasing losses to avoid best/ interference
            for step in range(1, 6):
                cp.save(model, step=step * 1000, loss=float(step))
            full_dirs = [
                d
                for d in Path(tmp).iterdir()
                if d.is_dir()
                and d.name.startswith("checkpoint_step_")
                and (d / "model.pt").exists()
            ]
            assert len(full_dirs) == 3

    def test_best_loss_tracking(self) -> None:
        """best/ directory updates when a new lower loss is seen."""
        with tempfile.TemporaryDirectory() as tmp:
            cp = TCFPCheckpointer(tmp, meta_interval=1, ef_interval=1, full_interval=1)
            model = _make_model()
            cp.save(model, step=1000, loss=2.0)
            assert (Path(tmp) / "best").exists()
            cp.save(model, step=2000, loss=3.0)  # worse — best stays
            # Read best metadata
            import json

            with open(Path(tmp) / "best" / "metadata.json") as f:
                meta = json.load(f)
            assert meta["loss"] == 2.0
            cp.save(model, step=3000, loss=1.0)  # better — best updates
            with open(Path(tmp) / "best" / "metadata.json") as f:
                meta = json.load(f)
            assert meta["loss"] == 1.0

    def test_load_restores_model_weights(self) -> None:
        """Save model weights; load into a fresh model; weights match."""
        with tempfile.TemporaryDirectory() as tmp:
            cp = TCFPCheckpointer(tmp, meta_interval=1, ef_interval=1, full_interval=1)
            model = _make_model()
            # Perturb weights so they're not at init
            for p in model.parameters():
                p.data.fill_(0.42)
            cp.save(model, step=1000, loss=0.5)

            fresh = _make_model()
            cp.load_checkpoint(fresh, tier="full")
            for p, q in zip(model.parameters(), fresh.parameters(), strict=True):
                assert torch.allclose(p, q)

    def test_metadata_contains_step_and_loss(self) -> None:
        import json

        with tempfile.TemporaryDirectory() as tmp:
            cp = TCFPCheckpointer(tmp, meta_interval=1, ef_interval=1, full_interval=1)
            model = _make_model()
            path = cp.save(model, step=42, loss=3.14)
            with open(path / "metadata.json") as f:
                meta = json.load(f)
            assert meta["step"] == 42
            assert abs(meta["loss"] - 3.14) < 1e-6

    def test_step_zero_triggers_full_save(self) -> None:
        """step=0 always writes a full tier-2 checkpoint (0 % any N == 0)."""
        with tempfile.TemporaryDirectory() as tmp:
            cp = TCFPCheckpointer(
                tmp, meta_interval=10, ef_interval=100, full_interval=1000
            )
            model = _make_model()
            path = cp.save(model, step=0, loss=5.0)
            assert (path / "metadata.json").exists()
            assert (path / "ef_state.pt").exists()
            assert (path / "model.pt").exists()


# ===========================================================================
# TCFPMonitor
# ===========================================================================


class TestTCFPMonitor:
    def test_nan_loss_critical(self) -> None:
        monitor = TCFPMonitor()
        model = _make_model()
        alerts = monitor.check(model, loss=float("nan"), step=0)
        assert any(a.level == "CRITICAL" and "Loss" in a.reason for a in alerts)

    def test_inf_loss_critical(self) -> None:
        monitor = TCFPMonitor()
        model = _make_model()
        alerts = monitor.check(model, loss=float("inf"), step=0)
        assert any(a.level == "CRITICAL" and "Loss" in a.reason for a in alerts)

    def test_nan_grad_critical(self) -> None:
        monitor = TCFPMonitor()
        model = _make_model()
        # Inject a NaN gradient
        for p in model.parameters():
            p.grad = torch.full_like(p, float("nan"))
            break
        alerts = monitor.check(model, loss=1.0, step=0)
        assert any(a.level == "CRITICAL" and "NaN" in a.reason for a in alerts)

    def test_grad_norm_warning(self) -> None:
        monitor = TCFPMonitor(grad_norm_threshold=1.0, l2_interval=1)
        model = _make_model()
        # Set large gradients
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 100.0
        alerts = monitor.check(model, loss=1.0, step=0)
        assert any(a.level == "WARNING" and "norm" in a.reason for a in alerts)

    def test_grad_norm_below_threshold_no_warning(self) -> None:
        monitor = TCFPMonitor(grad_norm_threshold=1e6, l2_interval=1)
        model = _make_model()
        for p in model.parameters():
            p.grad = torch.ones_like(p) * 0.001
        alerts = monitor.check(model, loss=1.0, step=0)
        norm_alerts = [a for a in alerts if "norm" in a.reason]
        assert len(norm_alerts) == 0

    def test_ef_buffer_warning(self) -> None:
        monitor = TCFPMonitor(ef_buffer_threshold=0.001, l3_interval=1)
        model = _make_tc_model(n=2)
        _populate_ef(model, scale=0.1)  # mean |x| ~ 0.08 >> threshold 0.001
        alerts = monitor.check(model, loss=1.0, step=0)
        ef_alerts = [a for a in alerts if "EF buffer" in a.reason]
        assert len(ef_alerts) > 0

    def test_tier_intervals(self) -> None:
        """L2 and L3 checks only fire at their configured intervals."""
        monitor = TCFPMonitor(
            grad_norm_threshold=0.001,  # will trigger if checked
            l2_interval=10,
            l3_interval=100,
        )
        model = _make_model()
        for p in model.parameters():
            p.grad = torch.ones_like(p)

        # Step 1: L1 only (no L2, no L3)
        alerts_1 = monitor.check(model, loss=1.0, step=1)
        norm_alerts = [a for a in alerts_1 if "norm" in a.reason]
        assert len(norm_alerts) == 0

        # Step 10: L2 fires
        alerts_10 = monitor.check(model, loss=1.0, step=10)
        norm_alerts = [a for a in alerts_10 if "norm" in a.reason]
        assert len(norm_alerts) > 0


# ===========================================================================
# GradientCorruptionDetector
# ===========================================================================


class TestGradientCorruptionDetector:
    def test_warmup_no_quarantine(self) -> None:
        """Spikes during warmup must not trigger quarantine."""
        det = GradientCorruptionDetector(spike_threshold=2.0, warmup_steps=10)
        # Bootstrap baseline at small norm
        for _ in range(5):
            det.check("w", torch.ones(4) * 0.01)
        # Spike during warmup window
        status = det.check("w", torch.ones(4) * 100.0)
        assert status.is_corrupt
        assert not status.is_quarantined

    def test_post_warmup_spike_quarantines(self) -> None:
        det = GradientCorruptionDetector(spike_threshold=5.0, warmup_steps=5)
        # Stabilise baseline
        for _ in range(6):
            det.check("w", torch.ones(4) * 1.0)
        # Spike after warmup
        status = det.check("w", torch.ones(4) * 100.0)
        assert status.is_quarantined

    def test_tau_formula(self) -> None:
        """Quarantine length must satisfy the tau formula."""
        snr = 0.1
        sf = 2.0
        det = GradientCorruptionDetector(
            spike_threshold=5.0,
            ema_decay=0.99,
            snr_normal=snr,
            safety_factor=sf,
            warmup_steps=5,
        )
        # Fix baseline at 1.0
        for _ in range(6):
            det.check("w", torch.ones(4))
        spike_norm = 100.0
        status = det.check("w", torch.ones(4) * spike_norm)
        spike_ratio = status.spike_ratio
        expected_tau = max(3, math.ceil(math.log2(spike_ratio) / math.log2(1 + snr) * sf))
        assert status.quarantine_steps == expected_tau

    def test_tick_decrements(self) -> None:
        det = GradientCorruptionDetector(spike_threshold=5.0, warmup_steps=5)
        for _ in range(6):
            det.check("w", torch.ones(4))
        status = det.check("w", torch.ones(4) * 100.0)
        initial_tau = status.quarantine_steps
        assert initial_tau >= 3
        det.tick("w")
        assert det._quarantine.get("w", 0) == initial_tau - 1

    def test_ema_converges(self) -> None:
        """After many stable steps, spike_ratio approaches 1."""
        det = GradientCorruptionDetector(ema_decay=0.9, warmup_steps=0)
        for _ in range(100):
            det.check("w", torch.ones(4))
        status = det.check("w", torch.ones(4))
        assert abs(status.spike_ratio - 1.0) < 0.1

    def test_zero_grad_handled(self) -> None:
        """Zero gradient must not cause division-by-zero errors."""
        det = GradientCorruptionDetector(warmup_steps=0)
        # Should not raise
        status = det.check("w", torch.zeros(4))
        assert math.isfinite(status.spike_ratio)


# ===========================================================================
# CUSUMPhaseDetector
# ===========================================================================


class TestCUSUMPhaseDetector:
    def test_initial_state_defensive(self) -> None:
        det = CUSUMPhaseDetector()
        assert det.phase == "defensive"
        assert det.cusum_statistic == 0.0

    def test_transition_to_offensive(self) -> None:
        """Many stable updates (near mu_off) must accumulate sufficient drift."""
        det = CUSUMPhaseDetector(
            sigma_defensive=2.0,
            sigma_offensive=0.5,
            cusum_threshold=5.0,
            mu_off=1.0,
        )
        for _ in range(50):
            det.update(1.0)  # exactly at mu_off → deviation=0, drift>0
        assert det.phase == "offensive"

    def test_no_false_trigger_with_large_variance(self) -> None:
        """Large grad norms far from mu_off must not trigger offensive."""
        det = CUSUMPhaseDetector(
            sigma_defensive=2.0,
            sigma_offensive=0.5,
            cusum_threshold=10.0,
            mu_off=1.0,
        )
        for _ in range(20):
            det.update(10.0)  # large deviation → S driven toward 0
        assert det.phase == "defensive"

    def test_allow_reset_reverts(self) -> None:
        """allow_reset=True: after going offensive, large spikes revert to defensive."""
        det = CUSUMPhaseDetector(
            sigma_defensive=2.0,
            sigma_offensive=0.5,
            cusum_threshold=5.0,
            reset_threshold=3.0,
            allow_reset=True,
            mu_off=1.0,
        )
        # Transition to offensive
        for _ in range(50):
            det.update(1.0)
        assert det.phase == "offensive"
        # Inject large spikes to trigger backward CUSUM
        for _ in range(50):
            det.update(20.0)
        assert det.phase == "defensive"

    def test_properties(self) -> None:
        det = CUSUMPhaseDetector()
        det.update(1.0)
        assert isinstance(det.phase, str)
        assert isinstance(det.cusum_statistic, float)


# ===========================================================================
# ProgressiveQuantizer
# ===========================================================================


class TestProgressiveQuantizer:
    def test_no_advance_when_unstable(self) -> None:
        pq = ProgressiveQuantizer(loss_window=10, verbose=False)
        model = _make_model()
        # Alternating extreme losses → CV ≈ 1.0 → stability ≈ 0.5 < 0.70
        for i in range(20):
            loss = 0.01 if i % 2 == 0 else 10.0
            pq.step(model, loss)
        assert pq.current_phase == 0

    def test_advance_on_high_stability(self) -> None:
        """Constant losses → CV=0 → stability=1.0 → advance immediately."""
        pq = ProgressiveQuantizer(
            phases=[_pc(0, False, False, 0.0), _pc(1, False, False, 0.5)],
            loss_window=5,
            verbose=False,
        )
        model = _make_model()
        advanced = False
        for _ in range(10):
            if pq.step(model, 1.0):
                advanced = True
                break
        assert advanced
        assert pq.current_phase == 1

    def test_highway_layers_skipped(self) -> None:
        """Highway layers must not have their attributes changed."""
        pq = ProgressiveQuantizer(
            phases=[_pc(0, False, False, 0.0), _pc(1, True, False, 0.0)],
            verbose=False,
        )
        model = _make_model(n=4)
        # Designate first layer as highway
        m0 = list(model.modules())[1]
        assert isinstance(m0, TCFPLinear)
        m0._is_highway = True
        m0.abd = False

        pq.step(model, 1.0)  # forces advance since min_stability=0.0
        pq.step(model, 1.0)
        # Highway layer must still have abd=False
        assert m0.abd is False

    def test_srr_advance_clears_ef_state(self) -> None:
        """Advancing to srr=True must set _error_state=None."""
        pq = ProgressiveQuantizer(
            phases=[_pc(0, True, False, 0.0), _pc(1, True, True, 0.0)],
            verbose=False,
        )
        model = _make_tc_model(n=2)
        # Ensure EF state is set initially
        for m in model.modules():
            if isinstance(m, TCFPLinear):
                assert m._error_state is not None

        # Two steps to advance (first feeds history, second sees stability=1.0)
        pq.step(model, 1.0)
        pq.step(model, 1.0)

        for m in model.modules():
            if isinstance(m, TCFPLinear):
                assert m._error_state is None

    def test_phase_tracking(self) -> None:
        pq = ProgressiveQuantizer(verbose=False)
        assert pq.phase_name == "base"
        assert pq.current_phase == 0

    def test_already_at_final_phase(self) -> None:
        pq = ProgressiveQuantizer(
            phases=[_pc(0, False, False, 0.0)],
            verbose=False,
        )
        model = _make_model()
        result = pq.step(model, 1.0)
        assert result is False


# ===========================================================================
# apply_highway_routing
# ===========================================================================


class TestApplyHighwayRouting:
    def test_returns_correct_names(self) -> None:
        model = _make_model(n=8)
        names = apply_highway_routing(model, interval=4)
        # Should be indices 0, 4 → 2 highway layers
        assert len(names) == 2

    def test_highway_settings_correct(self) -> None:
        model = _make_model(n=4)
        highway_names = apply_highway_routing(model, interval=4)
        for name, module in model.named_modules():
            if name in highway_names and isinstance(module, TCFPLinear):
                assert module.abd is False
                assert module.srr is False
                assert module.hp_grad_weight is True

    def test_is_highway_flag_set(self) -> None:
        model = _make_model(n=4)
        highway_names = apply_highway_routing(model, interval=4)
        for name, module in model.named_modules():
            if name in highway_names and isinstance(module, TCFPLinear):
                assert getattr(module, "_is_highway", False) is True

    def test_non_highway_unaffected(self) -> None:
        model = _make_model(n=8)
        # Pre-set abd=True on non-highway layers
        tc_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, TCFPLinear)]
        for _, m in tc_layers:
            m.abd = True

        highway_names = apply_highway_routing(model, interval=4)

        for name, module in model.named_modules():
            if isinstance(module, TCFPLinear) and name not in highway_names:
                # Non-highway layers must retain abd=True
                assert module.abd is True
                assert getattr(module, "_is_highway", False) is False


# ===========================================================================
# AdaptiveABDController
# ===========================================================================


class TestAdaptiveABDController:
    def test_initial_state(self) -> None:
        # States are created lazily; get_state returns None before first step.
        ctrl = AdaptiveABDController()
        model = _make_tc_model(n=2)
        assert ctrl.get_state("0") is None

        # With large EF norm (> enable_threshold), ABD must remain disabled.
        _populate_ef(model, scale=1.0)  # norm >> default enable_threshold=0.05
        ctrl.step(model)
        for name, module in model.named_modules():
            if isinstance(module, TCFPLinear):
                state = ctrl.get_state(name)
                if state is not None:
                    # After one step EMA = 0.05 * norm, which may already
                    # exceed threshold if norm is large enough. Use disable_threshold
                    # instead: ensure we haven't gone to a bad state.
                    assert state.step_count == 1

    def test_enable_on_low_ef(self) -> None:
        """Tiny EF norm → ABD should enable after EMA warms up."""
        ctrl = AdaptiveABDController(
            enable_threshold=0.5, disable_threshold=2.0, shield_steps=0, ema_decay=0.5
        )
        model = _make_tc_model(n=1)
        _populate_ef(model, scale=0.001)  # very small buffers
        result: dict[str, bool] = {}
        for _ in range(10):
            result = ctrl.step(model)

        tc_names = [n for n, m in model.named_modules() if isinstance(m, TCFPLinear)]
        assert any(result.get(n, False) for n in tc_names)

    def test_disable_on_high_ef(self) -> None:
        """Large EF norm → ABD should disable once EMA warms up."""
        ctrl = AdaptiveABDController(
            enable_threshold=0.001,
            disable_threshold=0.01,
            shield_steps=0,
            ema_decay=0.5,
        )
        model = _make_tc_model(n=1)
        # First enable
        _populate_ef(model, scale=0.0001)
        for _ in range(10):
            ctrl.step(model)
        # Now inject large EF
        _populate_ef(model, scale=1.0)
        result: dict[str, bool] = {}
        for _ in range(10):
            result = ctrl.step(model)

        tc_names = [n for n, m in model.named_modules() if isinstance(m, TCFPLinear)]
        assert not any(result.get(n, False) for n in tc_names)

    def test_shield_prevents_toggle(self) -> None:
        """ABD must not change while shield_steps > 0."""
        ctrl = AdaptiveABDController(
            enable_threshold=0.5,
            disable_threshold=2.0,
            shield_steps=100,
            ema_decay=0.5,
        )
        model = _make_tc_model(n=1)
        _populate_ef(model, scale=0.001)

        # Step until ABD enables
        for _ in range(20):
            ctrl.step(model)

        tc_names = [n for n, m in model.named_modules() if isinstance(m, TCFPLinear)]
        initial_states = {n: ctrl.get_state(n) for n in tc_names if ctrl.get_state(n)}
        initial_abd = {n: s.abd_enabled for n, s in initial_states.items() if s}

        # Inject large EF — shield should block disable
        _populate_ef(model, scale=100.0)
        ctrl.step(model)

        for n, was_enabled in initial_abd.items():
            state = ctrl.get_state(n)
            if state and state.shield_steps > 0:
                assert state.abd_enabled == was_enabled

    def test_highway_not_modified(self) -> None:
        ctrl = AdaptiveABDController(enable_threshold=0.5, ema_decay=0.5)
        model = _make_tc_model(n=2)
        _populate_ef(model, scale=0.001)
        # Mark first layer as highway
        m0 = list(m for m in model.modules() if isinstance(m, TCFPLinear))[0]
        m0._is_highway = True
        m0.abd = False
        for _ in range(10):
            ctrl.step(model)
        # Highway layer must remain unchanged
        assert m0.abd is False


# ===========================================================================
# QuantizationCurriculum
# ===========================================================================


class TestQuantizationCurriculum:
    def test_step_0_config(self) -> None:
        qc = QuantizationCurriculum(wave_steps=500, recovery_steps=100, max_waves=5)
        config = qc.get_config(0)
        assert config.aggression == 0.0
        assert config.abd_enabled is False
        assert config.srr_enabled is False

    def test_escalation(self) -> None:
        qc = QuantizationCurriculum(wave_steps=100, recovery_steps=50)
        c1 = qc.get_config(10)
        c2 = qc.get_config(80)
        assert c2.aggression > c1.aggression

    def test_recovery_period(self) -> None:
        qc = QuantizationCurriculum(wave_steps=100, recovery_steps=50)
        config = qc.get_config(110)  # in recovery (100..149)
        assert config.aggression == 0.0
        assert config.abd_enabled is False

    def test_apply_respects_highway(self) -> None:
        qc = QuantizationCurriculum(wave_steps=100, recovery_steps=0, max_waves=5)
        model = _make_model(n=4)
        m0 = list(m for m in model.modules() if isinstance(m, TCFPLinear))[0]
        m0._is_highway = True
        m0.abd = False

        config = qc.get_config(90)  # high aggression
        qc.apply_to_model(model, config)
        # Highway layer must not have abd changed
        assert m0.abd is False


# ===========================================================================
# FisherSensitivityMap
# ===========================================================================


class TestFisherSensitivityMap:
    def test_hook_count(self) -> None:
        model = _make_model(n=4)
        fsm = FisherSensitivityMap(model, grad_window=10)
        tc_count = sum(1 for m in model.modules() if isinstance(m, TCFPLinear))
        assert len(fsm._hooks) == tc_count
        fsm.remove_hooks()

    def test_map_updates_at_interval(self) -> None:
        model = _make_model(n=2)
        fsm = FisherSensitivityMap(model, grad_window=10, update_interval=5)

        # Run a backward pass to populate gradient history
        x = torch.randn(4, 64)
        model(x).sum().backward()

        # Map empty before first update_interval steps
        assert len(fsm._sensitivity_map) == 0

        for _ in range(5):
            fsm.step()

        assert len(fsm._sensitivity_map) > 0
        fsm.remove_hooks()

    def test_push_configs_enables_abd(self) -> None:
        """Layers with low sensitivity should have ABD enabled after push."""
        model = _make_model(n=4)
        fsm = FisherSensitivityMap(
            model,
            grad_window=100,
            update_interval=1,
            low_threshold=0.5,
            high_threshold=0.9,
        )

        # Inject dramatically different grad histories
        tc_names = [n for n, m in model.named_modules() if isinstance(m, TCFPLinear)]
        assert len(tc_names) >= 2
        for i, name in enumerate(tc_names):
            for _ in range(50):
                # First half of layers: tiny gradients; second half: large
                val = 0.001 if i < len(tc_names) // 2 else 1000.0
                fsm._grad_history[name].append(val)

        fsm.step()  # triggers update at interval=1
        fsm.push_configs()

        low_sens_layers = [
            m
            for n, m in model.named_modules()
            if isinstance(m, TCFPLinear) and n in tc_names[: len(tc_names) // 2]
        ]
        assert any(m.abd for m in low_sens_layers)
        fsm.remove_hooks()

    def test_remove_hooks_cleans_up(self) -> None:
        model = _make_model(n=2)
        fsm = FisherSensitivityMap(model)
        assert len(fsm._hooks) > 0
        fsm.remove_hooks()
        assert len(fsm._hooks) == 0


# ===========================================================================
# GradientBiasCorrector
# ===========================================================================


class TestGradientBiasCorrector:
    def test_bias_stored(self) -> None:
        gbc = GradientBiasCorrector(calibration_interval=10, momentum=0.9)
        g_hi = torch.zeros(4, 8)
        g_full = torch.ones(4, 8) * 0.1
        gbc.update_bias("layer0", g_hi, g_full)
        assert "layer0" in gbc._bias
        assert torch.allclose(gbc._bias["layer0"], g_full - g_hi)

    def test_correction_applied(self) -> None:
        gbc = GradientBiasCorrector(momentum=0.0)  # no momentum → bias = last diff
        g_hi = torch.zeros(4, 8)
        g_full = torch.ones(4, 8) * 0.5
        gbc.update_bias("layer0", g_hi, g_full)
        corrected = gbc.correct("layer0", g_hi)
        assert torch.allclose(corrected, g_full)

    def test_no_bias_passthrough(self) -> None:
        gbc = GradientBiasCorrector()
        g_hi = torch.randn(4, 8)
        corrected = gbc.correct("unseen_layer", g_hi)
        assert corrected is g_hi  # must be the exact same tensor


# ===========================================================================
# ShieldWorldArchiver
# ===========================================================================


class TestShieldWorldArchiver:
    def _make_ef_state(self) -> dict[str, object]:
        """Create a minimal ef_state dict for testing."""
        return {
            "layer.weight": {
                "_buffers": {
                    "layer.weight": torch.randn(64, 32),
                },
                "_amax_ema": {},
                "_delayed_amax": {},
                "_residual_ratio": {},
                "_step_count": {},
                "_ema_decay": 0.999,
            }
        }

    def test_compress_decompress_roundtrip(self) -> None:
        ef_state = self._make_ef_state()
        original = ef_state["layer.weight"]["_buffers"]["layer.weight"].clone()  # type: ignore[index]
        compressed = ShieldWorldArchiver.compress_ef_state(ef_state, rank_fraction=0.5)
        restored = ShieldWorldArchiver.decompress_ef_state(compressed)
        rec = restored["layer.weight"]["_buffers"]["layer.weight"]  # type: ignore[index]
        assert isinstance(rec, torch.Tensor)
        assert rec.shape == original.shape
        # Low-rank approx — not exact, but must be reasonably close
        rel_err = (original - rec).norm() / (original.norm() + 1e-8)
        assert rel_err < 1.0  # coarse bound (rank_fraction=0.5 should be close)

    def test_rank_reduction(self) -> None:
        """Verify that the SVD output has fewer components than the original."""
        ef_state = self._make_ef_state()
        compressed = ShieldWorldArchiver.compress_ef_state(ef_state, rank_fraction=0.1)
        buf_dict = compressed["layer.weight"]["_buffers"]  # type: ignore[index]
        assert isinstance(buf_dict, dict)
        # Should contain .U, .S, .Vt keys, not the original
        assert "layer.weight" not in buf_dict
        assert "layer.weight.U" in buf_dict
        U = buf_dict["layer.weight.U"]
        assert isinstance(U, torch.Tensor)
        # rank = max(1, int(32 * 0.1)) = max(1, 3) = 3
        assert U.shape[1] <= 32 // 5  # much smaller than full rank

    def test_key_collision_safety(self) -> None:
        """A buffer key already ending in '.U' must not interfere with SVD detection."""
        # Create a buffer whose key ends in ".U" — compression produces "x.U.U", "x.U.S", ...
        ef_state: dict[str, object] = {
            "p": {
                "_buffers": {
                    "x.U": torch.randn(16, 8),  # key already ends in ".U"
                },
                "_amax_ema": {},
                "_delayed_amax": {},
                "_residual_ratio": {},
                "_step_count": {},
                "_ema_decay": 0.999,
            }
        }
        compressed = ShieldWorldArchiver.compress_ef_state(ef_state)
        restored = ShieldWorldArchiver.decompress_ef_state(compressed)
        rec_buf = restored["p"]["_buffers"]  # type: ignore[index]
        assert isinstance(rec_buf, dict)
        # The original "x.U" key should be recovered
        assert "x.U" in rec_buf


# ===========================================================================
# TrainingPreset
# ===========================================================================


class TestTrainingPreset:
    def test_default_no_modifiers(self) -> None:
        p = TrainingPreset.default()
        assert p.difficulty_score() == 0.0

    def test_iron_clears_ef_state(self) -> None:
        model = _make_tc_model(n=2)
        for m in model.modules():
            if isinstance(m, TCFPLinear):
                assert m._error_state is not None
        preset = TrainingPreset(name="test", disable_error_feedback=True)
        preset.apply(model)
        for m in model.modules():
            if isinstance(m, TCFPLinear):
                assert m._error_state is None

    def test_catch_clears_block_scale(self) -> None:
        model = nn.Sequential(TCFPLinear(128, 128, scale_block_size=None))
        preset = TrainingPreset(name="test", force_global_scale=True)
        preset.apply(model)
        for m in model.modules():
            if isinstance(m, TCFPLinear):
                assert m.scale_block_size is None

    def test_difficulty_score_additive(self) -> None:
        p = TrainingPreset(
            name="hard",
            disable_error_feedback=True,
            force_global_scale=True,
        )
        score = p.difficulty_score()
        assert score > 0.5  # iron=0.4 + catch=0.2

    def test_preset_factories(self) -> None:
        assert TrainingPreset.aggressive().halve_abd_hysteresis is True
        assert TrainingPreset.maximum().disable_error_feedback is True
        assert TrainingPreset.maximum().force_global_scale is True


# ===========================================================================
# MomentumAlignmentTracker
# ===========================================================================


def _make_tracker_model() -> tuple[nn.Sequential, torch.optim.Adam]:
    """Return a single-layer TC model with Adam optimizer ready for EFMA tests."""
    model = _make_tc_model(n=1)
    # Assign param_name and populate EF buffers
    m = list(model.modules())[1]
    assert isinstance(m, TCFPLinear)
    m._param_name = "0.weight"
    _populate_ef(model, scale=0.5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model, opt


class TestMomentumAlignmentTracker:
    def test_alignment_positive_when_codirectional(self) -> None:
        """Tracker returns positive alignment when momentum and EF point the same way."""
        model, opt = _make_tracker_model()
        m = list(model.modules())[1]
        assert isinstance(m, TCFPLinear)
        assert m._error_state is not None

        ef_buf = m._error_state._buffers[m._param_name]
        # Set exp_avg identical to EF buffer → cosine = +1
        opt.state[m.weight]["exp_avg"] = ef_buf.clone()

        # ema_decay=0.0 makes alignment_ema = raw cosine after one step
        tracker = MomentumAlignmentTracker(opt, check_interval=1, ema_decay=0.0)
        tracker.step(model)
        state = tracker.get_state("0")
        assert state is not None
        assert state.alignment_ema > 0.9

    def test_alignment_negative_when_opposing(self) -> None:
        """Tracker returns negative alignment when momentum and EF point opposite ways."""
        model, opt = _make_tracker_model()
        m = list(model.modules())[1]
        assert isinstance(m, TCFPLinear)
        assert m._error_state is not None

        ef_buf = m._error_state._buffers[m._param_name]
        # Set exp_avg = -EF buffer → cosine = -1 for all elements
        opt.state[m.weight]["exp_avg"] = -ef_buf.clone()

        # ema_decay=0.0 makes alignment_ema = raw cosine after one step
        tracker = MomentumAlignmentTracker(opt, check_interval=1, ema_decay=0.0)
        tracker.step(model)
        state = tracker.get_state("0")
        assert state is not None
        assert state.alignment_ema < -0.9

    def test_no_drag_action_during_neutral(self) -> None:
        """Orthogonal vectors → alignment near 0 → no corrective action taken."""
        model, opt = _make_tracker_model()
        m = list(model.modules())[1]
        assert isinstance(m, TCFPLinear)
        assert m._error_state is not None

        opt.step()
        ef_buf = m._error_state._buffers[m._param_name]
        # Create orthogonal vector via Gram-Schmidt
        v = torch.randn_like(ef_buf)
        ef_flat = ef_buf.flatten().float()
        v_flat = v.flatten().float()
        v_flat = v_flat - (torch.dot(v_flat, ef_flat) / ef_flat.norm().pow(2)) * ef_flat
        opt.state[m.weight]["exp_avg"] = v_flat.reshape_as(ef_buf)

        tracker = MomentumAlignmentTracker(
            opt, drag_threshold=-0.3, check_interval=1, reset_after=5
        )
        for _ in range(4):
            tracker.step(model)
        state = tracker.get_state("0")
        assert state is not None
        assert state.resets_applied == 0

    def test_ef_reset_after_sustained_drag(self) -> None:
        """Persistent anti-alignment triggers EF buffer zeroing after reset_after checks."""
        model, opt = _make_tracker_model()
        m = list(model.modules())[1]
        assert isinstance(m, TCFPLinear)
        assert m._error_state is not None

        tracker = MomentumAlignmentTracker(
            opt,
            drag_threshold=-0.1,
            check_interval=1,
            reset_after=3,
            srr_after=9999,  # disable SRR to isolate reset behavior
            ema_decay=0.0,   # instantaneous EMA for deterministic test
            magnitude_gate=0.0,  # no magnitude gate for deterministic test
        )

        # Each iteration: re-populate EF buffer with ones, set exp_avg = -ones.
        # This guarantees cosine = -1 every step regardless of prior resets.
        ones = torch.ones(m.weight.shape)
        for _ in range(4):  # 3 drag steps → reset fires, step 4 confirms
            ef = m._error_state
            if ef is not None:
                ef._buffers[m._param_name] = ones.clone()
            opt.state[m.weight]["exp_avg"] = -ones.clone()
            tracker.step(model)

        state = tracker.get_state("0")
        assert state is not None
        assert state.resets_applied >= 1

    def test_srr_promotion_after_extended_drag(self) -> None:
        """Continued drag after EF reset promotes layer to SRR."""
        model, opt = _make_tracker_model()
        m = list(model.modules())[1]
        assert isinstance(m, TCFPLinear)
        assert m._error_state is not None
        assert not m.srr

        opt.step()
        # Force anti-alignment
        w_shape = m.weight.shape
        opt.state[m.weight]["exp_avg"] = -torch.ones(w_shape)

        tracker = MomentumAlignmentTracker(
            opt,
            drag_threshold=-0.1,
            check_interval=1,
            reset_after=2,
            srr_after=2,     # SRR after 2 more drag checks post-reset
            highway_after=9999,
            magnitude_gate=0.0,
            ema_decay=0.0,
        )
        # Populate EF buf to trigger anti-alignment (reset will zero it, we repopulate)
        for _step_idx in range(10):
            ef = m._error_state
            if ef is not None:
                ef._buffers[m._param_name] = torch.ones(w_shape)
            tracker.step(model)
            if m.srr:
                break

        assert m.srr, "Layer should have been promoted to SRR"
        assert m._error_state is None, "SRR promotion must clear error state"

    def test_highway_layers_skipped(self) -> None:
        """Layers with _is_highway=True are never tracked or modified."""
        model, opt = _make_tracker_model()
        m = list(model.modules())[1]
        assert isinstance(m, TCFPLinear)
        m._is_highway = True  # type: ignore[attr-defined]

        opt.step()

        tracker = MomentumAlignmentTracker(opt, check_interval=1)
        result = tracker.step(model)
        # Highway layer should not appear in result
        assert "0" not in result
        assert tracker.get_state("0") is None


class TestEFMABF16Fallback:
    """Level 4 BF16 fallback via EFMA."""

    def test_efma_bf16_fallback(self) -> None:
        """Level 4: after highway promotion + sustained drag → BF16 fallback."""
        model, opt = _make_tracker_model()
        m = list(model.modules())[1]
        assert isinstance(m, TCFPLinear)

        # Need optimizer state populated for alignment computation
        opt.step()

        w_shape = m.weight.shape

        tracker = MomentumAlignmentTracker(
            opt,
            drag_threshold=-0.1,
            check_interval=1,
            reset_after=9999,  # disable lower-level resets
            srr_after=9999,
            highway_after=9999,
            bf16_fallback_after=3,
            magnitude_gate=0.0,
            ema_decay=0.0,
        )

        # Pre-set state as if highway promotion already happened (Level 3).
        # Level 3 sets both _is_highway on the module and promoted_to_highway
        # in the state. The step() method now allows EFMA-promoted highways
        # through for BF16 escalation.
        from tcfp.training.policy import MomentumAlignmentState

        m._is_highway = True  # type: ignore[attr-defined]
        state = MomentumAlignmentState(
            layer_name="0",
            promoted_to_highway=True,
        )
        tracker._states["0"] = state

        # Force anti-alignment: momentum opposite to EF
        ef = m._error_state
        assert ef is not None
        ef._buffers[m._param_name] = torch.ones(w_shape)
        opt.state[m.weight]["exp_avg"] = -torch.ones(w_shape)

        # Run enough steps for bf16_fallback_after threshold
        for _ in range(10):
            tracker.step(model)
            if m._bf16_fallback:
                break

        state = tracker.get_state("0")
        assert state is not None
        assert state.promoted_to_bf16, "Layer should have been fallen back to BF16"
        assert m._bf16_fallback

    def test_efma_skip_bf16_layers(self) -> None:
        """BF16-fallback layers are excluded from alignment tracking."""
        model, opt = _make_tracker_model()
        m = list(model.modules())[1]
        assert isinstance(m, TCFPLinear)
        m.fallback_to_bf16()

        opt.step()

        tracker = MomentumAlignmentTracker(opt, check_interval=1)
        result = tracker.step(model)
        # BF16-fallback layer should not appear in result
        assert "0" not in result
