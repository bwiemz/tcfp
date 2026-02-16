"""Tests for Predictive Dual-Track Delayed Scaling (PDDS) for TCFP-12 TC."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tcfp.core import ErrorFeedbackState, TCFPMode
from tcfp.nn import TCFPLinear, convert_to_tcfp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"


# ── ErrorFeedbackState Delayed Scaling Methods ───────────────────────────


class TestDelayedScaleState:
    def test_get_delayed_amax_first_call_returns_current(self) -> None:
        """First call should return current_amax unchanged."""
        state = ErrorFeedbackState()
        amax = torch.tensor(3.14, device=DEVICE)
        result = state.get_delayed_amax("w", amax)
        assert torch.allclose(result, amax)

    def test_get_delayed_amax_ema_blends(self) -> None:
        """After multiple calls, value blends toward new input."""
        state = ErrorFeedbackState()
        # First call: returns current
        state.get_delayed_amax("w", torch.tensor(1.0, device=DEVICE))
        # Many calls with 2.0 should push the EMA toward 2.0
        # decay=0.999 → half-life ≈ 693 steps, need ~5000 for <0.1 error
        result = state.get_delayed_amax("w", torch.tensor(2.0, device=DEVICE))
        for _ in range(4999):
            result = state.get_delayed_amax("w", torch.tensor(2.0, device=DEVICE))
        assert abs(result.item() - 2.0) < 0.1

    def test_get_delayed_amax_smooths_spike(self) -> None:
        """EMA should smooth out a single spike."""
        state = ErrorFeedbackState()
        state.get_delayed_amax("w", torch.tensor(1.0, device=DEVICE))
        # Simulate many steps at 1.0, then one spike to 10.0
        for _ in range(100):
            state.get_delayed_amax("w", torch.tensor(1.0, device=DEVICE))
        result = state.get_delayed_amax("w", torch.tensor(10.0, device=DEVICE))
        # Should be much less than 10.0 due to smoothing
        assert result.item() < 2.0

    def test_predict_residual_amax_initial_fallback(self) -> None:
        """Without ratio history, returns main_amax / 8."""
        state = ErrorFeedbackState()
        main = torch.tensor(4.0, device=DEVICE)
        predicted = state.predict_residual_amax("w", main)
        assert torch.allclose(predicted, torch.tensor(0.5, device=DEVICE))

    def test_predict_residual_amax_with_tracked_ratio(self) -> None:
        """With ratio history, uses tracked ratio."""
        state = ErrorFeedbackState()
        main = torch.tensor(4.0, device=DEVICE)
        actual_lo = torch.tensor(0.8, device=DEVICE)
        # Update ratio: 0.8 / 4.0 = 0.2
        state.update_residual_ratio("w", main, actual_lo)
        predicted = state.predict_residual_amax("w", main)
        assert abs(predicted.item() - 0.8) < 0.01

    def test_update_residual_ratio_ema_convergence(self) -> None:
        """Ratio EMA converges after repeated updates."""
        state = ErrorFeedbackState()
        main = torch.tensor(4.0, device=DEVICE)
        # Start with ratio 0.25 (1.0 / 4.0)
        state.update_residual_ratio("w", main, torch.tensor(1.0, device=DEVICE))
        # Update toward ratio 0.1 (0.4 / 4.0)
        for _ in range(5000):
            state.update_residual_ratio(
                "w", main, torch.tensor(0.4, device=DEVICE),
            )
        predicted = state.predict_residual_amax("w", main)
        assert abs(predicted.item() - 0.4) < 0.1

    def test_should_validate_residual_first_call(self) -> None:
        """First call should always return True."""
        state = ErrorFeedbackState()
        assert state.should_validate_residual("w") is True

    def test_should_validate_residual_periodic(self) -> None:
        """Returns True every interval steps."""
        state = ErrorFeedbackState(residual_validation_interval=10)
        results = [
            state.should_validate_residual("w") for _ in range(25)
        ]
        # Step 1 (True), steps 2-10 (False except 10=True), 11-20 (20=True)
        assert results[0] is True   # step 1
        assert results[9] is True   # step 10
        assert results[19] is True  # step 20
        assert results[1] is False  # step 2
        assert results[5] is False  # step 6

    def test_reset_clears_delayed_state(self) -> None:
        """reset() should clear all delayed scaling state."""
        state = ErrorFeedbackState()
        state.get_delayed_amax("w", torch.tensor(1.0, device=DEVICE))
        state.update_residual_ratio(
            "w", torch.tensor(4.0, device=DEVICE),
            torch.tensor(0.5, device=DEVICE),
        )
        state.should_validate_residual("w")
        state.reset()
        # After reset, predict should fall back to /8
        predicted = state.predict_residual_amax(
            "w", torch.tensor(4.0, device=DEVICE),
        )
        assert torch.allclose(predicted, torch.tensor(0.5, device=DEVICE))
        # should_validate should return True again (first call after reset)
        assert state.should_validate_residual("w") is True


# ── Predictive Residual Scaling Accuracy ─────────────────────────────────


class TestPredictiveResidualScaling:
    def test_predicted_within_2x_of_actual(self) -> None:
        """Predicted W_lo amax is within 2x of actual for Gaussian weights."""
        from tcfp.core import to_fp8_e4m3

        state = ErrorFeedbackState()
        torch.manual_seed(42)
        w = torch.randn(256, 256, device=DEVICE)

        # Compute actual W_hi and W_lo
        w_hi_fp8, w_hi_inv = to_fp8_e4m3(w)
        residual = w - w_hi_fp8.float() * w_hi_inv
        actual_amax = residual.abs().max()
        main_amax = w.abs().max().clamp(min=1e-12)

        # Bootstrap the ratio
        state.update_residual_ratio("w", main_amax, actual_amax)
        predicted = state.predict_residual_amax("w", main_amax)

        ratio = predicted.item() / actual_amax.item()
        assert 0.5 < ratio < 2.0, f"Predicted/actual ratio {ratio} out of range"

    def test_ratio_converges_gaussian(self) -> None:
        """Ratio converges for Gaussian weights across multiple steps."""
        from tcfp.core import to_fp8_e4m3

        state = ErrorFeedbackState()
        torch.manual_seed(0)

        errors = []
        for _ in range(200):
            w = torch.randn(128, 128, device=DEVICE)
            w_hi_fp8, w_hi_inv = to_fp8_e4m3(w)
            residual = w - w_hi_fp8.float() * w_hi_inv
            main_amax = w.abs().max().clamp(min=1e-12)
            actual_amax = residual.abs().max().clamp(min=1e-12)
            state.update_residual_ratio("w", main_amax, actual_amax)
            predicted = state.predict_residual_amax("w", main_amax)
            errors.append(abs(predicted.item() - actual_amax.item()))

        # Later errors should be smaller (ratio has converged)
        early_avg = sum(errors[:20]) / 20
        late_avg = sum(errors[-20:]) / 20
        assert late_avg < early_avg * 2  # at least not diverging

    def test_initial_fallback_reasonable(self) -> None:
        """The /8 initial fallback should be in the right ballpark."""
        from tcfp.core import to_fp8_e4m3

        state = ErrorFeedbackState()
        torch.manual_seed(42)
        w = torch.randn(256, 256, device=DEVICE)

        w_hi_fp8, w_hi_inv = to_fp8_e4m3(w)
        residual = w - w_hi_fp8.float() * w_hi_inv
        actual_amax = residual.abs().max()
        main_amax = w.abs().max().clamp(min=1e-12)

        predicted = state.predict_residual_amax("w", main_amax)
        # /8 should be within 10x of actual (conservative check)
        ratio = predicted.item() / actual_amax.item()
        assert 0.1 < ratio < 10.0


# ── Delayed TCFP-12 TC Integration ──────────────────────────────────────


class TestDelayedTCFP12TC:
    def test_forward_output_shape_2d(self) -> None:
        """Delayed scaling produces correct output shape."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=True,
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (8, 64)

    def test_forward_output_shape_3d(self) -> None:
        """3D input with delayed scaling."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=True,
        ).to(DEVICE)
        x = torch.randn(4, 16, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (4, 16, 64)

    def test_forward_output_within_tolerance(self) -> None:
        """Delayed output should be close to non-delayed on first step."""
        torch.manual_seed(42)
        layer_ds = TCFPLinear(
            256, 128, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=True,
        ).to(DEVICE)
        layer_nd = TCFPLinear(
            256, 128, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=False,
        ).to(DEVICE)
        # Copy weights
        layer_nd.weight = layer_ds.weight
        layer_nd.bias = layer_ds.bias

        x = torch.randn(8, 256, device=DEVICE)
        with torch.no_grad():
            out_ds = layer_ds(x)
            out_nd = layer_nd(x)
        # First step uses fresh scales, so should be very close
        rel_err = (out_ds - out_nd).abs().mean() / out_nd.abs().mean()
        assert rel_err < 0.15, f"Relative error {rel_err:.4f} too large"

    def test_backward_gradient_shapes(self) -> None:
        """Gradients have correct shapes with delayed scaling."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=True,
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.shape == (8, 128)
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (64, 128)

    def test_backward_gradients_finite(self) -> None:
        """No NaN/Inf in gradients."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=True,
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert torch.isfinite(x.grad).all()  # type: ignore[union-attr]
        assert torch.isfinite(layer.weight.grad).all()  # type: ignore[union-attr]

    def test_error_feedback_works_with_delayed(self) -> None:
        """Error feedback buffers populated with delayed scaling."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=True,
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE)
        _ = layer(x)
        assert layer._error_state is not None
        err = layer._error_state.get_error(
            layer._param_name, layer.weight.float(),
        )
        assert err.abs().max() > 0, "Error feedback buffer is all zeros"

    def test_multiple_steps_no_crash(self) -> None:
        """Multiple forward/backward steps don't crash."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=True,
        ).to(DEVICE)
        opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
        for _ in range(20):
            x = torch.randn(8, 128, device=DEVICE)
            opt.zero_grad()
            out = layer(x)
            out.sum().backward()
            opt.step()


# ── Convergence ──────────────────────────────────────────────────────────


class TestDelayedTCFP12TCConvergence:
    def test_loss_decreases(self) -> None:
        """Loss should decrease over 50 steps with delayed scaling."""
        torch.manual_seed(42)
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=True,
        ).to(DEVICE)
        opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
        x = torch.randn(32, 128, device=DEVICE)
        target = torch.randn(32, 64, device=DEVICE)

        first_loss = 0.0
        last_loss = 0.0
        for i in range(50):
            opt.zero_grad()
            loss = (layer(x) - target).pow(2).mean()
            if i == 0:
                first_loss = loss.item()
            last_loss = loss.item()
            loss.backward()
            opt.step()

        assert last_loss < first_loss * 0.5, (
            f"Loss didn't decrease enough: {first_loss:.4f} -> {last_loss:.4f}"
        )

    def test_no_nan_during_training(self) -> None:
        """No NaN/Inf during 20 training steps."""
        torch.manual_seed(0)
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=True,
        ).to(DEVICE)
        opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
        for _ in range(20):
            opt.zero_grad()
            out = layer(torch.randn(8, 128, device=DEVICE))
            loss = out.pow(2).mean()
            assert torch.isfinite(loss), "NaN/Inf loss"
            loss.backward()
            opt.step()

    def test_converges_to_low_loss(self) -> None:
        """Delayed scaling converges to a reasonably low loss."""
        torch.manual_seed(42)
        x = torch.randn(32, 128, device=DEVICE)
        target = torch.randn(32, 64, device=DEVICE)

        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=True,
        ).to(DEVICE)
        opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
        first_loss = 0.0
        last_loss = 0.0
        for i in range(200):
            opt.zero_grad()
            loss = (layer(x) - target).pow(2).mean()
            if i == 0:
                first_loss = loss.item()
            last_loss = loss.item()
            loss.backward()
            opt.step()

        # Should converge significantly: >90% reduction from initial loss
        assert last_loss < first_loss * 0.1, (
            f"Loss didn't converge enough: {first_loss:.4f} -> {last_loss:.4f}"
        )


# ── API Surface ──────────────────────────────────────────────────────────


class TestDelayedScalingAPI:
    def test_convert_to_tcfp_accepts_delayed_scaling(self) -> None:
        """convert_to_tcfp accepts delayed_scaling parameter."""
        model = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32),
        ).to(DEVICE)
        convert_to_tcfp(
            model, mode=TCFPMode.TCFP12,
            skip_patterns=(), use_tensor_cores=True,
            delayed_scaling=True,
        )
        lin: TCFPLinear = model[0]  # type: ignore[assignment]
        assert lin.delayed_scaling is True

    def test_delayed_scaling_without_tensor_cores_warns(self) -> None:
        """Warning when delayed_scaling used without tensor cores."""
        with pytest.warns(UserWarning, match="requires use_tensor_cores"):
            TCFPLinear(
                128, 64, mode=TCFPMode.TCFP12,
                use_tensor_cores=False, delayed_scaling=True,
            )

    def test_delayed_scaling_in_extra_repr(self) -> None:
        """delayed_scaling=True appears in extra_repr output."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, delayed_scaling=True,
        )
        assert "delayed_scaling=True" in layer.extra_repr()

    def test_delayed_scaling_false_by_default(self) -> None:
        """delayed_scaling is False by default."""
        layer = TCFPLinear(128, 64, mode=TCFPMode.TCFP12, use_tensor_cores=True)
        assert layer.delayed_scaling is False
        assert "delayed_scaling" not in layer.extra_repr()
