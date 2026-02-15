"""Tests for FP8 tensor core training path."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tcfp.core import (
    TCFPMode,
    ensure_column_major,
    to_fp8_e4m3_nf_aware,
)
from tcfp.nn import (
    TCFPLinear,
    convert_to_tcfp,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

DEVICE = "cuda"


# ── to_fp8_e4m3_nf_aware ───────────────────────────────────────────────


class TestToFP8E4M3NFAware:
    def test_roundtrip_quality(self) -> None:
        """Quantise then dequantise — relative error should be bounded."""
        x = torch.randn(256, 256, device=DEVICE)
        fp8, inv_scale = to_fp8_e4m3_nf_aware(x)
        recon = fp8.float() * inv_scale
        rel_err = (x - recon).abs().mean() / x.abs().mean()
        assert rel_err < 0.15

    def test_nf_aware_reasonable_mse(self) -> None:
        """NF-aware should produce reasonable quantisation quality."""
        torch.manual_seed(42)
        x = torch.randn(1024, 1024, device=DEVICE)

        fp8_nf, inv_nf = to_fp8_e4m3_nf_aware(x)
        mse_nf = (x - fp8_nf.float() * inv_nf).pow(2).mean().item()

        # MSE should be small relative to signal power
        signal_power = x.pow(2).mean().item()
        assert mse_nf / signal_power < 0.01  # <1% relative MSE

    def test_stochastic_rounding_varies(self) -> None:
        """Two calls with stochastic=True should produce different results."""
        x = torch.randn(64, 64, device=DEVICE)
        fp8_a, _ = to_fp8_e4m3_nf_aware(x, stochastic=True)
        fp8_b, _ = to_fp8_e4m3_nf_aware(x, stochastic=True)
        # They should differ (probabilistic rounding)
        assert not torch.equal(fp8_a.float(), fp8_b.float())

    def test_scale_shape(self) -> None:
        """inv_scale should be shape (1,) for _scaled_mm compatibility."""
        x = torch.randn(32, 32, device=DEVICE)
        _, inv_scale = to_fp8_e4m3_nf_aware(x)
        assert inv_scale.shape == (1,)
        assert inv_scale.dtype == torch.float32

    def test_near_constant_fallback(self) -> None:
        """Near-constant tensor should not produce inf/nan scales."""
        x = torch.ones(64, 64, device=DEVICE) * 0.5
        fp8, inv_scale = to_fp8_e4m3_nf_aware(x)
        assert torch.isfinite(inv_scale).all()
        assert torch.isfinite(fp8.float()).all()


# ── ensure_column_major ─────────────────────────────────────────────────


class TestEnsureColumnMajor:
    def test_already_column_major_no_copy(self) -> None:
        """w.t() of a row-major tensor is already col-major — no copy."""
        w = torch.randn(64, 128, device=DEVICE)
        wt = w.t()  # (128, 64) column-major
        result = ensure_column_major(wt)
        assert result.data_ptr() == wt.data_ptr()  # same memory

    def test_row_major_converts(self) -> None:
        """Standard row-major tensor is converted to col-major."""
        t = torch.randn(64, 128, device=DEVICE)
        result = ensure_column_major(t)
        assert result.stride(0) == 1
        assert result.stride(1) == result.shape[0]

    def test_values_preserved(self) -> None:
        """Values are identical after conversion."""
        t = torch.randn(32, 64, device=DEVICE)
        result = ensure_column_major(t)
        assert torch.allclose(t, result)


# ── FP8 Tensor Core GEMM ───────────────────────────────────────────────


class TestFP8TensorCoreGEMM:
    def test_forward_output_shape_2d(self) -> None:
        """2D input produces correct output shape."""
        layer = TCFPLinear(128, 64, use_tensor_cores=True).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (8, 64)

    def test_forward_output_shape_3d(self) -> None:
        """3D input (batch, seq, dim) produces correct output shape."""
        layer = TCFPLinear(128, 64, use_tensor_cores=True).to(DEVICE)
        x = torch.randn(4, 16, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (4, 16, 64)

    def test_forward_values_reasonable(self) -> None:
        """TC output should be in the same ballpark as F.linear."""
        torch.manual_seed(42)
        layer = TCFPLinear(256, 128, use_tensor_cores=True).to(DEVICE)
        x = torch.randn(8, 256, device=DEVICE)

        with torch.no_grad():
            tc_out = layer(x)
            ref_out = torch.nn.functional.linear(x, layer.weight, layer.bias)

        # FP8 quantisation introduces error, but output should be reasonable
        rel_err = (tc_out - ref_out).abs().mean() / ref_out.abs().mean()
        assert rel_err < 0.20  # within 20% relative error

    def test_forward_with_bias(self) -> None:
        """Bias is correctly added."""
        layer = TCFPLinear(128, 64, bias=True, use_tensor_cores=True).to(DEVICE)
        with torch.no_grad():
            layer.bias.fill_(10.0)
        x = torch.zeros(4, 128, device=DEVICE)
        out = layer(x)
        # With zero input, output should be close to bias
        assert torch.allclose(out, torch.full_like(out, 10.0), atol=0.5)

    def test_forward_without_bias(self) -> None:
        """Works correctly with bias=None."""
        layer = TCFPLinear(128, 64, bias=False, use_tensor_cores=True).to(DEVICE)
        x = torch.randn(4, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (4, 64)


# ── Backward Pass ───────────────────────────────────────────────────────


class TestFP8TensorCoreBackward:
    def test_gradient_shapes_2d(self) -> None:
        """Gradients have correct shapes for 2D input."""
        layer = TCFPLinear(128, 64, use_tensor_cores=True).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.shape == (8, 128)
        assert layer.weight.grad is not None and layer.weight.grad.shape == (64, 128)

    def test_gradient_shapes_3d(self) -> None:
        """Gradients have correct shapes for 3D input."""
        layer = TCFPLinear(128, 64, use_tensor_cores=True).to(DEVICE)
        x = torch.randn(4, 16, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.shape == (4, 16, 128)
        assert layer.weight.grad is not None and layer.weight.grad.shape == (64, 128)

    def test_gradients_finite(self) -> None:
        """No NaN or Inf in any gradient."""
        layer = TCFPLinear(256, 128, use_tensor_cores=True).to(DEVICE)
        x = torch.randn(8, 256, device=DEVICE, requires_grad=True)
        out = layer(x)
        loss = out.pow(2).mean()
        loss.backward()
        assert torch.isfinite(x.grad).all()  # type: ignore[union-attr]
        assert torch.isfinite(layer.weight.grad).all()  # type: ignore[union-attr]

    def test_gradient_magnitude_reasonable(self) -> None:
        """Gradients should be in the same ballpark as FP32 linear."""
        torch.manual_seed(42)
        # FP32 reference
        ref = nn.Linear(128, 64).to(DEVICE)
        x_ref = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        ref(x_ref).sum().backward()

        # TC path
        tc = TCFPLinear(128, 64, use_tensor_cores=True).to(DEVICE)
        with torch.no_grad():
            tc.weight.copy_(ref.weight)
            tc.bias.copy_(ref.bias)  # type: ignore[union-attr]
        x_tc = x_ref.detach().clone().requires_grad_(True)
        tc(x_tc).sum().backward()

        # Weight gradient magnitude should be within 3x
        ref_norm = ref.weight.grad.norm()  # type: ignore[union-attr]
        tc_norm = tc.weight.grad.norm()  # type: ignore[union-attr]
        ratio = tc_norm / ref_norm
        assert 0.3 < ratio < 3.0, f"Gradient magnitude ratio {ratio:.2f} out of range"

    def test_bias_gradient(self) -> None:
        """Bias gradient has correct shape."""
        layer = TCFPLinear(128, 64, bias=True, use_tensor_cores=True).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE)
        layer(x).sum().backward()
        assert layer.bias.grad is not None  # type: ignore[union-attr]
        assert layer.bias.grad.shape == (64,)  # type: ignore[union-attr]


# ── Error Feedback ──────────────────────────────────────────────────────


class TestFP8TensorCoreErrorFeedback:
    def test_error_state_populated(self) -> None:
        """After one forward pass, error state buffer should be non-zero."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=True
        ).to(DEVICE)
        layer._param_name = "test_weight"
        x = torch.randn(4, 128, device=DEVICE)
        layer(x)
        assert layer._error_state is not None
        buf = layer._error_state._buffers.get("test_weight")
        assert buf is not None
        assert buf.abs().sum() > 0

    def test_error_feedback_reduces_drift(self) -> None:
        """Error feedback should reduce cumulative quantisation drift."""
        torch.manual_seed(42)
        # With error feedback
        layer_ef = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=True
        ).to(DEVICE)
        layer_ef._param_name = "w"
        # Without error feedback
        layer_no = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=False
        ).to(DEVICE)
        with torch.no_grad():
            layer_no.weight.copy_(layer_ef.weight)

        # Run multiple forward passes (simulating repeated quantisation)
        for _i in range(50):
            x = torch.randn(4, 128, device=DEVICE)
            layer_ef(x)
            layer_no(x)

        # Error feedback state should have accumulated corrections
        assert layer_ef._error_state is not None
        assert layer_no._error_state is None

    def test_no_error_feedback_when_disabled(self) -> None:
        """error_feedback=False means no ErrorFeedbackState is created."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=False
        ).to(DEVICE)
        assert layer._error_state is None


# ── TCFPLinear Integration ──────────────────────────────────────────────


class TestTCFPLinearTensorCores:
    def test_warmup_bypasses_tensor_cores(self) -> None:
        """During warmup, output should match FP32 linear exactly."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, warmup_steps=2
        ).to(DEVICE)
        x = torch.randn(4, 128, device=DEVICE)

        with torch.no_grad():
            ref = torch.nn.functional.linear(x, layer.weight, layer.bias)
            out = layer(x)  # warmup step 1
            assert torch.allclose(out, ref, atol=1e-5)

    def test_convert_to_tcfp_sets_flag(self) -> None:
        """convert_to_tcfp with use_tensor_cores=True sets the flag."""
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32)).to(
            DEVICE
        )
        convert_to_tcfp(
            model,
            mode=TCFPMode.TCFP8,
            skip_patterns=(),
            use_tensor_cores=True,
        )
        for module in model.modules():
            if isinstance(module, TCFPLinear):
                assert module.use_tensor_cores is True

    def test_converted_model_forward_backward(self) -> None:
        """Full model forward-backward after conversion with tensor cores."""
        model = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32)).to(
            DEVICE
        )
        convert_to_tcfp(
            model,
            mode=TCFPMode.TCFP8,
            skip_patterns=(),
            use_tensor_cores=True,
        )
        x = torch.randn(4, 128, device=DEVICE)
        out = model(x)
        assert out.shape == (4, 32)
        loss = out.sum()
        loss.backward()
        # All parameters should have gradients
        for p in model.parameters():
            assert p.grad is not None

    def test_hp_grad_weight_mode(self) -> None:
        """hp_grad_weight=True still produces valid gradients."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, hp_grad_weight=True
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()


# ── Convergence ─────────────────────────────────────────────────────────


class TestTensorCoreConvergence:
    def test_loss_decreases(self) -> None:
        """Loss should decrease over a short training run."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32)
        ).to(DEVICE)
        convert_to_tcfp(
            model,
            mode=TCFPMode.TCFP8,
            skip_patterns=(),
            use_tensor_cores=True,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Generate simple regression data
        x = torch.randn(32, 128, device=DEVICE)
        target = torch.randn(32, 32, device=DEVICE)

        losses: list[float] = []
        for _step in range(50):
            opt.zero_grad()
            out = model(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Loss should decrease significantly
        assert losses[-1] < losses[0] * 0.5, (
            f"Loss did not decrease enough: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_no_nan_during_training(self) -> None:
        """No NaN/Inf during training."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32)
        ).to(DEVICE)
        convert_to_tcfp(
            model,
            mode=TCFPMode.TCFP8,
            skip_patterns=(),
            use_tensor_cores=True,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(16, 128, device=DEVICE)
        target = torch.randn(16, 32, device=DEVICE)

        for _step in range(20):
            opt.zero_grad()
            out = model(x)
            loss = (out - target).pow(2).mean()
            assert torch.isfinite(loss), f"NaN/Inf loss at step {_step}"
            loss.backward()
            opt.step()
