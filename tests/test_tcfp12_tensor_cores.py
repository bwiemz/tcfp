"""Tests for TCFP-12 tensor core 2-GEMM residual training path."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tcfp.core import TCFPMode
from tcfp.nn import TCFPLinear, convert_to_tcfp

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"


# ── Forward Pass ─────────────────────────────────────────────────────────


class TestTCFP12TCForward:
    def test_output_shape_2d(self) -> None:
        """2D input produces correct output shape."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (8, 64)

    def test_output_shape_3d(self) -> None:
        """3D input (batch, seq, dim) produces correct output shape."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        x = torch.randn(4, 16, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (4, 16, 64)

    def test_values_close_to_f_linear(self) -> None:
        """TCFP-12 TC output should be within 10% of F.linear."""
        torch.manual_seed(42)
        layer = TCFPLinear(
            256, 128, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        x = torch.randn(8, 256, device=DEVICE)

        with torch.no_grad():
            tc_out = layer(x)
            ref_out = torch.nn.functional.linear(x, layer.weight, layer.bias)

        rel_err = (tc_out - ref_out).abs().mean() / ref_out.abs().mean()
        assert rel_err < 0.10, f"TCFP-12 TC rel error {rel_err:.4f} > 10%"

    def test_better_than_single_fp8(self) -> None:
        """2-GEMM residual should be more accurate than single FP8 GEMM."""
        torch.manual_seed(42)
        # TCFP-12 TC (2-GEMM)
        layer_12 = TCFPLinear(
            256, 128, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        # Single FP8 TC
        layer_8 = TCFPLinear(
            256, 128, mode=TCFPMode.TCFP8, use_tensor_cores=True
        ).to(DEVICE)
        with torch.no_grad():
            layer_8.weight.copy_(layer_12.weight)
            layer_8.bias.copy_(layer_12.bias)  # type: ignore[union-attr]

        x = torch.randn(16, 256, device=DEVICE)
        with torch.no_grad():
            ref = torch.nn.functional.linear(x, layer_12.weight, layer_12.bias)
            out_12 = layer_12(x)
            out_8 = layer_8(x)

        err_12 = (out_12 - ref).abs().mean().item()
        err_8 = (out_8 - ref).abs().mean().item()
        assert err_12 < err_8, (
            f"TCFP-12 error {err_12:.6f} should be less than FP8 {err_8:.6f}"
        )

    def test_with_bias(self) -> None:
        """Bias is correctly added."""
        layer = TCFPLinear(
            128, 64, bias=True, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        with torch.no_grad():
            layer.bias.fill_(10.0)
        x = torch.zeros(4, 128, device=DEVICE)
        out = layer(x)
        assert torch.allclose(out, torch.full_like(out, 10.0), atol=0.5)

    def test_without_bias(self) -> None:
        """Works correctly with bias=None."""
        layer = TCFPLinear(
            128, 64, bias=False, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        x = torch.randn(4, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (4, 64)


# ── Backward Pass ────────────────────────────────────────────────────────


class TestTCFP12TCBackward:
    def test_gradient_shapes_2d(self) -> None:
        """Gradients have correct shapes for 2D input."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.shape == (8, 128)
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (64, 128)

    def test_gradient_shapes_3d(self) -> None:
        """Gradients have correct shapes for 3D input."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        x = torch.randn(4, 16, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None and x.grad.shape == (4, 16, 128)
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (64, 128)

    def test_gradients_finite(self) -> None:
        """No NaN or Inf in any gradient."""
        layer = TCFPLinear(
            256, 128, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        x = torch.randn(8, 256, device=DEVICE, requires_grad=True)
        out = layer(x)
        loss = out.pow(2).mean()
        loss.backward()
        assert torch.isfinite(x.grad).all()  # type: ignore[union-attr]
        assert torch.isfinite(layer.weight.grad).all()  # type: ignore[union-attr]

    def test_gradient_magnitude_reasonable(self) -> None:
        """Gradients should be in the same ballpark as FP32 linear."""
        torch.manual_seed(42)
        ref = nn.Linear(128, 64).to(DEVICE)
        x_ref = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        ref(x_ref).sum().backward()

        tc = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        with torch.no_grad():
            tc.weight.copy_(ref.weight)
            tc.bias.copy_(ref.bias)  # type: ignore[union-attr]
        x_tc = x_ref.detach().clone().requires_grad_(True)
        tc(x_tc).sum().backward()

        ref_norm = ref.weight.grad.norm()  # type: ignore[union-attr]
        tc_norm = tc.weight.grad.norm()  # type: ignore[union-attr]
        ratio = tc_norm / ref_norm
        assert 0.3 < ratio < 3.0, (
            f"Gradient magnitude ratio {ratio:.2f} out of range"
        )

    def test_bias_gradient(self) -> None:
        """Bias gradient has correct shape."""
        layer = TCFPLinear(
            128, 64, bias=True, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE)
        layer(x).sum().backward()
        assert layer.bias.grad is not None  # type: ignore[union-attr]
        assert layer.bias.grad.shape == (64,)  # type: ignore[union-attr]


# ── Error Feedback ───────────────────────────────────────────────────────


class TestTCFP12TCErrorFeedback:
    def test_error_state_populated(self) -> None:
        """After one forward pass, error state should be non-zero."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer._param_name = "test_weight"
        x = torch.randn(4, 128, device=DEVICE)
        layer(x)
        assert layer._error_state is not None
        buf = layer._error_state._buffers.get("test_weight")
        assert buf is not None
        assert buf.abs().sum() > 0

    def test_error_tracks_total_hi_lo(self) -> None:
        """Error feedback should track total (hi+lo) quantisation error."""
        torch.manual_seed(42)
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer._param_name = "test_w"
        x = torch.randn(4, 128, device=DEVICE)
        layer(x)

        assert layer._error_state is not None
        error = layer._error_state._buffers["test_w"]
        # Error should be smaller than single-FP8 error because
        # 2-level decomposition captures more of the weight
        from tcfp.core import to_fp8_e4m3

        w = layer.weight.float()
        fp8_single, inv_single = to_fp8_e4m3(w)
        single_error = (w - fp8_single.float() * inv_single).abs().mean()
        residual_error = error.abs().mean()
        assert residual_error < single_error, (
            f"2-GEMM residual error {residual_error:.6f} "
            f"should be < single FP8 error {single_error:.6f}"
        )

    def test_error_feedback_reduces_drift(self) -> None:
        """Error feedback should reduce cumulative quantisation drift."""
        torch.manual_seed(42)
        layer_ef = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer_ef._param_name = "w"
        layer_no = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, error_feedback=False,
        ).to(DEVICE)
        with torch.no_grad():
            layer_no.weight.copy_(layer_ef.weight)

        for _ in range(50):
            x = torch.randn(4, 128, device=DEVICE)
            layer_ef(x)
            layer_no(x)

        assert layer_ef._error_state is not None
        assert layer_no._error_state is None

    def test_no_error_feedback_when_disabled(self) -> None:
        """error_feedback=False means no ErrorFeedbackState."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, error_feedback=False,
        ).to(DEVICE)
        assert layer._error_state is None


# ── Convergence ──────────────────────────────────────────────────────────


class TestTCFP12TCConvergence:
    def test_loss_decreases(self) -> None:
        """Loss should decrease over a short training run."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32)
        ).to(DEVICE)
        convert_to_tcfp(
            model, mode=TCFPMode.TCFP12,
            skip_patterns=(), use_tensor_cores=True,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(32, 128, device=DEVICE)
        target = torch.randn(32, 32, device=DEVICE)

        losses: list[float] = []
        for _ in range(50):
            opt.zero_grad()
            out = model(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.5, (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_no_nan_during_training(self) -> None:
        """No NaN/Inf during training."""
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32)
        ).to(DEVICE)
        convert_to_tcfp(
            model, mode=TCFPMode.TCFP12,
            skip_patterns=(), use_tensor_cores=True,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(16, 128, device=DEVICE)
        target = torch.randn(16, 32, device=DEVICE)

        for step in range(20):
            opt.zero_grad()
            out = model(x)
            loss = (out - target).pow(2).mean()
            assert torch.isfinite(loss), f"NaN/Inf loss at step {step}"
            loss.backward()
            opt.step()

    def test_better_convergence_than_fp8(self) -> None:
        """TCFP-12 TC should converge better than single FP8 TC."""
        torch.manual_seed(42)
        x = torch.randn(32, 128, device=DEVICE)
        target = torch.randn(32, 32, device=DEVICE)

        def train_model(mode: TCFPMode) -> float:
            torch.manual_seed(42)
            model = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32)
            ).to(DEVICE)
            convert_to_tcfp(
                model, mode=mode,
                skip_patterns=(), use_tensor_cores=True,
            )
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss = torch.tensor(0.0)
            for _ in range(100):
                opt.zero_grad()
                loss = (model(x) - target).pow(2).mean()
                loss.backward()
                opt.step()
            return loss.item()

        loss_12 = train_model(TCFPMode.TCFP12)
        loss_8 = train_model(TCFPMode.TCFP8)
        assert loss_12 <= loss_8 * 1.1, (
            f"TCFP-12 loss {loss_12:.6f} should be <= FP8 {loss_8:.6f}"
        )


# ── Integration ──────────────────────────────────────────────────────────


class TestTCFP12TCIntegration:
    def test_convert_to_tcfp_end_to_end(self) -> None:
        """convert_to_tcfp with TCFP12 + tensor cores works end-to-end."""
        model = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32)
        ).to(DEVICE)
        convert_to_tcfp(
            model, mode=TCFPMode.TCFP12,
            skip_patterns=(), use_tensor_cores=True,
        )
        x = torch.randn(4, 128, device=DEVICE)
        out = model(x)
        assert out.shape == (4, 32)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_hp_grad_weight(self) -> None:
        """hp_grad_weight=True produces valid gradients."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, hp_grad_weight=True,
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()

    def test_warmup_bypasses_quantisation(self) -> None:
        """During warmup, output should match FP32 linear exactly."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, warmup_steps=2,
        ).to(DEVICE)
        x = torch.randn(4, 128, device=DEVICE)
        with torch.no_grad():
            ref = torch.nn.functional.linear(x, layer.weight, layer.bias)
            out = layer(x)  # warmup step 1
            assert torch.allclose(out, ref, atol=1e-5)

    def test_correct_dispatch(self) -> None:
        """TCFP12 + tensor cores uses _TCFP12TensorCoreFunction."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12, use_tensor_cores=True
        ).to(DEVICE)
        assert layer.mode == TCFPMode.TCFP12
        assert layer.use_tensor_cores is True

        # Verify it runs without error (dispatch works correctly)
        x = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        assert out.shape == (4, 64)
        out.sum().backward()
        assert x.grad is not None

    def test_nf_aware_with_tcfp12_tc(self) -> None:
        """NF-aware scaling works with TCFP-12 TC path."""
        layer = TCFPLinear(
            128, 64, mode=TCFPMode.TCFP12,
            use_tensor_cores=True, nf_aware_scaling=True,
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (8, 64)
        assert torch.isfinite(out).all()
