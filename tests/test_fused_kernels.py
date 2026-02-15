"""Tests for fused Triton 2-GEMM kernels."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tcfp.core import TCFPMode, to_fp8_e4m3, to_fp8_e5m2

# Skip entire module if no CUDA or no Triton
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    ),
]

DEVICE = "cuda"

_HAS_TRITON = False
try:
    from tcfp.kernels import (
        fused_dual_gemm_backward_dx,
        fused_dual_gemm_forward,
        is_triton_available,
    )

    _HAS_TRITON = is_triton_available()
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not _HAS_TRITON, reason="Triton not available"
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _reference_dual_gemm_forward(
    act_fp8: torch.Tensor,
    w_hi_fp8: torch.Tensor,
    w_lo_fp8: torch.Tensor,
    act_inv: torch.Tensor,
    w_hi_inv: torch.Tensor,
    w_lo_inv: torch.Tensor,
) -> torch.Tensor:
    """Reference: two torch._scaled_mm calls + add."""
    hi = torch._scaled_mm(
        act_fp8,
        w_hi_fp8.t(),
        scale_a=act_inv,
        scale_b=w_hi_inv,
        out_dtype=torch.float32,
        use_fast_accum=False,
    )
    lo = torch._scaled_mm(
        act_fp8,
        w_lo_fp8.t(),
        scale_a=act_inv,
        scale_b=w_lo_inv,
        out_dtype=torch.float32,
        use_fast_accum=False,
    )
    return hi + lo


def _reference_dual_gemm_backward_dx(
    grad_fp8: torch.Tensor,
    w_hi_fp8: torch.Tensor,
    w_lo_fp8: torch.Tensor,
    grad_inv: torch.Tensor,
    w_hi_inv: torch.Tensor,
    w_lo_inv: torch.Tensor,
) -> torch.Tensor:
    """Reference: two torch._scaled_mm calls + add for dX."""
    from tcfp.core import ensure_column_major

    hi = torch._scaled_mm(
        grad_fp8,
        ensure_column_major(w_hi_fp8),
        scale_a=grad_inv,
        scale_b=w_hi_inv,
        out_dtype=torch.float32,
        use_fast_accum=False,
    )
    lo = torch._scaled_mm(
        grad_fp8,
        ensure_column_major(w_lo_fp8),
        scale_a=grad_inv,
        scale_b=w_lo_inv,
        out_dtype=torch.float32,
        use_fast_accum=False,
    )
    return hi + lo


def _make_fp8_pair(
    shape: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create FP8 hi/lo weight decomposition from random data."""
    w = torch.randn(shape, device=DEVICE)
    w_hi_fp8, w_hi_inv = to_fp8_e4m3(w)
    residual = w - w_hi_fp8.float() * w_hi_inv
    w_lo_fp8, w_lo_inv = to_fp8_e4m3(residual)
    return w_hi_fp8, w_hi_inv, w_lo_fp8, w_lo_inv


# ── Forward Correctness ─────────────────────────────────────────────────


@requires_triton
class TestFusedForwardCorrectness:
    """Fused forward kernel matches two-_scaled_mm reference."""

    @pytest.mark.parametrize(
        "M, K, N",
        [(64, 128, 64), (256, 512, 256), (128, 256, 512)],
        ids=["small", "medium", "non-square"],
    )
    def test_matches_reference(self, M: int, K: int, N: int) -> None:
        torch.manual_seed(42)
        act = torch.randn(M, K, device=DEVICE)
        act_fp8, act_inv = to_fp8_e4m3(act)
        w_hi_fp8, w_hi_inv, w_lo_fp8, w_lo_inv = _make_fp8_pair((N, K))

        fused = fused_dual_gemm_forward(
            act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv
        )
        ref = _reference_dual_gemm_forward(
            act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv
        )

        assert fused.shape == ref.shape == (M, N)
        torch.testing.assert_close(fused, ref, atol=1e-2, rtol=1e-2)

    def test_large_matrices(self) -> None:
        """Fused kernel works at datacenter-scale dimensions."""
        torch.manual_seed(42)
        M, K, N = 1024, 1024, 1024
        act = torch.randn(M, K, device=DEVICE)
        act_fp8, act_inv = to_fp8_e4m3(act)
        w_hi_fp8, w_hi_inv, w_lo_fp8, w_lo_inv = _make_fp8_pair((N, K))

        fused = fused_dual_gemm_forward(
            act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv
        )
        ref = _reference_dual_gemm_forward(
            act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv
        )
        torch.testing.assert_close(fused, ref, atol=1e-2, rtol=1e-2)

    def test_zero_lo_matches_single_gemm(self) -> None:
        """When W_lo is zero, output = single GEMM from W_hi."""
        torch.manual_seed(42)
        M, K, N = 64, 128, 64
        act = torch.randn(M, K, device=DEVICE)
        act_fp8, act_inv = to_fp8_e4m3(act)
        w = torch.randn(N, K, device=DEVICE)
        w_hi_fp8, w_hi_inv = to_fp8_e4m3(w)

        # Zero residual
        w_lo_fp8 = torch.zeros(N, K, device=DEVICE).to(torch.float8_e4m3fn)
        w_lo_inv = torch.ones(1, device=DEVICE, dtype=torch.float32)

        fused = fused_dual_gemm_forward(
            act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv
        )
        single = torch._scaled_mm(
            act_fp8,
            w_hi_fp8.t(),
            scale_a=act_inv,
            scale_b=w_hi_inv,
            out_dtype=torch.float32,
            use_fast_accum=False,
        )
        torch.testing.assert_close(fused, single, atol=1e-4, rtol=1e-4)

    def test_scales_applied_correctly(self) -> None:
        """Verify scales are correctly multiplied into the output."""
        torch.manual_seed(42)
        M, K, N = 64, 64, 64
        act = torch.randn(M, K, device=DEVICE)
        act_fp8, act_inv = to_fp8_e4m3(act)
        w_hi_fp8, w_hi_inv, w_lo_fp8, w_lo_inv = _make_fp8_pair((N, K))

        out1 = fused_dual_gemm_forward(
            act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv
        )
        # Double the hi scale — output should change proportionally
        doubled_inv = w_hi_inv * 2.0
        out2 = fused_dual_gemm_forward(
            act_fp8, w_hi_fp8, w_lo_fp8, act_inv, doubled_inv, w_lo_inv
        )
        # The hi contribution should double
        hi_only = fused_dual_gemm_forward(
            act_fp8,
            w_hi_fp8,
            w_lo_fp8,
            act_inv,
            w_hi_inv,
            torch.zeros(1, device=DEVICE, dtype=torch.float32),
        )
        diff = out2 - out1
        torch.testing.assert_close(diff, hi_only, atol=1e-2, rtol=1e-2)


# ── Backward dX Correctness ─────────────────────────────────────────────


@requires_triton
class TestFusedBackwardDxCorrectness:
    """Fused backward dX kernel matches reference."""

    @pytest.mark.parametrize(
        "M, D_out, D_in",
        [(64, 128, 64), (256, 512, 256)],
        ids=["small", "medium"],
    )
    def test_matches_reference(
        self, M: int, D_out: int, D_in: int
    ) -> None:
        torch.manual_seed(42)
        grad = torch.randn(M, D_out, device=DEVICE)
        grad_fp8, grad_inv = to_fp8_e5m2(grad)
        w_hi_fp8, w_hi_inv, w_lo_fp8, w_lo_inv = _make_fp8_pair(
            (D_out, D_in)
        )

        fused = fused_dual_gemm_backward_dx(
            grad_fp8, w_hi_fp8, w_lo_fp8, grad_inv, w_hi_inv, w_lo_inv
        )
        ref = _reference_dual_gemm_backward_dx(
            grad_fp8, w_hi_fp8, w_lo_fp8, grad_inv, w_hi_inv, w_lo_inv
        )

        assert fused.shape == ref.shape == (M, D_in)
        torch.testing.assert_close(fused, ref, atol=1e-2, rtol=1e-2)

    def test_output_shape(self) -> None:
        """Output has correct (M, D_in) shape."""
        M, D_out, D_in = 128, 256, 512
        grad = torch.randn(M, D_out, device=DEVICE)
        grad_fp8, grad_inv = to_fp8_e5m2(grad)
        w_hi_fp8, w_hi_inv, w_lo_fp8, w_lo_inv = _make_fp8_pair(
            (D_out, D_in)
        )

        result = fused_dual_gemm_backward_dx(
            grad_fp8, w_hi_fp8, w_lo_fp8, grad_inv, w_hi_inv, w_lo_inv
        )
        assert result.shape == (M, D_in)
        assert result.dtype == torch.float32


# ── Integration Tests ────────────────────────────────────────────────────


@requires_triton
class TestFusedKernelIntegration:
    """End-to-end tests with TCFPLinear."""

    def test_tcfp12_linear_fwd_bwd(self) -> None:
        """TCFPLinear with fused kernel runs forward + backward."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128,
            64,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            use_fused_kernel=True,
        ).to(DEVICE)

        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        assert out.shape == (8, 64)

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (8, 128)
        assert torch.isfinite(x.grad).all()

    def test_fwd_bwd_3d_input(self) -> None:
        """3D input (batch, seq, dim) works with fused kernels."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            256,
            128,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            use_fused_kernel=True,
        ).to(DEVICE)

        x = torch.randn(4, 16, 256, device=DEVICE, requires_grad=True)
        out = layer(x)
        assert out.shape == (4, 16, 128)

        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 16, 256)

    def test_error_feedback_works(self) -> None:
        """Error feedback state works with fused kernels."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128,
            64,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            error_feedback=True,
            use_fused_kernel=True,
        ).to(DEVICE)

        x = torch.randn(8, 128, device=DEVICE)
        # Run two forward passes — error feedback should accumulate
        out1 = layer(x)
        out1.sum().backward()
        out2 = layer(x)
        out2.sum().backward()
        # Error state should be populated
        assert layer._error_state is not None
        assert len(layer._error_state._buffers) > 0

    def test_delayed_scaling_works(self) -> None:
        """Delayed scaling + fused kernel produces valid results."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128,
            64,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            delayed_scaling=True,
            use_fused_kernel=True,
        ).to(DEVICE)

        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        assert torch.isfinite(out).all()
        loss = out.sum()
        loss.backward()
        assert torch.isfinite(x.grad).all()  # type: ignore[union-attr]

    def test_convert_to_tcfp_passes_fused(self) -> None:
        """convert_to_tcfp forwards use_fused_kernel parameter."""
        from tcfp.nn import TCFPLinear, convert_to_tcfp

        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        ).to(DEVICE)
        convert_to_tcfp(
            model,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            use_fused_kernel=True,
        )
        for m in model.modules():
            if isinstance(m, TCFPLinear):
                assert m.use_fused_kernel == _HAS_TRITON

    def test_training_convergence(self) -> None:
        """Loss decreases over training with fused kernels."""
        from tcfp.nn import convert_to_tcfp

        torch.manual_seed(42)
        # Fixed data for deterministic convergence
        x = torch.randn(32, 128, device=DEVICE)
        target = torch.randn(32, 32, device=DEVICE)

        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        ).to(DEVICE)
        convert_to_tcfp(
            model,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            use_fused_kernel=True,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses = []
        for _ in range(50):
            out = model(x)
            loss = nn.functional.mse_loss(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # Compare avg of first 5 vs last 5 for robustness
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        import math

        assert all(math.isfinite(v) for v in losses), "NaN or Inf"
        assert late < early, (
            f"Loss did not decrease: {early:.4f} -> {late:.4f}"
        )


# ── Fallback Tests ───────────────────────────────────────────────────────


class TestFallback:
    """Tests that the system works without fused kernels."""

    def test_tcfp12_without_triton(self) -> None:
        """TCFPLinear works when use_fused_kernel=False."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128,
            64,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            use_fused_kernel=False,
        ).to(DEVICE)

        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        assert out.shape == (8, 64)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_extra_repr_shows_fused(self) -> None:
        """extra_repr includes use_fused_kernel when active."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128,
            64,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            use_fused_kernel=True,
        )
        r = layer.extra_repr()
        if _HAS_TRITON:
            assert "use_fused_kernel=True" in r
        else:
            assert "use_fused_kernel" not in r


# ── Edge Cases ───────────────────────────────────────────────────────────


@requires_triton
class TestFusedKernelEdgeCases:
    """Edge cases for dimension alignment and scale values."""

    def test_minimum_dimensions(self) -> None:
        """Works with minimum 16x16 matrices."""
        M, K, N = 16, 16, 16
        act = torch.randn(M, K, device=DEVICE)
        act_fp8, act_inv = to_fp8_e4m3(act)
        w_hi_fp8, w_hi_inv, w_lo_fp8, w_lo_inv = _make_fp8_pair((N, K))

        fused = fused_dual_gemm_forward(
            act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv
        )
        assert fused.shape == (M, N)
        assert torch.isfinite(fused).all()

    def test_non_power_of_2_dims(self) -> None:
        """Works with dimensions not powers of 2 (but divisible by 16)."""
        M, K, N = 48, 80, 112
        act = torch.randn(M, K, device=DEVICE)
        act_fp8, act_inv = to_fp8_e4m3(act)
        w_hi_fp8, w_hi_inv, w_lo_fp8, w_lo_inv = _make_fp8_pair((N, K))

        fused = fused_dual_gemm_forward(
            act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv
        )
        ref = _reference_dual_gemm_forward(
            act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv
        )
        torch.testing.assert_close(fused, ref, atol=1e-2, rtol=1e-2)

    def test_all_outputs_finite(self) -> None:
        """No NaN or Inf in output for typical inputs."""
        torch.manual_seed(123)
        M, K, N = 256, 256, 256
        act = torch.randn(M, K, device=DEVICE)
        act_fp8, act_inv = to_fp8_e4m3(act)
        w_hi_fp8, w_hi_inv, w_lo_fp8, w_lo_inv = _make_fp8_pair((N, K))

        fused = fused_dual_gemm_forward(
            act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv
        )
        assert torch.isfinite(fused).all()
