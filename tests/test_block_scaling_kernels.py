"""Tests for per-block FP8 scaling Triton kernels."""
from __future__ import annotations

import math
import warnings

import pytest
import torch
import torch.nn as nn

from tcfp.core import (
    FP8_E4M3_MAX,
    TCFPMode,
)

# Skip entire module if no CUDA
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    ),
]

DEVICE = "cuda"

_HAS_TRITON = False
try:
    from tcfp.kernels import (
        block_dequantize_fp8,
        block_quantize_fp8,
        block_scaled_gemm,
        fused_block_scaled_backward_dx,
        fused_block_scaled_dual_gemm_backward_dx,
        fused_block_scaled_dual_gemm_forward,
        fused_wtquant_block_scaled_dual_gemm_forward,
        is_triton_available,
    )

    _HAS_TRITON = is_triton_available()
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not _HAS_TRITON, reason="Triton not available"
)


# ── Reference implementations ───────────────────────────────────────────


def _reference_block_scaled_gemm(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Reference: dequantize per-block then FP32 matmul."""
    a_deq = block_dequantize_fp8(a_fp8, sa, block_size)  # type: ignore[reportPossiblyUnboundVariable]
    b_deq = block_dequantize_fp8(b_fp8, sb, block_size)  # type: ignore[reportPossiblyUnboundVariable]
    return a_deq @ b_deq.t()


def _make_block_quantized(
    M: int, K: int, block_size: int, device: str = DEVICE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a random FP8 tensor with per-block scales."""
    x = torch.randn(M, K, device=device)
    fp8, scales = block_quantize_fp8(x, block_size=block_size)  # type: ignore[reportPossiblyUnboundVariable]
    return fp8, scales


# ── Test: Block Quantization ────────────────────────────────────────────


class TestBlockQuantize:
    @requires_triton
    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_output_shapes(self, block_size: int) -> None:
        M, K = 128, 256
        x = torch.randn(M, K, device=DEVICE)
        fp8, scales = block_quantize_fp8(x, block_size=block_size)  # type: ignore[reportPossiblyUnboundVariable]

        assert fp8.shape == (M, K)
        assert scales.shape == (K // block_size, M)

    @requires_triton
    def test_output_dtypes(self) -> None:
        x = torch.randn(64, 128, device=DEVICE)
        fp8, scales = block_quantize_fp8(x, block_size=32)  # type: ignore[reportPossiblyUnboundVariable]
        assert fp8.dtype == torch.float8_e4m3fn
        assert scales.dtype == torch.int8

    @requires_triton
    def test_e5m2_dtype(self) -> None:
        x = torch.randn(64, 128, device=DEVICE)
        fp8, scales = block_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            x, block_size=32, fp8_dtype=torch.float8_e5m2,
        )
        assert fp8.dtype == torch.float8_e5m2

    @requires_triton
    def test_scales_are_int8_exponents(self) -> None:
        x = torch.randn(64, 128, device=DEVICE) * 10
        _, scales = block_quantize_fp8(x, block_size=32)  # type: ignore[reportPossiblyUnboundVariable]
        assert scales.dtype == torch.int8
        # Exponents clamped to >= -126 in kernel
        assert (scales >= -126).all()

    @requires_triton
    def test_roundtrip_quality(self) -> None:
        """Dequantized values should be close to original."""
        x = torch.randn(128, 256, device=DEVICE)
        fp8, scales = block_quantize_fp8(x, block_size=32)  # type: ignore[reportPossiblyUnboundVariable]
        x_recon = block_dequantize_fp8(fp8, scales, 32)  # type: ignore[reportPossiblyUnboundVariable]
        # Relative MSE should be small
        mse = (x - x_recon).pow(2).mean()
        signal = x.pow(2).mean()
        assert mse / signal < 0.01  # < 1% relative MSE

    @requires_triton
    def test_values_in_fp8_range(self) -> None:
        x = torch.randn(64, 128, device=DEVICE) * 100
        fp8, _ = block_quantize_fp8(x, block_size=32)  # type: ignore[reportPossiblyUnboundVariable]
        vals = fp8.float()
        assert vals.abs().max() <= FP8_E4M3_MAX

    @requires_triton
    def test_zero_input(self) -> None:
        x = torch.zeros(64, 128, device=DEVICE)
        fp8, scales = block_quantize_fp8(x, block_size=32)  # type: ignore[reportPossiblyUnboundVariable]
        assert torch.isfinite(fp8.float()).all()
        assert torch.isfinite(scales).all()

    @requires_triton
    def test_large_values(self) -> None:
        x = torch.randn(64, 128, device=DEVICE) * 1000
        fp8, scales = block_quantize_fp8(x, block_size=64)  # type: ignore[reportPossiblyUnboundVariable]
        assert torch.isfinite(fp8.float()).all()
        assert torch.isfinite(scales).all()


# ── Test: Block-Scaled GEMM ─────────────────────────────────────────────


class TestBlockScaledGemm:
    @requires_triton
    @pytest.mark.parametrize(
        "M,K,N,block_size",
        [
            (64, 64, 64, 32),
            (128, 128, 128, 64),
            (64, 128, 64, 128),
            (128, 256, 64, 32),
        ],
    )
    def test_matches_reference(
        self, M: int, K: int, N: int, block_size: int
    ) -> None:
        a_fp8, sa = _make_block_quantized(M, K, block_size)
        b_fp8, sb = _make_block_quantized(N, K, block_size)

        result = block_scaled_gemm(  # type: ignore[reportPossiblyUnboundVariable]
            a_fp8, b_fp8, sa, sb, block_size=block_size,
        )
        ref = _reference_block_scaled_gemm(a_fp8, b_fp8, sa, sb, block_size)

        assert torch.allclose(result, ref, atol=1e-3, rtol=1e-3), (
            f"Max diff: {(result - ref).abs().max().item():.6f}"
        )

    @requires_triton
    def test_output_shape_and_dtype(self) -> None:
        a_fp8, sa = _make_block_quantized(64, 128, 32)
        b_fp8, sb = _make_block_quantized(96, 128, 32)

        result = block_scaled_gemm(a_fp8, b_fp8, sa, sb, block_size=32)  # type: ignore[reportPossiblyUnboundVariable]
        assert result.shape == (64, 96)
        assert result.dtype == torch.float32

    @requires_triton
    def test_all_finite(self) -> None:
        a_fp8, sa = _make_block_quantized(128, 128, 64)
        b_fp8, sb = _make_block_quantized(128, 128, 64)

        result = block_scaled_gemm(a_fp8, b_fp8, sa, sb, block_size=64)  # type: ignore[reportPossiblyUnboundVariable]
        assert torch.isfinite(result).all()

    @requires_triton
    def test_identity_scales(self) -> None:
        """When all exponents are 0 (scale=1.0), should match per-tensor FP8."""
        M, K, N = 64, 64, 64
        a = torch.randn(M, K, device=DEVICE)
        b = torch.randn(N, K, device=DEVICE)
        a_fp8 = a.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        b_fp8 = b.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

        # int8 exponent 0 → exp2(0) = 1.0 (identity scale)
        zero_a = torch.zeros(K // 32, M, device=DEVICE, dtype=torch.int8)
        zero_b = torch.zeros(K // 32, N, device=DEVICE, dtype=torch.int8)

        result = block_scaled_gemm(a_fp8, b_fp8, zero_a, zero_b, block_size=32)  # type: ignore[reportPossiblyUnboundVariable]
        ref = a_fp8.float() @ b_fp8.float().t()

        assert torch.allclose(result, ref, atol=1e-3)


# ── Test: Fused Block-Scaled Dual GEMM ──────────────────────────────────


class TestFusedBlockScaledDualGemm:
    @requires_triton
    @pytest.mark.parametrize(
        "M,K,N,block_size",
        [
            (64, 64, 64, 32),
            (128, 128, 64, 64),
            (64, 128, 128, 128),
        ],
    )
    def test_matches_two_separate_gemms(
        self, M: int, K: int, N: int, block_size: int
    ) -> None:
        a_fp8, sa = _make_block_quantized(M, K, block_size)
        bhi_fp8, sbhi = _make_block_quantized(N, K, block_size)
        blo_fp8, sblo = _make_block_quantized(N, K, block_size)

        result = fused_block_scaled_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            a_fp8, bhi_fp8, blo_fp8, sa, sbhi, sblo, block_size=block_size,
        )
        ref_hi = _reference_block_scaled_gemm(
            a_fp8, bhi_fp8, sa, sbhi, block_size,
        )
        ref_lo = _reference_block_scaled_gemm(
            a_fp8, blo_fp8, sa, sblo, block_size,
        )
        ref = ref_hi + ref_lo

        assert torch.allclose(result, ref, atol=1e-3, rtol=1e-3), (
            f"Max diff: {(result - ref).abs().max().item():.6f}"
        )

    @requires_triton
    def test_backward_dx(self) -> None:
        M, D_out, D_in = 64, 128, 64
        bs = 32

        grad_fp8, gs = _make_block_quantized(M, D_out, bs)
        whi_fp8, whi_s = _make_block_quantized(D_out, D_in, bs)
        wlo_fp8, wlo_s = _make_block_quantized(D_out, D_in, bs)

        # Convert to E5M2 for grad (but keep as E4M3 for testing)
        result = fused_block_scaled_dual_gemm_backward_dx(  # type: ignore[reportPossiblyUnboundVariable]
            grad_fp8, whi_fp8, wlo_fp8, gs, whi_s, wlo_s, block_size=bs,
        )
        assert result.shape == (M, D_in)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    @requires_triton
    def test_zero_lo_matches_single_gemm(self) -> None:
        M, K, N = 64, 128, 64
        bs = 32

        a_fp8, sa = _make_block_quantized(M, K, bs)
        bhi_fp8, sbhi = _make_block_quantized(N, K, bs)
        # Zero W_lo
        blo_fp8 = torch.zeros(N, K, device=DEVICE, dtype=torch.float8_e4m3fn)
        sblo = torch.zeros(K // bs, N, device=DEVICE)

        result = fused_block_scaled_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            a_fp8, bhi_fp8, blo_fp8, sa, sbhi, sblo, block_size=bs,
        )
        ref = _reference_block_scaled_gemm(a_fp8, bhi_fp8, sa, sbhi, bs)

        assert torch.allclose(result, ref, atol=1e-3)

    @requires_triton
    def test_large_matrices(self) -> None:
        M, K, N = 256, 512, 256
        bs = 128

        a_fp8, sa = _make_block_quantized(M, K, bs)
        bhi_fp8, sbhi = _make_block_quantized(N, K, bs)
        blo_fp8, sblo = _make_block_quantized(N, K, bs)

        result = fused_block_scaled_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            a_fp8, bhi_fp8, blo_fp8, sa, sbhi, sblo, block_size=bs,
        )
        assert result.shape == (M, N)
        assert torch.isfinite(result).all()


# ── Test: Fused block-scaled backward dX kernel ──────────────────────────


def _reference_backward_dx(
    grad_fp8: torch.Tensor,
    w_hi_fp8: torch.Tensor,
    w_lo_fp8: torch.Tensor,
    grad_scales: torch.Tensor,
    w_hi_scales: torch.Tensor,
    w_lo_scales: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Reference: dequant all to FP32, then two separate matmuls."""
    w_hi_deq = block_dequantize_fp8(w_hi_fp8, w_hi_scales, block_size)  # type: ignore
    w_lo_deq = block_dequantize_fp8(w_lo_fp8, w_lo_scales, block_size)  # type: ignore
    grad_deq = block_dequantize_fp8(grad_fp8, grad_scales, block_size)  # type: ignore
    return grad_deq @ w_hi_deq + grad_deq @ w_lo_deq


class TestFusedBackwardDX:
    """Tests for fused_block_scaled_backward_dx Triton kernel."""

    @requires_triton
    @pytest.mark.parametrize(
        "M,D_out,D_in,bs",
        [
            (64, 128, 64, 64),
            (128, 256, 128, 128),
            (64, 256, 128, 64),
            (64, 128, 64, 32),
        ],
    )
    def test_matches_reference(self, M: int, D_out: int, D_in: int, bs: int) -> None:
        """Fused backward dX must numerically match the reference two-matmul path."""
        grad_fp8, grad_s = _make_block_quantized(M, D_out, bs)
        # W has shape (D_out, D_in); its scales are (D_in//bs, D_out)
        w_hi = torch.randn(D_out, D_in, device=DEVICE)
        w_lo = torch.randn(D_out, D_in, device=DEVICE)
        from tcfp.kernels import block_quantize_fp8  # type: ignore
        w_hi_fp8, w_hi_s = block_quantize_fp8(w_hi, block_size=bs)  # type: ignore
        w_lo_fp8, w_lo_s = block_quantize_fp8(w_lo, block_size=bs)  # type: ignore

        ref = _reference_backward_dx(
            grad_fp8, w_hi_fp8, w_lo_fp8, grad_s, w_hi_s, w_lo_s, bs,
        )
        result = fused_block_scaled_backward_dx(  # type: ignore
            grad_fp8, w_hi_fp8, w_lo_fp8, grad_s, w_hi_s, w_lo_s, block_size=bs,
        )

        assert result.shape == (M, D_in)
        assert result.dtype == torch.float32
        # tl.dot with float32 inputs uses TF32 on modern GPUs (10 mantissa bits
        # vs 23 for pure float32), causing ~1e-2 absolute differences vs the
        # pure FP32 reference. This is acceptable for gradient computation.
        assert torch.allclose(result, ref, atol=1e-2, rtol=1e-3), (
            f"Max diff: {(result - ref).abs().max().item():.2e}"
        )

    @requires_triton
    def test_output_shape_dtype(self) -> None:
        M, D_out, D_in, bs = 128, 256, 128, 64
        grad_fp8, grad_s = _make_block_quantized(M, D_out, bs)
        w_hi_fp8, w_hi_s = _make_block_quantized(D_out, D_in, bs)
        w_lo_fp8, w_lo_s = _make_block_quantized(D_out, D_in, bs)

        result = fused_block_scaled_backward_dx(  # type: ignore
            grad_fp8, w_hi_fp8, w_lo_fp8, grad_s, w_hi_s, w_lo_s, block_size=bs,
        )
        assert result.shape == (M, D_in)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()

    @requires_triton
    def test_zero_lo_matches_hi_only(self) -> None:
        """When W_lo is zero (sentinel scales), result = grad @ W_hi only."""
        M, D_out, D_in, bs = 64, 128, 64, 64
        grad_fp8, grad_s = _make_block_quantized(M, D_out, bs)
        w_hi_fp8, w_hi_s = _make_block_quantized(D_out, D_in, bs)
        # Zero W_lo: FP8 zeros + sentinel scale int8(-128)
        w_lo_fp8 = torch.zeros(D_out, D_in, device=DEVICE, dtype=torch.float8_e4m3fn)
        # Sentinel -128 → exp2(-128)≈0, but with sentinel restore → 0.0
        w_lo_s = torch.full((D_in // bs, D_out), -128, device=DEVICE, dtype=torch.int8)

        result = fused_block_scaled_backward_dx(  # type: ignore
            grad_fp8, w_hi_fp8, w_lo_fp8, grad_s, w_hi_s, w_lo_s, block_size=bs,
        )
        # Reference: grad @ W_hi only (W_lo zeroed by sentinel)
        w_hi_deq = block_dequantize_fp8(w_hi_fp8, w_hi_s, bs)  # type: ignore
        grad_deq = block_dequantize_fp8(grad_fp8, grad_s, bs)  # type: ignore
        ref_hi_only = grad_deq @ w_hi_deq

        # tl.dot uses TF32 on modern GPUs (~1e-2 abs error vs pure FP32 ref).
        assert torch.allclose(result, ref_hi_only, atol=1e-2), (
            f"Max diff: {(result - ref_hi_only).abs().max().item():.2e}"
        )

    @requires_triton
    def test_dispatch_via_legacy_fn(self) -> None:
        """fused_block_scaled_dual_gemm_backward_dx dispatches to Triton kernel."""
        M, D_out, D_in, bs = 64, 128, 64, 64
        grad_fp8, grad_s = _make_block_quantized(M, D_out, bs)
        w_hi_fp8, w_hi_s = _make_block_quantized(D_out, D_in, bs)
        w_lo_fp8, w_lo_s = _make_block_quantized(D_out, D_in, bs)

        ref = _reference_backward_dx(
            grad_fp8, w_hi_fp8, w_lo_fp8, grad_s, w_hi_s, w_lo_s, bs,
        )
        result = fused_block_scaled_dual_gemm_backward_dx(  # type: ignore
            grad_fp8, w_hi_fp8, w_lo_fp8, grad_s, w_hi_s, w_lo_s, block_size=bs,
        )
        # tl.dot uses TF32 on modern GPUs (~1e-2 abs error vs pure FP32 ref).
        assert torch.allclose(result, ref, atol=1e-2, rtol=1e-3)


# ── Test: Integration with TCFPLinear ───────────────────────────────────


class TestBlockScaledIntegration:
    @requires_triton
    def test_tcfp12_linear_fwd_bwd(self) -> None:
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128, 64, bias=True,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=32,
        ).cuda()

        x = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        y = layer(x)
        assert y.shape == (4, 64)

        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 128)

    @requires_triton
    def test_3d_input(self) -> None:
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128, 64, bias=False,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=64,
        ).cuda()

        x = torch.randn(2, 8, 128, device=DEVICE, requires_grad=True)
        y = layer(x)
        assert y.shape == (2, 8, 64)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 8, 128)

    @requires_triton
    def test_error_feedback_works(self) -> None:
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128, 64,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            error_feedback=True,
            scale_block_size=32,
        ).cuda()

        x = torch.randn(4, 128, device=DEVICE)
        # Run forward twice to populate error feedback
        _ = layer(x)
        _ = layer(x)

        assert layer._error_state is not None
        assert len(layer._error_state._buffers) > 0

    @requires_triton
    def test_convert_to_tcfp_passes_param(self) -> None:
        from tcfp.nn import TCFPLinear, convert_to_tcfp

        model = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, 32))
        convert_to_tcfp(
            model,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=32,
        )
        for m in model.modules():
            if isinstance(m, TCFPLinear):
                assert m.scale_block_size == 32

    @requires_triton
    def test_training_convergence(self) -> None:
        from tcfp.nn import TCFPLinear

        torch.manual_seed(42)
        layer = TCFPLinear(
            128, 64, bias=True,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=64,
        ).cuda()
        opt = torch.optim.SGD(layer.parameters(), lr=0.01)

        # Fixed data
        x = torch.randn(16, 128, device=DEVICE)
        target = torch.randn(16, 64, device=DEVICE)

        losses: list[float] = []
        for _ in range(50):
            y = layer(x)
            loss = (y - target).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        avg_first5 = sum(losses[:5]) / 5
        avg_last5 = sum(losses[-5:]) / 5
        assert avg_last5 < avg_first5, (
            f"Loss didn't decrease: first5={avg_first5:.4f}, last5={avg_last5:.4f}"
        )

    @requires_triton
    def test_no_nan_during_training(self) -> None:
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128, 128,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=128,
        ).cuda()
        opt = torch.optim.Adam(layer.parameters(), lr=1e-3)

        for _ in range(20):
            x = torch.randn(8, 128, device=DEVICE)
            y = layer(x)
            loss = y.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            v = loss.item()
            assert math.isfinite(v), f"NaN/Inf loss: {v}"

    @requires_triton
    def test_gradient_shapes(self) -> None:
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128, 64, bias=True,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=32,
        ).cuda()

        x = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        y = layer(x)
        y.sum().backward()

        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (64, 128)
        assert layer.bias is not None
        assert layer.bias.grad is not None
        assert layer.bias.grad.shape == (64,)

    def test_delayed_scaling_conflict_warns(self) -> None:
        from tcfp.nn import TCFPLinear

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layer = TCFPLinear(
                128, 64,
                mode=TCFPMode.TCFP12,
                use_tensor_cores=True,
                delayed_scaling=True,
                scale_block_size=32,
            )
            # Should warn about mutual exclusivity
            warning_msgs = [str(x.message) for x in w]
            assert any("mutually exclusive" in msg for msg in warning_msgs)
            assert not layer.delayed_scaling

    def test_fallback_when_no_triton(self) -> None:
        from tcfp.nn import TCFPLinear

        # When _HAS_FUSED_KERNELS is False, scale_block_size should
        # fall back to None. We test the repr to check.
        layer = TCFPLinear(
            128, 64,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=32,
        )
        repr_str = layer.extra_repr()
        if not _HAS_TRITON:
            assert "scale_block_size" not in repr_str

    def test_invalid_block_size_raises(self) -> None:
        from tcfp.nn import TCFPLinear

        with pytest.raises(ValueError, match="32, 64, or 128"):
            TCFPLinear(
                128, 64,
                mode=TCFPMode.TCFP12,
                use_tensor_cores=True,
                scale_block_size=16,
            )

    def test_indivisible_in_features_raises(self) -> None:
        from tcfp.nn import TCFPLinear

        with pytest.raises(ValueError, match="in_features.*divisible"):
            TCFPLinear(
                100, 128,
                mode=TCFPMode.TCFP12,
                use_tensor_cores=True,
                scale_block_size=128,
            )

    def test_indivisible_out_features_raises(self) -> None:
        from tcfp.nn import TCFPLinear

        with pytest.raises(ValueError, match="out_features.*divisible"):
            TCFPLinear(
                128, 100,
                mode=TCFPMode.TCFP12,
                use_tensor_cores=True,
                scale_block_size=128,
            )

    @requires_triton
    def test_extra_repr_shows_block_size(self) -> None:
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128, 64,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=64,
        )
        assert "scale_block_size=64" in layer.extra_repr()


# ── Sparsity Fast-Path Tests ─────────────────────────────────────────────


class TestSparsityFastPath:
    """Tests for the W_lo sparsity skip in fused dual GEMM kernels."""

    @requires_triton
    def test_zero_lo_sparse_skip(self) -> None:
        """All-zero W_lo (exp=-128 sentinel) matches W_hi-only reference."""
        M, K, N = 64, 128, 64
        bs = 32

        a_fp8, sa = _make_block_quantized(M, K, bs)
        bhi_fp8, sbhi = _make_block_quantized(N, K, bs)
        blo_fp8 = torch.zeros(N, K, device=DEVICE, dtype=torch.float8_e4m3fn)
        # int8 -128 sentinel: exp2(-128) → 0.0 in GEMM kernel
        sblo = torch.full(
            (K // bs, N), -128, device=DEVICE, dtype=torch.int8,
        )

        result = fused_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            a_fp8, bhi_fp8, blo_fp8, sa, sbhi, sblo,
            block_size=bs, sparse_threshold=0.0,
        )
        ref = _reference_block_scaled_gemm(a_fp8, bhi_fp8, sa, sbhi, bs)
        assert torch.allclose(result, ref, atol=1e-3)

    @requires_triton
    def test_partial_sparse_correct(self) -> None:
        """Mixed sparse/dense K-blocks produce correct output."""
        M, K, N = 64, 256, 64
        bs = 64
        num_kb = K // bs

        a_fp8, sa = _make_block_quantized(M, K, bs)
        bhi_fp8, sbhi = _make_block_quantized(N, K, bs)
        blo_fp8, sblo = _make_block_quantized(N, K, bs)

        # Zero out odd K-blocks in W_lo (set exp=-128 sentinel)
        for kb in range(num_kb):
            if kb % 2 == 1:
                sblo[kb, :] = -128

        result_sparse = fused_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            a_fp8, bhi_fp8, blo_fp8, sa, sbhi, sblo,
            block_size=bs, sparse_threshold=0.0,
        )

        # Reference: manual dequant with the same zero-scale pattern
        num_blocks = K // bs
        a_blocked = a_fp8.float().reshape(M, num_blocks, bs)
        a_deq = (a_blocked * torch.exp2(sa.float()).t().unsqueeze(-1)).reshape(M, K)
        bhi_blocked = bhi_fp8.float().reshape(N, num_blocks, bs)
        bhi_deq = (bhi_blocked * torch.exp2(sbhi.float()).t().unsqueeze(-1)).reshape(N, K)
        blo_blocked = blo_fp8.float().reshape(N, num_blocks, bs)
        blo_deq = (blo_blocked * torch.exp2(sblo.float()).t().unsqueeze(-1)).reshape(N, K)
        ref = a_deq @ (bhi_deq + blo_deq).t()

        assert torch.allclose(result_sparse, ref, atol=1e-2)

    @requires_triton
    def test_sparse_threshold_zero_no_regression(self) -> None:
        """SPARSE_THRESHOLD=0.0 with normal scales produces same output."""
        M, K, N = 128, 128, 64
        bs = 32

        a_fp8, sa = _make_block_quantized(M, K, bs)
        bhi_fp8, sbhi = _make_block_quantized(N, K, bs)
        blo_fp8, sblo = _make_block_quantized(N, K, bs)

        # All scales are > 0 (from _make_block_quantized), so sparsity
        # path should never trigger. Output must be identical.
        result = fused_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            a_fp8, bhi_fp8, blo_fp8, sa, sbhi, sblo,
            block_size=bs, sparse_threshold=0.0,
        )

        # Reference with full dequant
        num_blocks = K // bs
        a_blocked = a_fp8.float().reshape(M, num_blocks, bs)
        a_deq = (a_blocked * torch.exp2(sa.float()).t().unsqueeze(-1)).reshape(M, K)
        bhi_blocked = bhi_fp8.float().reshape(N, num_blocks, bs)
        bhi_deq = (bhi_blocked * torch.exp2(sbhi.float()).t().unsqueeze(-1)).reshape(N, K)
        blo_blocked = blo_fp8.float().reshape(N, num_blocks, bs)
        blo_deq = (blo_blocked * torch.exp2(sblo.float()).t().unsqueeze(-1)).reshape(N, K)
        ref = a_deq @ (bhi_deq + blo_deq).t()

        assert torch.allclose(result, ref, atol=1e-2)


# ── Test: Persistent Tiling ─────────────────────────────────────────────


class TestPersistentKernel:
    """Verify persistent tiling produces correct results."""

    @requires_triton
    def test_persistent_matches_reference(self) -> None:
        """Large M*N: persistent output matches reference dequant-matmul."""
        M, K, N = 512, 512, 512
        bs = 128

        a_fp8, sa = _make_block_quantized(M, K, bs)
        bhi_fp8, sbhi = _make_block_quantized(N, K, bs)
        blo_fp8, sblo = _make_block_quantized(N, K, bs)

        result = fused_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            a_fp8, bhi_fp8, blo_fp8, sa, sbhi, sblo,
            block_size=bs, persistent=True,
        )

        # Reference with full dequant
        num_blocks = K // bs
        a_blk = a_fp8.float().reshape(M, num_blocks, bs)
        a_deq = (a_blk * torch.exp2(sa.float()).t().unsqueeze(-1)).reshape(M, K)
        bhi_blk = bhi_fp8.float().reshape(N, num_blocks, bs)
        bhi_deq = (bhi_blk * torch.exp2(sbhi.float()).t().unsqueeze(-1)).reshape(N, K)
        blo_blk = blo_fp8.float().reshape(N, num_blocks, bs)
        blo_deq = (blo_blk * torch.exp2(sblo.float()).t().unsqueeze(-1)).reshape(N, K)
        ref = a_deq @ (bhi_deq + blo_deq).t()

        assert torch.allclose(result, ref, atol=1e-2)

    @requires_triton
    def test_small_problem_no_regression(self) -> None:
        """Tiny problem (total_tiles < NUM_SMs): loop runs once, output correct."""
        M, K, N = 64, 64, 64
        bs = 32

        a_fp8, sa = _make_block_quantized(M, K, bs)
        bhi_fp8, sbhi = _make_block_quantized(N, K, bs)
        blo_fp8, sblo = _make_block_quantized(N, K, bs)

        result = fused_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            a_fp8, bhi_fp8, blo_fp8, sa, sbhi, sblo,
            block_size=bs, persistent=True,
        )

        num_blocks = K // bs
        a_blk = a_fp8.float().reshape(M, num_blocks, bs)
        a_deq = (a_blk * torch.exp2(sa.float()).t().unsqueeze(-1)).reshape(M, K)
        bhi_blk = bhi_fp8.float().reshape(N, num_blocks, bs)
        bhi_deq = (bhi_blk * torch.exp2(sbhi.float()).t().unsqueeze(-1)).reshape(N, K)
        blo_blk = blo_fp8.float().reshape(N, num_blocks, bs)
        blo_deq = (blo_blk * torch.exp2(sblo.float()).t().unsqueeze(-1)).reshape(N, K)
        ref = a_deq @ (bhi_deq + blo_deq).t()

        assert torch.allclose(result, ref, atol=1e-2)

    @requires_triton
    def test_non_divisible_dimensions(self) -> None:
        """Non-divisible M, N: partial tiles handled correctly in persistent mode."""
        M, K, N = 100, 128, 100
        bs = 64

        a_fp8, sa = _make_block_quantized(M, K, bs)
        bhi_fp8, sbhi = _make_block_quantized(N, K, bs)
        blo_fp8, sblo = _make_block_quantized(N, K, bs)

        result = fused_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            a_fp8, bhi_fp8, blo_fp8, sa, sbhi, sblo,
            block_size=bs, persistent=True,
        )

        num_blocks = K // bs
        a_blk = a_fp8.float().reshape(M, num_blocks, bs)
        a_deq = (a_blk * torch.exp2(sa.float()).t().unsqueeze(-1)).reshape(M, K)
        bhi_blk = bhi_fp8.float().reshape(N, num_blocks, bs)
        bhi_deq = (bhi_blk * torch.exp2(sbhi.float()).t().unsqueeze(-1)).reshape(N, K)
        blo_blk = blo_fp8.float().reshape(N, num_blocks, bs)
        blo_deq = (blo_blk * torch.exp2(sblo.float()).t().unsqueeze(-1)).reshape(N, K)
        ref = a_deq @ (bhi_deq + blo_deq).t()

        assert torch.allclose(result, ref, atol=1e-2)

    @requires_triton
    def test_persistent_flag_false(self) -> None:
        """persistent=False produces identical output to persistent=True."""
        M, K, N = 256, 256, 256
        bs = 128

        a_fp8, sa = _make_block_quantized(M, K, bs)
        bhi_fp8, sbhi = _make_block_quantized(N, K, bs)
        blo_fp8, sblo = _make_block_quantized(N, K, bs)

        result_persistent = fused_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            a_fp8, bhi_fp8, blo_fp8, sa, sbhi, sblo,
            block_size=bs, persistent=True,
        )
        result_non_persistent = fused_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            a_fp8, bhi_fp8, blo_fp8, sa, sbhi, sblo,
            block_size=bs, persistent=False,
        )

        assert torch.allclose(result_persistent, result_non_persistent, atol=1e-6)


# ── ABD (Asymmetric Backward Decomposition) ────────────────────────────


class TestABD:
    """Tests for ABD with block-scaled path."""

    @requires_triton
    def test_abd_forward_unchanged(self) -> None:
        """ABD=True produces identical forward output to ABD=False."""
        from tcfp.nn import TCFPLinear

        torch.manual_seed(42)
        layer_std = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=False,
            scale_block_size=32,
        ).to(DEVICE)
        layer_abd = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=False,
            scale_block_size=32, abd=True,
        ).to(DEVICE)
        layer_abd.weight = layer_std.weight
        layer_abd.bias = layer_std.bias

        x = torch.randn(8, 128, device=DEVICE)
        out_std = layer_std(x)
        out_abd = layer_abd(x)
        assert torch.equal(out_std, out_abd), "ABD should not affect forward"

    @requires_triton
    def test_abd_backward_shapes(self) -> None:
        """Gradients have correct shapes with ABD + block scaling."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, scale_block_size=32, abd=True,
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == (8, 128)
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (64, 128)

    @requires_triton
    def test_abd_convergence(self) -> None:
        """Training loss decreases with ABD + block scaling."""
        from tcfp.nn import TCFPLinear

        torch.manual_seed(42)
        model = torch.nn.Sequential(
            TCFPLinear(
                128, 64, use_tensor_cores=True,
                scale_block_size=32, abd=True,
            ),
            torch.nn.ReLU(),
            TCFPLinear(
                64, 32, use_tensor_cores=True,
                scale_block_size=32, abd=True,
            ),
        ).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

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

        assert losses[-1] < losses[0] * 0.5, (
            f"ABD loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    @requires_triton
    def test_abd_convert_to_tcfp(self) -> None:
        """convert_to_tcfp(abd=True) propagates to all TCFPLinear layers."""
        from tcfp.nn import TCFPLinear, convert_to_tcfp

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        ).to(DEVICE)
        convert_to_tcfp(
            model, skip_patterns=(),
            use_tensor_cores=True, scale_block_size=32, abd=True,
        )
        for module in model.modules():
            if isinstance(module, TCFPLinear):
                assert module.abd is True


# ── SRR (Stochastic Residual Rounding) ──────────────────────────────────


class TestSRR:
    """Tests for SRR with block-scaled path."""

    @requires_triton
    def test_srr_forward_runs(self) -> None:
        """SRR block-scaled forward produces correct output shape."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128, 64, use_tensor_cores=True,
            scale_block_size=32, srr=True,
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (8, 64)
        assert torch.isfinite(out).all()

    @requires_triton
    def test_srr_no_error_state(self) -> None:
        """SRR disables error feedback buffer."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128, 64, use_tensor_cores=True,
            scale_block_size=32, srr=True,
        )
        assert layer._error_state is None

    @requires_triton
    def test_srr_backward_finite(self) -> None:
        """Gradients are finite with block-scaled SRR."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            128, 64, bias=True, use_tensor_cores=True,
            scale_block_size=32, srr=True,
        ).to(DEVICE)
        x = torch.randn(8, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()

    @requires_triton
    def test_srr_convergence(self) -> None:
        """Training loss decreases with block-scaled SRR."""
        from tcfp.nn import TCFPLinear

        torch.manual_seed(42)
        model = torch.nn.Sequential(
            TCFPLinear(
                128, 64, use_tensor_cores=True,
                scale_block_size=32, srr=True,
            ),
            torch.nn.ReLU(),
            TCFPLinear(
                64, 32, use_tensor_cores=True,
                scale_block_size=32, srr=True,
            ),
        ).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

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

        assert losses[-1] < losses[0] * 0.5, (
            f"SRR loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    @requires_triton
    def test_srr_convert_to_tcfp(self) -> None:
        """convert_to_tcfp(srr=True) propagates to all TCFPLinear layers."""
        from tcfp.nn import TCFPLinear, convert_to_tcfp

        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
        ).to(DEVICE)
        convert_to_tcfp(
            model, skip_patterns=(),
            use_tensor_cores=True, scale_block_size=32, srr=True,
        )
        for module in model.modules():
            if isinstance(module, TCFPLinear):
                assert module.srr is True
                assert module._error_state is None


# ── Test: Fused weight-quantise block-scaled dual GEMM ───────────────────


class TestFusedWtQuantDualGemm:
    """Tests for fused_wtquant_block_scaled_dual_gemm_forward."""

    @requires_triton
    @pytest.mark.parametrize(
        "M,K,N,block_size",
        [
            (64, 64, 64, 32),
            (128, 128, 128, 64),
            (64, 128, 128, 128),
        ],
    )
    def test_output_shape_and_dtype(self, M: int, K: int, N: int, block_size: int) -> None:
        """Output has shape (M, N), dtype float32."""
        act_fp8, act_scales = _make_block_quantized(M, K, block_size)
        weight = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)

        out, w_hi, w_lo, s_hi, s_lo = fused_wtquant_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            act_fp8, weight, act_scales, block_size=block_size,
        )

        assert out.shape == (M, N)
        assert out.dtype == torch.float32
        assert w_hi.shape == (N, K)
        assert w_hi.dtype == torch.float8_e4m3fn
        assert s_hi.shape == (K // block_size, N)
        assert s_hi.dtype == torch.int8
        assert w_lo is not None and w_lo.shape == (N, K)
        assert s_lo is not None and s_lo.shape == (K // block_size, N)

    @requires_triton
    @pytest.mark.parametrize("block_size", [32, 64, 128])
    def test_matches_separate_quantize_plus_gemm(self, block_size: int) -> None:
        """Output matches fused_dual_quantize_fp8 + fused_block_scaled_dual_gemm_forward."""
        M, K, N = 128, 128, 128
        act_fp8, act_scales = _make_block_quantized(M, K, block_size)
        weight = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)

        # Reference: separate quantise then GEMM
        from tcfp.kernels import fused_dual_quantize_fp8  # pyright: ignore[reportMissingImports]

        w_hi_ref, w_lo_ref, s_hi_ref, s_lo_ref = fused_dual_quantize_fp8(
            weight, block_size=block_size,
        )
        ref_out = fused_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            act_fp8, w_hi_ref, w_lo_ref, act_scales, s_hi_ref, s_lo_ref,
            block_size=block_size,
        )

        # Fused kernel
        out, _, _, _, _ = fused_wtquant_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            act_fp8, weight, act_scales, block_size=block_size,
        )

        # Allow slightly larger tolerance: W quantisation is per-CTA-tile so
        # numerical order differs, but outputs should be close.
        assert torch.allclose(out, ref_out, atol=1e-2, rtol=1e-2), (
            f"Max diff: {(out - ref_out).abs().max().item():.6f}"
        )

    @requires_triton
    def test_output_is_finite(self) -> None:
        """Forward pass produces finite values for random inputs."""
        M, K, N, bs = 128, 128, 64, 64
        act_fp8, act_scales = _make_block_quantized(M, K, bs)
        weight = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)

        out, w_hi, w_lo, s_hi, s_lo = fused_wtquant_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            act_fp8, weight, act_scales, block_size=bs,
        )

        assert torch.isfinite(out).all()
        assert torch.isfinite(w_hi.float()).all()
        assert w_lo is not None and torch.isfinite(w_lo.float()).all()

    @requires_triton
    def test_abd_mode_returns_none_wlo(self) -> None:
        """With save_w_lo=False, W_lo and its scales are returned as None."""
        M, K, N, bs = 64, 64, 64, 32
        act_fp8, act_scales = _make_block_quantized(M, K, bs)
        weight = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)

        out, w_hi, w_lo, s_hi, s_lo = fused_wtquant_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            act_fp8, weight, act_scales, block_size=bs, save_w_lo=False,
        )

        assert w_lo is None
        assert s_lo is None
        assert out.shape == (M, N)
        assert torch.isfinite(out).all()

    @requires_triton
    def test_error_feedback_updates_buffer(self) -> None:
        """error_buf is updated in-place after the forward pass.

        Uses a non-zero initial error buffer to ensure the update is visible.
        Dual FP8 achieves near-zero residual for zero-initialized buffers
        (quantisation error ≈ machine precision), so a non-zero initial value
        is required to detect the write.
        """
        M, K, N, bs = 64, 128, 64, 32
        act_fp8, act_scales = _make_block_quantized(M, K, bs)
        weight = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)

        # Seed error_buf with random non-zero values (simulates step 2 onward)
        torch.manual_seed(99)
        error_buf = torch.randn(N, K, device=DEVICE, dtype=torch.float32) * 0.5
        original_error = error_buf.clone()

        fused_wtquant_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            act_fp8, weight, act_scales, block_size=bs, error_buf=error_buf,
        )

        # error_buf must have been overwritten (the kernel uses original then Python recomputes)
        assert not torch.equal(error_buf, original_error), (
            "error_buf was not updated after the forward pass"
        )
        assert torch.isfinite(error_buf).all()

    @requires_triton
    def test_error_feedback_affects_output(self) -> None:
        """A non-zero error buffer changes the GEMM output (kernel reads it).

        TCFP-12 dual-FP8 achieves near-zero quantisation residual in steady
        state, so testing residual magnitude across two steps is not meaningful.
        Instead, verify the kernel actually reads the error buffer by showing
        that a large non-zero buffer changes the output vs no-buffer mode.
        """
        M, K, N, bs = 64, 128, 64, 32
        act_fp8, act_scales = _make_block_quantized(M, K, bs)
        weight = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)

        # Run without error feedback
        out_no_ef, _, _, _, _ = fused_wtquant_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            act_fp8, weight, act_scales, block_size=bs, error_buf=None,
        )

        # Run with a large non-zero error buffer (simulates accumulated residual)
        torch.manual_seed(7)
        error_buf = torch.randn(N, K, device=DEVICE, dtype=torch.float32) * 0.5
        out_with_ef, _, _, _, _ = fused_wtquant_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            act_fp8, weight, act_scales, block_size=bs, error_buf=error_buf.clone(),
        )

        # Output MUST differ: kernel adds error to W before quantising, so the
        # quantised W changes and the GEMM result changes.
        assert not torch.allclose(out_no_ef, out_with_ef, atol=1e-6), (
            "Error feedback had no effect on output — kernel may not be reading it"
        )

    @requires_triton
    def test_srr_mode_runs_without_error(self) -> None:
        """stochastic_lo=True (SRR) produces finite output without raising."""
        M, K, N, bs = 64, 64, 64, 32
        act_fp8, act_scales = _make_block_quantized(M, K, bs)
        weight = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)

        out, w_hi, w_lo, s_hi, s_lo = fused_wtquant_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            act_fp8, weight, act_scales, block_size=bs, stochastic_lo=True,
        )
        assert out.shape == (M, N)
        assert torch.isfinite(out).all()

    @requires_triton
    def test_srr_and_error_buf_mutually_exclusive(self) -> None:
        """Combining stochastic_lo=True and error_buf raises ValueError."""
        M, K, N, bs = 64, 64, 64, 32
        act_fp8, act_scales = _make_block_quantized(M, K, bs)
        weight = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16)
        error_buf = torch.zeros(N, K, device=DEVICE, dtype=torch.float32)

        with pytest.raises(ValueError, match="mutually exclusive"):
            fused_wtquant_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
                act_fp8, weight, act_scales, block_size=bs,
                stochastic_lo=True, error_buf=error_buf,
            )

    @requires_triton
    def test_whi_scales_are_int8_exponents(self) -> None:
        """W_hi scales are stored as int8 E8M0 exponents, clamped >= -126."""
        M, K, N, bs = 64, 128, 64, 32
        act_fp8, act_scales = _make_block_quantized(M, K, bs)
        weight = torch.randn(N, K, device=DEVICE, dtype=torch.bfloat16) * 10

        _, _, _, s_hi, s_lo = fused_wtquant_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
            act_fp8, weight, act_scales, block_size=bs,
        )

        assert s_hi.dtype == torch.int8
        # Valid exponents are clamped to >= -126; only zero-residual sentinel is -128
        assert (s_hi >= -126).all(), "W_hi scales contain values below clamped minimum"
        assert s_lo is not None
        assert s_lo.dtype == torch.int8

    @requires_triton
    def test_integration_via_tcfplinear_large_n(self) -> None:
        """TCFPLinear with scale_block_size and large N uses fused_wtquant path."""
        from tcfp.nn import TCFPLinear

        # Use large N > 512 to trigger the fused_wtquant dispatch branch
        # (small N <= 512 uses fused_act path instead)
        layer = TCFPLinear(
            512, 1024, use_tensor_cores=True, scale_block_size=64,
        ).to(DEVICE)
        x = torch.randn(32, 512, device=DEVICE)
        out = layer(x)
        assert out.shape == (32, 1024)
        assert torch.isfinite(out).all()

    @requires_triton
    def test_integration_backward_finite(self) -> None:
        """Gradients are finite when using fused_wtquant path."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            512, 1024, bias=True, use_tensor_cores=True, scale_block_size=64,
        ).to(DEVICE)
        x = torch.randn(32, 512, device=DEVICE, requires_grad=True)
        out = layer(x)
        out.sum().backward()

        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
        assert layer.bias is not None
        assert layer.bias.grad is not None and torch.isfinite(layer.bias.grad).all()

    @requires_triton
    def test_integration_convergence(self) -> None:
        """Training loss decreases with fused_wtquant path over 50 steps."""
        from tcfp.nn import TCFPLinear

        torch.manual_seed(42)
        model = torch.nn.Sequential(
            TCFPLinear(512, 1024, use_tensor_cores=True, scale_block_size=64),
            torch.nn.ReLU(),
            TCFPLinear(1024, 256, use_tensor_cores=True, scale_block_size=64),
        ).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(32, 512, device=DEVICE)
        target = torch.randn(32, 256, device=DEVICE)

        losses: list[float] = []
        for _step in range(50):
            opt.zero_grad()
            out = model(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] * 0.5, (
            f"fused_wtquant loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
