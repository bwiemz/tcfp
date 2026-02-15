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
        fused_block_scaled_dual_gemm_backward_dx,
        fused_block_scaled_dual_gemm_forward,
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
    M, K = a_fp8.shape
    N = b_fp8.shape[0]
    num_kb = K // block_size

    # Dequantize A per-block
    a_blocked = a_fp8.float().reshape(M, num_kb, block_size)
    a_deq = (a_blocked * sa.unsqueeze(-1)).reshape(M, K)

    # Dequantize B per-block
    b_blocked = b_fp8.float().reshape(N, num_kb, block_size)
    b_deq = (b_blocked * sb.unsqueeze(-1)).reshape(N, K)

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
        assert scales.shape == (M, K // block_size)

    @requires_triton
    def test_output_dtypes(self) -> None:
        x = torch.randn(64, 128, device=DEVICE)
        fp8, scales = block_quantize_fp8(x, block_size=32)  # type: ignore[reportPossiblyUnboundVariable]
        assert fp8.dtype == torch.float8_e4m3fn
        assert scales.dtype == torch.float32

    @requires_triton
    def test_e5m2_dtype(self) -> None:
        x = torch.randn(64, 128, device=DEVICE)
        fp8, scales = block_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            x, block_size=32, fp8_dtype=torch.float8_e5m2,
        )
        assert fp8.dtype == torch.float8_e5m2

    @requires_triton
    def test_scales_are_power_of_2(self) -> None:
        x = torch.randn(64, 128, device=DEVICE) * 10
        _, scales = block_quantize_fp8(x, block_size=32)  # type: ignore[reportPossiblyUnboundVariable]
        log2_scales = torch.log2(scales)
        # Power-of-2 means log2 is integer
        assert torch.allclose(log2_scales, log2_scales.round(), atol=1e-6)

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
        """When all scales are 1.0, should match per-tensor FP8 result."""
        M, K, N = 64, 64, 64
        a = torch.randn(M, K, device=DEVICE)
        b = torch.randn(N, K, device=DEVICE)
        a_fp8 = a.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
        b_fp8 = b.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

        ones_a = torch.ones(M, K // 32, device=DEVICE)
        ones_b = torch.ones(N, K // 32, device=DEVICE)

        result = block_scaled_gemm(a_fp8, b_fp8, ones_a, ones_b, block_size=32)  # type: ignore[reportPossiblyUnboundVariable]
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
        sblo = torch.ones(N, K // bs, device=DEVICE)

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
