"""Tests for fused dual FP8 quantization kernel."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from tcfp.core import FP8_E4M3_MAX, TCFPMode

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
        fused_dual_quantize_fp8,
        is_triton_available,
    )

    _HAS_TRITON = is_triton_available()
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not _HAS_TRITON, reason="Triton not available"
)


# ── Reference: separate block_quantize_fp8 path ──────────────────────────


def _reference_dual_quantize(
    weight: torch.Tensor,
    block_size: int,
    error_buf: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Reference implementation using separate block_quantize_fp8 calls.

    Returns (w_hi_fp8, w_lo_fp8, scales_hi, scales_lo, new_error).
    new_error = weight - dequant(w_hi) - dequant(w_lo).
    """
    w = weight.float()
    if error_buf is not None:
        w_corrected = w + error_buf
    else:
        w_corrected = w

    w_hi_fp8, scales_hi = block_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
        w_corrected, block_size=block_size,
    )
    w_hi_deq = block_dequantize_fp8(w_hi_fp8, scales_hi, block_size)  # type: ignore[reportPossiblyUnboundVariable]
    residual = w_corrected - w_hi_deq

    w_lo_fp8, scales_lo = block_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
        residual, block_size=block_size,
    )
    w_lo_deq = block_dequantize_fp8(w_lo_fp8, scales_lo, block_size)  # type: ignore[reportPossiblyUnboundVariable]

    # Error relative to ORIGINAL weight (before error feedback)
    new_error = w - (w_hi_deq + w_lo_deq)
    return w_hi_fp8, w_lo_fp8, scales_hi, scales_lo, new_error


# ── Bitwise Parity Tests ─────────────────────────────────────────────────


@requires_triton
class TestBitwiseParity:
    """Fused kernel must produce identical results to separate kernels."""

    @pytest.mark.parametrize(
        "N, K, block_size",
        [
            (64, 128, 128),
            (128, 256, 128),
            (256, 512, 128),
            (64, 256, 64),
        ],
        ids=["small", "medium", "large", "bs64"],
    )
    def test_parity_w_hi(self, N: int, K: int, block_size: int) -> None:
        """Fused W_hi matches block_quantize_fp8(weight)."""
        torch.manual_seed(42)
        w = torch.randn(N, K, device=DEVICE)

        w_hi_fused, _, _, _ = fused_dual_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            w, block_size=block_size,
        )
        w_hi_ref, _, _, _, _ = _reference_dual_quantize(w, block_size)

        # Bitwise check — same ops, same order, should be exact
        assert torch.equal(w_hi_fused, w_hi_ref), (
            f"W_hi mismatch: {(w_hi_fused.float() - w_hi_ref.float()).abs().max().item()}"
        )

    @pytest.mark.parametrize(
        "N, K, block_size",
        [
            (64, 128, 128),
            (128, 256, 128),
            (256, 512, 128),
        ],
        ids=["small", "medium", "large"],
    )
    def test_parity_w_lo(self, N: int, K: int, block_size: int) -> None:
        """Fused W_lo matches block_quantize_fp8(residual)."""
        torch.manual_seed(42)
        w = torch.randn(N, K, device=DEVICE)

        _, w_lo_fused, _, _ = fused_dual_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            w, block_size=block_size,
        )
        _, w_lo_ref, _, _, _ = _reference_dual_quantize(w, block_size)

        assert torch.equal(w_lo_fused, w_lo_ref), (
            f"W_lo mismatch: {(w_lo_fused.float() - w_lo_ref.float()).abs().max().item()}"
        )

    @pytest.mark.parametrize(
        "N, K, block_size",
        [
            (64, 128, 128),
            (128, 256, 128),
            (256, 512, 128),
        ],
        ids=["small", "medium", "large"],
    )
    def test_parity_scales(self, N: int, K: int, block_size: int) -> None:
        """Fused scales match separate block_quantize_fp8 scales."""
        torch.manual_seed(42)
        w = torch.randn(N, K, device=DEVICE)

        _, _, sh_fused, sl_fused = fused_dual_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            w, block_size=block_size,
        )
        _, _, sh_ref, sl_ref, _ = _reference_dual_quantize(w, block_size)

        assert torch.equal(sh_fused, sh_ref), (
            f"scales_hi mismatch: {(sh_fused - sh_ref).abs().max().item()}"
        )
        assert torch.equal(sl_fused, sl_ref), (
            f"scales_lo mismatch: {(sl_fused - sl_ref).abs().max().item()}"
        )


# ── Output Format Tests ──────────────────────────────────────────────────


@requires_triton
class TestOutputFormat:
    """Verify shapes, dtypes, and value properties."""

    def test_output_dtypes(self) -> None:
        w = torch.randn(64, 128, device=DEVICE)
        w_hi, w_lo, sh, sl = fused_dual_quantize_fp8(w, block_size=128)  # type: ignore[reportPossiblyUnboundVariable]
        assert w_hi.dtype == torch.float8_e4m3fn
        assert w_lo.dtype == torch.float8_e4m3fn
        assert sh.dtype == torch.float32
        assert sl.dtype == torch.float32

    @pytest.mark.parametrize("block_size", [64, 128])
    def test_scales_shape(self, block_size: int) -> None:
        N, K = 128, 256
        w = torch.randn(N, K, device=DEVICE)
        _, _, sh, sl = fused_dual_quantize_fp8(w, block_size=block_size)  # type: ignore[reportPossiblyUnboundVariable]
        expected = (N, K // block_size)
        assert sh.shape == expected
        assert sl.shape == expected

    def test_scales_are_power_of_2(self) -> None:
        """All scales must be exact powers of 2."""
        torch.manual_seed(42)
        w = torch.randn(128, 256, device=DEVICE) * 10
        _, _, sh, sl = fused_dual_quantize_fp8(w, block_size=128)  # type: ignore[reportPossiblyUnboundVariable]

        for name, scales in [("hi", sh), ("lo", sl)]:
            log2_s = torch.log2(scales)
            assert torch.allclose(log2_s, log2_s.round(), atol=1e-6), (
                f"scales_{name} not power-of-2"
            )

    def test_all_finite(self) -> None:
        torch.manual_seed(123)
        w = torch.randn(256, 256, device=DEVICE)
        w_hi, w_lo, sh, sl = fused_dual_quantize_fp8(w, block_size=128)  # type: ignore[reportPossiblyUnboundVariable]
        assert torch.isfinite(w_hi.float()).all()
        assert torch.isfinite(w_lo.float()).all()
        assert torch.isfinite(sh).all()
        assert torch.isfinite(sl).all()

    def test_bf16_input(self) -> None:
        """BF16 weight is handled correctly (cast to FP32 inside kernel)."""
        torch.manual_seed(42)
        w_f32 = torch.randn(64, 128, device=DEVICE)
        w_bf16 = w_f32.bfloat16()

        hi_f32, lo_f32, sh_f32, sl_f32 = fused_dual_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            w_f32, block_size=128,
        )
        hi_bf, lo_bf, sh_bf, sl_bf = fused_dual_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            w_bf16, block_size=128,
        )
        # BF16 input may have small differences from FP32 due to input precision
        # but shapes and dtypes must match
        assert hi_bf.shape == hi_f32.shape
        assert hi_bf.dtype == torch.float8_e4m3fn


# ── Error Feedback Tests ─────────────────────────────────────────────────


@requires_triton
class TestErrorFeedback:
    """Verify in-place error buffer semantics."""

    def test_error_semantics(self) -> None:
        """new_error = weight - dequant(quantize(weight + old_error))."""
        torch.manual_seed(42)
        N, K, bs = 128, 256, 128
        w = torch.randn(N, K, device=DEVICE)
        old_error = torch.randn(N, K, device=DEVICE) * 0.01

        error_buf = old_error.clone()
        w_hi, w_lo, sh, sl = fused_dual_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            w, block_size=bs, error_buf=error_buf,
        )

        # Compute expected error
        _, _, _, _, expected_error = _reference_dual_quantize(
            w, bs, error_buf=old_error,
        )

        # error_buf should now contain the new error
        torch.testing.assert_close(error_buf, expected_error, atol=1e-5, rtol=1e-5)

    def test_error_inplace(self) -> None:
        """error_buf tensor is modified in-place (same data_ptr)."""
        torch.manual_seed(42)
        w = torch.randn(64, 128, device=DEVICE)
        error_buf = torch.zeros(64, 128, device=DEVICE)
        data_ptr_before = error_buf.data_ptr()

        fused_dual_quantize_fp8(w, block_size=128, error_buf=error_buf)  # type: ignore[reportPossiblyUnboundVariable]

        assert error_buf.data_ptr() == data_ptr_before
        # Error should be non-zero (quantization always has some error)
        assert error_buf.abs().sum() > 0

    def test_no_error_matches_zero_error(self) -> None:
        """Without error_buf, result matches error_buf=zeros."""
        torch.manual_seed(42)
        N, K, bs = 64, 128, 128
        w = torch.randn(N, K, device=DEVICE)

        # Without error feedback
        hi_no_err, lo_no_err, sh_no, sl_no = fused_dual_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            w, block_size=bs,
        )

        # With zero error buffer
        zero_err = torch.zeros(N, K, device=DEVICE)
        hi_zero, lo_zero, sh_z, sl_z = fused_dual_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            w, block_size=bs, error_buf=zero_err,
        )

        assert torch.equal(hi_no_err, hi_zero)
        assert torch.equal(lo_no_err, lo_zero)
        assert torch.equal(sh_no, sh_z)
        assert torch.equal(sl_no, sl_z)

    def test_error_with_parity(self) -> None:
        """Fused with error feedback matches reference with error feedback."""
        torch.manual_seed(42)
        N, K, bs = 128, 256, 128
        w = torch.randn(N, K, device=DEVICE)
        old_error = torch.randn(N, K, device=DEVICE) * 0.01

        error_buf = old_error.clone()
        hi_f, lo_f, sh_f, sl_f = fused_dual_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            w, block_size=bs, error_buf=error_buf,
        )

        hi_r, lo_r, sh_r, sl_r, _ = _reference_dual_quantize(
            w, bs, error_buf=old_error,
        )

        assert torch.equal(hi_f, hi_r), "W_hi mismatch with error feedback"
        assert torch.equal(lo_f, lo_r), "W_lo mismatch with error feedback"
        assert torch.equal(sh_f, sh_r), "scales_hi mismatch with error feedback"
        assert torch.equal(sl_f, sl_r), "scales_lo mismatch with error feedback"


# ── Assertion / Edge Case Tests ──────────────────────────────────────────


@requires_triton
class TestEdgeCases:
    """Input validation and edge cases."""

    def test_contiguous_assert(self) -> None:
        w = torch.randn(64, 128, device=DEVICE).t()  # non-contiguous
        with pytest.raises(AssertionError, match="contiguous"):
            fused_dual_quantize_fp8(w, block_size=128)  # type: ignore[reportPossiblyUnboundVariable]

    def test_k_divisibility_assert(self) -> None:
        w = torch.randn(64, 100, device=DEVICE)  # 100 not divisible by 128
        with pytest.raises(AssertionError, match="divisible"):
            fused_dual_quantize_fp8(w, block_size=128)  # type: ignore[reportPossiblyUnboundVariable]

    def test_error_buf_dtype_assert(self) -> None:
        w = torch.randn(64, 128, device=DEVICE)
        bad_err = torch.zeros(64, 128, device=DEVICE, dtype=torch.bfloat16)
        with pytest.raises(AssertionError, match="float32"):
            fused_dual_quantize_fp8(w, block_size=128, error_buf=bad_err)  # type: ignore[reportPossiblyUnboundVariable]

    def test_error_buf_shape_assert(self) -> None:
        w = torch.randn(64, 128, device=DEVICE)
        bad_err = torch.zeros(32, 128, device=DEVICE)
        with pytest.raises(AssertionError, match="shape"):
            fused_dual_quantize_fp8(w, block_size=128, error_buf=bad_err)  # type: ignore[reportPossiblyUnboundVariable]

    def test_zero_weight(self) -> None:
        """All-zero weight produces finite outputs."""
        w = torch.zeros(64, 128, device=DEVICE)
        w_hi, w_lo, sh, sl = fused_dual_quantize_fp8(w, block_size=128)  # type: ignore[reportPossiblyUnboundVariable]
        assert torch.isfinite(w_hi.float()).all()
        assert torch.isfinite(w_lo.float()).all()
        assert torch.isfinite(sh).all()
        assert torch.isfinite(sl).all()

    def test_large_values(self) -> None:
        """Large weight values produce finite, clamped outputs."""
        w = torch.randn(64, 128, device=DEVICE) * 1000
        w_hi, w_lo, sh, sl = fused_dual_quantize_fp8(w, block_size=128)  # type: ignore[reportPossiblyUnboundVariable]
        assert torch.isfinite(w_hi.float()).all()
        assert w_hi.float().abs().max() <= FP8_E4M3_MAX
        assert torch.isfinite(sh).all()

    @pytest.mark.parametrize("block_size", [64, 128])
    def test_block_sizes(self, block_size: int) -> None:
        """Works with different block sizes."""
        N, K = 128, 256
        w = torch.randn(N, K, device=DEVICE)
        w_hi, w_lo, sh, sl = fused_dual_quantize_fp8(w, block_size=block_size)  # type: ignore[reportPossiblyUnboundVariable]
        assert w_hi.shape == (N, K)
        assert sh.shape == (N, K // block_size)


# ── Integration Tests ────────────────────────────────────────────────────


@requires_triton
class TestIntegration:
    """End-to-end tests with GEMM and TCFPLinear."""

    def test_gemm_roundtrip(self) -> None:
        """Fused quant → block-scaled GEMM matches separate path."""
        from tcfp.kernels import fused_block_scaled_dual_gemm_forward

        torch.manual_seed(42)
        M, N, K, bs = 128, 64, 128, 128
        w = torch.randn(N, K, device=DEVICE)
        act = torch.randn(M, K, device=DEVICE)

        # Fused quantize
        w_hi, w_lo, sh, sl = fused_dual_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            w, block_size=bs,
        )
        act_fp8, act_s = block_quantize_fp8(act, block_size=bs)  # type: ignore[reportPossiblyUnboundVariable]

        out_fused = fused_block_scaled_dual_gemm_forward(
            act_fp8, w_hi, w_lo, act_s, sh, sl, block_size=bs,
        )

        # Separate quantize
        w_hi_r, w_lo_r, sh_r, sl_r, _ = _reference_dual_quantize(w, bs)
        out_ref = fused_block_scaled_dual_gemm_forward(
            act_fp8, w_hi_r, w_lo_r, act_s, sh_r, sl_r, block_size=bs,
        )

        torch.testing.assert_close(out_fused, out_ref, atol=1e-3, rtol=1e-3)

    def test_training_convergence(self) -> None:
        """Loss decreases using fused quant path."""
        from tcfp.nn import convert_to_tcfp

        torch.manual_seed(42)
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
            scale_block_size=128,
        )
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        losses: list[float] = []
        for _ in range(50):
            out = model(x)
            loss = nn.functional.mse_loss(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert all(math.isfinite(v) for v in losses), "NaN or Inf in losses"
        assert late < early, (
            f"Loss did not decrease: {early:.4f} -> {late:.4f}"
        )
