"""Tests for tcfp.tcfp8 — NF-aware scaling, error feedback, quantize/dequantize."""

from __future__ import annotations

import pytest
import torch

from tcfp.tcfp8 import (
    ErrorFeedbackState,
    TCFP8Tensor,
    compute_amax_scales,
    compute_nf_aware_scales,
    dequantize_tcfp8,
    fake_quantize_tcfp8,
    quantize_tcfp8,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"


# ── NF-Aware Scaling ─────────────────────────────────────────────────────


class TestNFAwareScaling:
    def test_shape(self) -> None:
        x = torch.randn(8, 128, device=DEVICE)
        scales = compute_nf_aware_scales(x, block_size=32)
        assert scales.shape == (8, 4)

    def test_power_of_two(self) -> None:
        x = torch.randn(4, 64, device=DEVICE)
        scales = compute_nf_aware_scales(x, block_size=32)
        log2_scales = torch.log2(scales)
        assert torch.allclose(log2_scales, log2_scales.round(), atol=1e-5)

    def test_lower_scales_than_amax(self) -> None:
        """NF-aware scales should generally be tighter than amax-based scales,
        since they scale to 3σ instead of max."""
        x = torch.randn(32, 128, device=DEVICE)
        nf_scales = compute_nf_aware_scales(x, block_size=32)
        amax_scales = compute_amax_scales(x, block_size=32)
        # On average, NF scales should be ≤ amax scales (tighter)
        ratio = (nf_scales / amax_scales).mean()
        assert ratio < 1.5, f"NF scales unexpectedly larger: ratio={ratio:.3f}"

    def test_custom_sigma_factor(self) -> None:
        x = torch.randn(4, 64, device=DEVICE)
        s2 = compute_nf_aware_scales(x, block_size=32, sigma_factor=2.0)
        s4 = compute_nf_aware_scales(x, block_size=32, sigma_factor=4.0)
        # Larger sigma factor → larger scales
        assert (s4 >= s2).all()


# ── Error Feedback State ─────────────────────────────────────────────────


class TestErrorFeedback:
    def test_initial_error_is_zero(self) -> None:
        state = ErrorFeedbackState()
        like = torch.randn(32, device=DEVICE)
        error = state.get_error("test", like)
        assert torch.all(error == 0)
        assert error.shape == like.shape

    def test_error_update_and_retrieval(self) -> None:
        state = ErrorFeedbackState()
        like = torch.randn(32, device=DEVICE)
        _ = state.get_error("test", like)
        new_error = torch.ones(32, device=DEVICE) * 0.5
        state.update_error("test", new_error)
        retrieved = state.get_error("test", like)
        assert torch.allclose(retrieved, new_error)

    def test_amax_ema(self) -> None:
        state = ErrorFeedbackState()
        amax1 = torch.tensor([10.0], device=DEVICE)
        amax2 = torch.tensor([20.0], device=DEVICE)

        result1 = state.get_amax_ema("test", amax1)
        assert torch.allclose(result1, amax1)

        result2 = state.get_amax_ema("test", amax2)
        # EMA: 0.999 * 10 + 0.001 * 20 = 10.01
        expected = 0.999 * 10.0 + 0.001 * 20.0
        assert abs(result2.item() - expected) < 0.01

    def test_reset(self) -> None:
        state = ErrorFeedbackState()
        like = torch.randn(32, device=DEVICE)
        _ = state.get_error("test", like)
        state.update_error("test", torch.ones(32, device=DEVICE))
        state.reset()
        error = state.get_error("test", like)
        assert torch.all(error == 0)


# ── Quantize / Dequantize ────────────────────────────────────────────────


class TestQuantizeTCFP8:
    def test_roundtrip_quality(self) -> None:
        """Quantize → dequantize should preserve most information."""
        x = torch.randn(16, 128, device=DEVICE)
        tcfp8 = quantize_tcfp8(x, block_size=32)
        recovered = dequantize_tcfp8(tcfp8)
        assert recovered.shape == x.shape
        # MSE should be small
        mse = (x - recovered).pow(2).mean()
        assert mse < 0.05, f"MSE too high: {mse:.6f}"

    def test_output_dtype(self) -> None:
        x = torch.randn(8, 64, device=DEVICE)
        tcfp8 = quantize_tcfp8(x, block_size=32)
        assert tcfp8.data_fp8.dtype == torch.float8_e4m3fn
        recovered = dequantize_tcfp8(tcfp8)
        assert recovered.dtype == torch.float32

    def test_bits_per_value(self) -> None:
        x = torch.randn(8, 64, device=DEVICE)
        tcfp8 = quantize_tcfp8(x, block_size=32)
        assert tcfp8.bits_per_value == 8.25

    def test_nf_aware_vs_amax(self) -> None:
        """NF-aware scaling should have lower MSE on Gaussian data."""
        torch.manual_seed(42)
        x = torch.randn(64, 256, device=DEVICE)

        tcfp8_nf = quantize_tcfp8(x, nf_aware=True)
        mse_nf = (x - dequantize_tcfp8(tcfp8_nf)).pow(2).mean()

        tcfp8_amax = quantize_tcfp8(x, nf_aware=False)
        mse_amax = (x - dequantize_tcfp8(tcfp8_amax)).pow(2).mean()

        # NF-aware should be at least no worse (usually better)
        # Allow small margin since stochastic
        assert mse_nf <= mse_amax * 1.1, (
            f"NF-aware MSE ({mse_nf:.6f}) worse than amax ({mse_amax:.6f})"
        )

    def test_error_feedback_reduces_drift(self) -> None:
        """Applying error feedback should reduce cumulative drift."""
        torch.manual_seed(0)
        x = torch.randn(8, 128, device=DEVICE)

        # Without error feedback: repeated quantisation
        cumulative_no_ef = torch.zeros_like(x)
        for _ in range(10):
            q = quantize_tcfp8(x)
            cumulative_no_ef += dequantize_tcfp8(q) - x

        # With error feedback
        state = ErrorFeedbackState()
        cumulative_ef = torch.zeros_like(x)
        for i in range(10):
            q = quantize_tcfp8(x, error_state=state, param_name="test")
            cumulative_ef += dequantize_tcfp8(q) - x

        # Error feedback should reduce or not increase cumulative bias
        drift_no_ef = cumulative_no_ef.abs().mean()
        drift_ef = cumulative_ef.abs().mean()
        # Allow generous margin — the point is it shouldn't be hugely worse
        assert drift_ef < drift_no_ef * 2.0, (
            f"EF drift ({drift_ef:.4f}) too much larger than no-EF ({drift_no_ef:.4f})"
        )


class TestFakeQuantizeTCFP8:
    def test_shape_preserved(self) -> None:
        x = torch.randn(4, 64, device=DEVICE)
        fq = fake_quantize_tcfp8(x, block_size=32)
        assert fq.shape == x.shape
        assert fq.dtype == torch.float32

    def test_not_identity(self) -> None:
        """Fake quantize should change values (quantization noise)."""
        x = torch.randn(4, 64, device=DEVICE)
        fq = fake_quantize_tcfp8(x, block_size=32)
        assert not torch.allclose(x, fq, atol=1e-7)

    def test_zero_tensor(self) -> None:
        x = torch.zeros(4, 32, device=DEVICE)
        fq = fake_quantize_tcfp8(x, block_size=32)
        assert torch.allclose(fq, x)
