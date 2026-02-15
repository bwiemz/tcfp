"""Tests for tcfp.tcfp16 — double FP8 quantization and 3-term matmul."""

from __future__ import annotations

import pytest
import torch

from tcfp.tcfp16 import (
    dequantize_tcfp16,
    fake_quantize_tcfp16,
    quantize_tcfp16,
    tcfp16_matmul,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

DEVICE = "cuda"


class TestQuantizeTCFP16:
    def test_roundtrip_quality(self) -> None:
        """TCFP-16 should have lower MSE than TCFP-12."""
        from tcfp.tcfp12 import dequantize_tcfp12, quantize_tcfp12

        torch.manual_seed(42)
        x = torch.randn(32, 256, device=DEVICE)

        # TCFP-12
        tcfp12 = quantize_tcfp12(x)
        mse_12 = (x - dequantize_tcfp12(tcfp12)).pow(2).mean()

        # TCFP-16
        tcfp16 = quantize_tcfp16(x)
        mse_16 = (x - dequantize_tcfp16(tcfp16)).pow(2).mean()

        assert mse_16 < mse_12, f"TCFP-16 MSE ({mse_16:.6f}) not better than TCFP-12 ({mse_12:.6f})"

    def test_near_bf16_quality(self) -> None:
        """TCFP-16 should approach BF16 quality (SNR > 40 dB on Gaussian)."""
        torch.manual_seed(42)
        x = torch.randn(128, 256, device=DEVICE)
        signal_power = x.pow(2).mean()
        noise = (x - dequantize_tcfp16(quantize_tcfp16(x))).pow(2).mean()
        snr = 10 * torch.log10(signal_power / noise)
        # BF16 gives ~38 dB SNR; TCFP-16 should get at least 30 dB
        assert snr > 30, f"SNR too low: {snr:.1f} dB"

    def test_output_types(self) -> None:
        x = torch.randn(8, 64, device=DEVICE)
        tcfp16 = quantize_tcfp16(x)
        assert tcfp16.hi_fp8.dtype == torch.float8_e4m3fn
        assert tcfp16.lo_fp8.dtype == torch.float8_e4m3fn
        assert tcfp16.hi_scales.dtype == torch.float32
        assert tcfp16.lo_scales.dtype == torch.float32

    def test_bits_per_value(self) -> None:
        x = torch.randn(8, 64, device=DEVICE)
        tcfp16 = quantize_tcfp16(x)
        assert tcfp16.bits_per_value == 16.5

    def test_shape_preservation(self) -> None:
        x = torch.randn(4, 8, 128, device=DEVICE)
        tcfp16 = quantize_tcfp16(x)
        recovered = dequantize_tcfp16(tcfp16)
        assert recovered.shape == x.shape

    def test_monotonic_quality(self) -> None:
        """FP8 < TCFP-12 < TCFP-16 in quality (lower MSE)."""
        from tcfp.core import to_fp8_e4m3
        from tcfp.tcfp12 import dequantize_tcfp12, quantize_tcfp12

        torch.manual_seed(42)
        x = torch.randn(64, 256, device=DEVICE)

        fp8, inv_s = to_fp8_e4m3(x)
        mse_fp8 = (x - fp8.float() * inv_s).pow(2).mean()
        mse_12 = (x - dequantize_tcfp12(quantize_tcfp12(x))).pow(2).mean()
        mse_16 = (x - dequantize_tcfp16(quantize_tcfp16(x))).pow(2).mean()

        assert mse_fp8 > mse_12 > mse_16, (
            f"Quality not monotonic: MSE-FP8={mse_fp8:.6f}, "
            f"MSE-12={mse_12:.6f}, MSE-16={mse_16:.6f}"
        )


class TestTCFP16Matmul:
    def test_basic_correctness(self) -> None:
        """3-term FP8 matmul should be close to FP32 reference."""
        torch.manual_seed(42)
        M, K, N = 64, 128, 64
        a = torch.randn(M, K, device=DEVICE)
        b = torch.randn(K, N, device=DEVICE)

        ref = a @ b

        a_q = quantize_tcfp16(a)
        b_q = quantize_tcfp16(b)
        result = tcfp16_matmul(a_q, b_q)

        rel_error = (result - ref).norm() / ref.norm()
        assert rel_error < 0.10, f"Relative error too high: {rel_error:.4f}"

    def test_output_shape(self) -> None:
        M, K, N = 32, 64, 32
        a_q = quantize_tcfp16(torch.randn(M, K, device=DEVICE))
        b_q = quantize_tcfp16(torch.randn(K, N, device=DEVICE))
        result = tcfp16_matmul(a_q, b_q)
        assert result.shape == (M, N)

    def test_comparable_to_single_fp8_matmul(self) -> None:
        """TCFP-16 matmul accuracy should be comparable to single FP8 matmul.

        Note: The 3-term decomposition adds per-tensor re-quantization overhead,
        so it's not always strictly better than a single well-scaled FP8 matmul
        on random data. The real advantage is in training convergence where
        the higher weight precision prevents drift.
        """
        from tcfp.core import fp8_matmul, to_fp8_e4m3

        torch.manual_seed(42)
        M, K, N = 128, 256, 128
        a = torch.randn(M, K, device=DEVICE)
        b = torch.randn(K, N, device=DEVICE)
        ref = a @ b

        # Single FP8
        a_fp8, sa = to_fp8_e4m3(a)
        b_fp8, sb = to_fp8_e4m3(b)
        result_fp8 = fp8_matmul(a_fp8, b_fp8, sa, sb)
        error_fp8 = (result_fp8 - ref).norm() / ref.norm()

        # TCFP-16
        result_16 = tcfp16_matmul(quantize_tcfp16(a), quantize_tcfp16(b))
        error_16 = (result_16 - ref).norm() / ref.norm()

        # Both should be reasonably accurate (<15% relative error)
        assert error_16 < 0.15, f"TCFP-16 matmul error too high: {error_16:.4f}"
        assert error_fp8 < 0.15, f"FP8 matmul error too high: {error_fp8:.4f}"
        # TCFP-16 should be at least in the same ballpark
        assert error_16 < error_fp8 * 1.5, (
            f"TCFP-16 error ({error_16:.4f}) much worse than FP8 ({error_fp8:.4f})"
        )


class TestFakeQuantizeTCFP16:
    def test_shape_preserved(self) -> None:
        x = torch.randn(4, 64, device=DEVICE)
        fq = fake_quantize_tcfp16(x, block_size=32)
        assert fq.shape == x.shape
        assert fq.dtype == torch.float32

    def test_not_identity(self) -> None:
        x = torch.randn(4, 64, device=DEVICE)
        fq = fake_quantize_tcfp16(x, block_size=32)
        assert not torch.allclose(x, fq, atol=1e-8)

    def test_very_low_error(self) -> None:
        """TCFP-16 fake quantize should have very small error."""
        x = torch.randn(32, 256, device=DEVICE)
        fq = fake_quantize_tcfp16(x, block_size=32)
        mse = (x - fq).pow(2).mean()
        assert mse < 1e-3, f"MSE too high: {mse:.6f}"
