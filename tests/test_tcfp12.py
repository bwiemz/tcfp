"""Tests for tcfp.tcfp12 — FP8 + 4-bit residual quantization."""

from __future__ import annotations

import pytest
import torch

from tcfp.tcfp12 import (
    RESIDUAL_4BIT_MAX,
    TCFP12Tensor,
    dequantize_tcfp12,
    fake_quantize_tcfp12,
    quantize_tcfp12,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"


class TestQuantizeTCFP12:
    def test_roundtrip_quality(self) -> None:
        """TCFP-12 should have lower MSE than plain FP8."""
        from tcfp.tcfp8 import dequantize_tcfp8, quantize_tcfp8

        torch.manual_seed(42)
        x = torch.randn(32, 256, device=DEVICE)

        # TCFP-8
        tcfp8 = quantize_tcfp8(x)
        mse_8 = (x - dequantize_tcfp8(tcfp8)).pow(2).mean()

        # TCFP-12
        tcfp12 = quantize_tcfp12(x)
        mse_12 = (x - dequantize_tcfp12(tcfp12)).pow(2).mean()

        # TCFP-12 must be strictly better
        assert mse_12 < mse_8, (
            f"TCFP-12 MSE ({mse_12:.6f}) not better than TCFP-8 ({mse_8:.6f})"
        )

    def test_output_types(self) -> None:
        x = torch.randn(8, 64, device=DEVICE)
        tcfp12 = quantize_tcfp12(x)
        assert tcfp12.main_fp8.dtype == torch.float8_e4m3fn
        assert tcfp12.residual_int4.dtype == torch.int8
        assert tcfp12.main_scales.dtype == torch.float32
        assert tcfp12.residual_scales.dtype == torch.float32

    def test_residual_range(self) -> None:
        """Residual int4 values should be in [-7, +7]."""
        x = torch.randn(16, 128, device=DEVICE) * 10
        tcfp12 = quantize_tcfp12(x)
        assert tcfp12.residual_int4.abs().max() <= RESIDUAL_4BIT_MAX

    def test_bits_per_value(self) -> None:
        x = torch.randn(8, 64, device=DEVICE)
        tcfp12 = quantize_tcfp12(x)
        # 12 + 2*(8/32) = 12.5
        assert tcfp12.bits_per_value == 12.5

    def test_shape_preservation(self) -> None:
        x = torch.randn(4, 8, 128, device=DEVICE)
        tcfp12 = quantize_tcfp12(x)
        recovered = dequantize_tcfp12(tcfp12)
        assert recovered.shape == x.shape

    def test_mse_quality_gaussian(self) -> None:
        """On Gaussian data, TCFP-12 MSE should be much lower than FP8."""
        torch.manual_seed(0)
        x = torch.randn(128, 256, device=DEVICE)
        tcfp12 = quantize_tcfp12(x)
        recovered = dequantize_tcfp12(tcfp12)
        mse = (x - recovered).pow(2).mean()
        # TCFP-12 should achieve well under 1e-3 MSE on unit Gaussian
        assert mse < 0.01, f"MSE too high: {mse:.6f}"

    def test_snr_improvement(self) -> None:
        """Signal-to-noise ratio should be significantly better than FP8."""
        from tcfp.tcfp8 import dequantize_tcfp8, quantize_tcfp8

        torch.manual_seed(42)
        x = torch.randn(64, 256, device=DEVICE)

        # SNR = 10 * log10(signal_power / noise_power)
        signal_power = x.pow(2).mean()

        # TCFP-8
        noise_8 = (x - dequantize_tcfp8(quantize_tcfp8(x))).pow(2).mean()
        snr_8 = 10 * torch.log10(signal_power / noise_8)

        # TCFP-12
        noise_12 = (x - dequantize_tcfp12(quantize_tcfp12(x))).pow(2).mean()
        snr_12 = 10 * torch.log10(signal_power / noise_12)

        # TCFP-12 should have at least 6 dB better SNR (~4 bits improvement would be 24 dB)
        assert snr_12 > snr_8 + 3, (
            f"SNR improvement insufficient: TCFP-12={snr_12:.1f}dB, TCFP-8={snr_8:.1f}dB"
        )


class TestFakeQuantizeTCFP12:
    def test_shape_preserved(self) -> None:
        x = torch.randn(4, 64, device=DEVICE)
        fq = fake_quantize_tcfp12(x, block_size=32)
        assert fq.shape == x.shape
        assert fq.dtype == torch.float32

    def test_not_identity(self) -> None:
        x = torch.randn(4, 64, device=DEVICE)
        fq = fake_quantize_tcfp12(x, block_size=32)
        assert not torch.allclose(x, fq, atol=1e-7)

    def test_zero_tensor(self) -> None:
        x = torch.zeros(4, 32, device=DEVICE)
        fq = fake_quantize_tcfp12(x, block_size=32)
        assert torch.allclose(fq, x)
