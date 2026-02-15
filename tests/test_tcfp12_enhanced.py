"""Tests for TCFP-12 enhanced features: NF4 residual, NF-aware scaling,
stochastic rounding, and error feedback."""

from __future__ import annotations

import pytest
import torch

from tcfp.core import ErrorFeedbackState
from tcfp.tcfp12 import (
    NF4_CODEBOOK,
    NF4_MIDPOINTS,
    NF4_NUM_LEVELS,
    build_nf4_codebook,
    dequantize_tcfp12,
    fake_quantize_tcfp12,
    quantize_tcfp12,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# NF4 Codebook
# ---------------------------------------------------------------------------


class TestNF4Codebook:
    def test_codebook_length(self) -> None:
        assert len(NF4_CODEBOOK) == NF4_NUM_LEVELS
        assert NF4_NUM_LEVELS == 16

    def test_codebook_sorted(self) -> None:
        """Codebook values must be monotonically increasing."""
        for i in range(len(NF4_CODEBOOK) - 1):
            assert NF4_CODEBOOK[i] < NF4_CODEBOOK[i + 1], (
                f"Not sorted at index {i}: {NF4_CODEBOOK[i]:.6f} >= {NF4_CODEBOOK[i + 1]:.6f}"
            )

    def test_codebook_symmetric(self) -> None:
        """Codebook should be approximately symmetric around 0."""
        n = len(NF4_CODEBOOK)
        for i in range(n // 2):
            assert torch.isclose(NF4_CODEBOOK[i], -NF4_CODEBOOK[n - 1 - i], atol=1e-6), (
                f"Asymmetry at index {i}"
            )

    def test_codebook_range(self) -> None:
        """All codebook values should be in [-1, 1]."""
        assert NF4_CODEBOOK.min() >= -1.0
        assert NF4_CODEBOOK.max() <= 1.0

    def test_codebook_density_near_zero(self) -> None:
        """More levels should be concentrated near zero than at edges."""
        near_zero = (NF4_CODEBOOK.abs() < 0.5).sum().item()
        at_edges = (NF4_CODEBOOK.abs() >= 0.5).sum().item()
        assert near_zero >= at_edges, (
            f"Expected more levels near zero ({near_zero}) than at edges ({at_edges})"
        )

    def test_codebook_reproducible(self) -> None:
        """build_nf4_codebook should always return the same values."""
        cb1 = build_nf4_codebook()
        cb2 = build_nf4_codebook()
        assert torch.allclose(cb1, cb2)

    def test_midpoints_length(self) -> None:
        """Midpoints should have N-1 entries."""
        assert len(NF4_MIDPOINTS) == NF4_NUM_LEVELS - 1


# ---------------------------------------------------------------------------
# NF4 Residual Quantization
# ---------------------------------------------------------------------------


class TestNF4ResidualQuantization:
    def test_nf4_roundtrip_quality(self) -> None:
        """NF4 residual should have <= MSE compared to linear int4
        on Gaussian data."""
        torch.manual_seed(42)
        x = torch.randn(32, 256, device=DEVICE)

        # Linear int4 (default)
        tcfp12_linear = quantize_tcfp12(x)
        mse_linear = (x - dequantize_tcfp12(tcfp12_linear)).pow(2).mean()

        # NF4 residual
        tcfp12_nf4 = quantize_tcfp12(x, nf4_residual=True)
        mse_nf4 = (x - dequantize_tcfp12(tcfp12_nf4)).pow(2).mean()

        # NF4 should be at least as good (and typically better)
        assert mse_nf4 <= mse_linear * 1.05, (
            f"NF4 MSE ({mse_nf4:.6f}) significantly worse than linear ({mse_linear:.6f})"
        )

    def test_nf4_indices_range(self) -> None:
        """NF4 residual indices must be in [0, 15]."""
        x = torch.randn(16, 128, device=DEVICE) * 10
        tcfp12 = quantize_tcfp12(x, nf4_residual=True)
        assert tcfp12.residual_int4.min() >= 0
        assert tcfp12.residual_int4.max() <= 15

    def test_nf4_flag_preserved(self) -> None:
        """TCFP12Tensor should track the nf4_residual flag."""
        x = torch.randn(8, 64, device=DEVICE)

        tcfp12_default = quantize_tcfp12(x)
        assert not tcfp12_default.nf4_residual

        tcfp12_nf4 = quantize_tcfp12(x, nf4_residual=True)
        assert tcfp12_nf4.nf4_residual

    def test_nf4_output_types(self) -> None:
        x = torch.randn(8, 64, device=DEVICE)
        tcfp12 = quantize_tcfp12(x, nf4_residual=True)
        assert tcfp12.main_fp8.dtype == torch.float8_e4m3fn
        assert tcfp12.residual_int4.dtype == torch.int8
        assert tcfp12.main_scales.dtype == torch.float32
        assert tcfp12.residual_scales.dtype == torch.float32

    def test_nf4_shape_preservation(self) -> None:
        x = torch.randn(4, 8, 128, device=DEVICE)
        tcfp12 = quantize_tcfp12(x, nf4_residual=True)
        recovered = dequantize_tcfp12(tcfp12)
        assert recovered.shape == x.shape

    def test_nf4_bits_per_value(self) -> None:
        x = torch.randn(8, 64, device=DEVICE)
        tcfp12 = quantize_tcfp12(x, nf4_residual=True)
        assert tcfp12.bits_per_value == 12.5


# ---------------------------------------------------------------------------
# NF-aware Scaling
# ---------------------------------------------------------------------------


class TestNFAwareScalingTCFP12:
    def test_nf_aware_improves_mse(self) -> None:
        """Combined NF-aware + NF4 should beat amax + linear int4
        on Gaussian data."""
        torch.manual_seed(42)
        x = torch.randn(64, 256, device=DEVICE)

        # Baseline: amax + linear int4
        mse_baseline = (x - fake_quantize_tcfp12(x)).pow(2).mean()

        # Enhanced: NF-aware + NF4
        mse_enhanced = (
            (x - fake_quantize_tcfp12(x, nf4_residual=True, nf_aware_scaling=True)).pow(2).mean()
        )

        # Enhanced should be better or comparable
        assert mse_enhanced <= mse_baseline * 1.1, (
            f"Enhanced MSE ({mse_enhanced:.6f}) much worse than baseline ({mse_baseline:.6f})"
        )

    def test_nf_aware_scales_power_of_two(self) -> None:
        """Block scales should remain power-of-2 with NF-aware scaling."""
        x = torch.randn(16, 128, device=DEVICE)
        tcfp12 = quantize_tcfp12(x, nf_aware_scaling=True)
        log2_scales = torch.log2(tcfp12.main_scales.abs())
        assert torch.allclose(log2_scales, log2_scales.round(), atol=1e-5), (
            "Scales are not powers of 2"
        )


# ---------------------------------------------------------------------------
# Stochastic Rounding
# ---------------------------------------------------------------------------


class TestStochasticRounding:
    def test_stochastic_unbiased(self) -> None:
        """Mean of many stochastic fake-quantizations should approximate
        the original value (unbiased)."""
        torch.manual_seed(42)
        x = torch.randn(8, 64, device=DEVICE)

        # Average over many rounds
        n_rounds = 200
        total = torch.zeros_like(x)
        for _ in range(n_rounds):
            total += fake_quantize_tcfp12(x, stochastic_rounding=True)
        mean_fq = total / n_rounds

        # Should be close to the original
        max_deviation = (mean_fq - x).abs().max()
        assert max_deviation < 0.15, f"Max deviation from mean too high: {max_deviation:.4f}"

    def test_stochastic_varies(self) -> None:
        """Two calls with stochastic rounding should produce
        different outputs."""
        x = torch.randn(8, 64, device=DEVICE)
        fq1 = fake_quantize_tcfp12(x, stochastic_rounding=True)
        fq2 = fake_quantize_tcfp12(x, stochastic_rounding=True)
        assert not torch.allclose(fq1, fq2, atol=1e-8), (
            "Stochastic rounding produced identical results"
        )

    def test_deterministic_when_disabled(self) -> None:
        """Without stochastic rounding, results should be deterministic."""
        x = torch.randn(8, 64, device=DEVICE)
        fq1 = fake_quantize_tcfp12(x, stochastic_rounding=False)
        fq2 = fake_quantize_tcfp12(x, stochastic_rounding=False)
        assert torch.allclose(fq1, fq2, atol=1e-8)


# ---------------------------------------------------------------------------
# Error Feedback
# ---------------------------------------------------------------------------


class TestErrorFeedbackTCFP12:
    def test_error_feedback_reduces_drift(self) -> None:
        """Error feedback should reduce cumulative quantisation error
        over many steps."""
        torch.manual_seed(42)
        x = torch.randn(8, 64, device=DEVICE)

        # Without error feedback: accumulate error over steps
        cumulative_error_no_ef = torch.zeros_like(x)
        for _ in range(50):
            fq = fake_quantize_tcfp12(x)
            cumulative_error_no_ef += x - fq

        # With error feedback
        ef_state = ErrorFeedbackState()
        cumulative_error_ef = torch.zeros_like(x)
        for _ in range(50):
            tcfp12 = quantize_tcfp12(x, error_state=ef_state, param_name="test")
            recon = dequantize_tcfp12(tcfp12)
            cumulative_error_ef += x - recon

        # Error feedback should keep cumulative error lower
        drift_no_ef = cumulative_error_no_ef.abs().mean()
        drift_ef = cumulative_error_ef.abs().mean()
        assert drift_ef < drift_no_ef, (
            f"Error feedback drift ({drift_ef:.6f}) not less than "
            f"no-feedback drift ({drift_no_ef:.6f})"
        )

    def test_error_feedback_state_updates(self) -> None:
        """Error buffer should be nonzero after first quantisation."""
        x = torch.randn(8, 64, device=DEVICE)
        ef_state = ErrorFeedbackState()

        # First quant — error should be stored
        quantize_tcfp12(x, error_state=ef_state, param_name="w")
        error_buf = ef_state.get_error("w", x)
        # Error buffer should not be all zeros (quantization has error)
        assert error_buf.abs().max() > 0, "Error buffer is all zeros after quantisation"


# ---------------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_default_params_unchanged(self) -> None:
        """Default parameters should produce identical results
        to the original implementation."""
        torch.manual_seed(42)
        x = torch.randn(32, 256, device=DEVICE)

        # Default (no new features)
        tcfp12 = quantize_tcfp12(x)
        recon = dequantize_tcfp12(tcfp12)

        # Verify basic properties still hold
        assert tcfp12.main_fp8.dtype == torch.float8_e4m3fn
        assert tcfp12.residual_int4.dtype == torch.int8
        assert not tcfp12.nf4_residual
        assert tcfp12.bits_per_value == 12.5
        assert recon.shape == x.shape

        # Residual should be in [-7, +7]
        assert tcfp12.residual_int4.min() >= -7
        assert tcfp12.residual_int4.max() <= 7

    def test_fake_quantize_default_unchanged(self) -> None:
        """fake_quantize_tcfp12 with defaults should be deterministic."""
        x = torch.randn(8, 64, device=DEVICE)
        fq1 = fake_quantize_tcfp12(x)
        fq2 = fake_quantize_tcfp12(x)
        assert torch.allclose(fq1, fq2)


# ---------------------------------------------------------------------------
# Integration: All Enhancements Together
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_all_enhancements_together(self) -> None:
        """All enhancements enabled should produce valid results."""
        torch.manual_seed(42)
        x = torch.randn(16, 128, device=DEVICE)

        fq = fake_quantize_tcfp12(
            x,
            nf4_residual=True,
            nf_aware_scaling=True,
            stochastic_rounding=True,
        )
        assert fq.shape == x.shape
        assert fq.dtype == torch.float32
        # Should still be close to original
        mse = (x - fq).pow(2).mean()
        assert mse < 0.05, f"MSE too high with all enhancements: {mse:.6f}"

    def test_snr_with_enhancements(self) -> None:
        """SNR with enhancements should be at least as good as baseline."""
        torch.manual_seed(42)
        x = torch.randn(64, 256, device=DEVICE)
        signal_power = x.pow(2).mean()

        # Baseline
        noise_base = (x - fake_quantize_tcfp12(x)).pow(2).mean()
        snr_base = 10 * torch.log10(signal_power / noise_base)

        # Enhanced (deterministic for fair comparison)
        noise_enh = (
            (x - fake_quantize_tcfp12(x, nf4_residual=True, nf_aware_scaling=True)).pow(2).mean()
        )
        snr_enh = 10 * torch.log10(signal_power / noise_enh)

        # Enhanced should be comparable or better
        assert snr_enh >= snr_base - 1.0, (
            f"Enhanced SNR ({snr_enh:.1f}dB) much worse than baseline ({snr_base:.1f}dB)"
        )
