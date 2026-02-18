"""Tests for tcfp.nn — Drop-in TCFP modules, STE, model conversion."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tcfp.core import TCFPMode
from tcfp.nn import (
    TCFPEmbedding,
    TCFPLayerNorm,
    TCFPLinear,
    convert_to_tcfp,
    diagnose,
    export,
    ste_fake_quantize,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

DEVICE = "cuda"


# ── STE Fake Quantize ────────────────────────────────────────────────────


class TestSTEFakeQuantize:
    def test_forward_changes_values(self) -> None:
        x = torch.randn(4, 64, device=DEVICE, requires_grad=True)
        out = ste_fake_quantize(x, TCFPMode.TCFP12)
        assert not torch.allclose(x, out, atol=1e-7)

    def test_gradient_passes_through(self) -> None:
        """STE should pass gradients through unchanged."""
        x = torch.randn(4, 64, device=DEVICE, requires_grad=True)
        out = ste_fake_quantize(x, TCFPMode.TCFP12)
        loss = out.sum()
        loss.backward()
        # STE: grad should be all ones (from sum)
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))

    def test_all_modes(self) -> None:
        x = torch.randn(4, 64, device=DEVICE, requires_grad=True)
        for mode in TCFPMode:
            out = ste_fake_quantize(x, mode, block_size=32)
            assert out.shape == x.shape


# ── TCFPLinear ────────────────────────────────────────────────────────────


class TestTCFPLinear:
    def test_forward_shape(self) -> None:
        layer = TCFPLinear(128, 64, mode=TCFPMode.TCFP12).to(DEVICE)
        x = torch.randn(4, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (4, 64)

    def test_batched_forward(self) -> None:
        layer = TCFPLinear(128, 64, mode=TCFPMode.TCFP12).to(DEVICE)
        x = torch.randn(2, 8, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (2, 8, 64)

    def test_backward(self) -> None:
        layer = TCFPLinear(128, 64, mode=TCFPMode.TCFP12).to(DEVICE)
        x = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (64, 128)
        assert x.grad is not None

    def test_no_bias(self) -> None:
        layer = TCFPLinear(64, 32, bias=False, mode=TCFPMode.TCFP12).to(DEVICE)
        assert layer.bias is None
        x = torch.randn(4, 64, device=DEVICE)
        out = layer(x)
        assert out.shape == (4, 32)

    def test_all_modes(self) -> None:
        for mode in TCFPMode:
            layer = TCFPLinear(64, 32, mode=mode, block_size=32).to(DEVICE)
            x = torch.randn(2, 64, device=DEVICE)
            out = layer(x)
            assert out.shape == (2, 32)

    def test_extra_repr(self) -> None:
        layer = TCFPLinear(128, 64, mode=TCFPMode.TCFP12)
        rep = layer.extra_repr()
        assert "TCFP12" in rep
        assert "128" in rep
        assert "64" in rep


# ── TCFPLayerNorm ─────────────────────────────────────────────────────────


class TestTCFPLayerNorm:
    def test_forward_shape(self) -> None:
        layer = TCFPLayerNorm(128, mode=TCFPMode.TCFP12).to(DEVICE)
        x = torch.randn(4, 8, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (4, 8, 128)

    def test_normalized_output(self) -> None:
        """Output should be approximately normalized along the last dim."""
        layer = TCFPLayerNorm(128, mode=TCFPMode.TCFP12).to(DEVICE)
        x = torch.randn(4, 128, device=DEVICE) * 10 + 5
        out = layer(x)
        # Should be roughly zero-mean, unit-var after normalization
        # (quantized weight ≈ ones, bias = zeros)
        assert out.mean(dim=-1).abs().max() < 1.0
        assert (out.std(dim=-1) - 1.0).abs().max() < 1.0


# ── TCFPEmbedding ────────────────────────────────────────────────────────


class TestTCFPEmbedding:
    def test_forward_shape(self) -> None:
        emb = TCFPEmbedding(1000, 128, mode=TCFPMode.TCFP12).to(DEVICE)
        idx = torch.randint(0, 1000, (4, 16), device=DEVICE)
        out = emb(idx)
        assert out.shape == (4, 16, 128)

    def test_padding_idx(self) -> None:
        emb = TCFPEmbedding(100, 64, padding_idx=0, mode=TCFPMode.TCFP12).to(DEVICE)
        idx = torch.zeros(4, dtype=torch.long, device=DEVICE)
        out = emb(idx)
        # Padding embedding should be close to zero (may not be exactly zero after quantize)
        assert out.abs().max() < 0.5


# ── Model Conversion ─────────────────────────────────────────────────────


class TestModelConversion:
    def _make_simple_model(self) -> nn.Module:
        return nn.Sequential(
            nn.Embedding(1000, 128),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
        )

    def test_convert_replaces_layers(self) -> None:
        model = self._make_simple_model().to(DEVICE)
        convert_to_tcfp(model, mode=TCFPMode.TCFP12, block_size=32)

        # Check types
        assert isinstance(model[0], TCFPEmbedding)  # pyright: ignore[reportIndexIssue]
        assert isinstance(model[1], TCFPLinear)  # pyright: ignore[reportIndexIssue]
        assert isinstance(model[2], TCFPLayerNorm)  # pyright: ignore[reportIndexIssue]
        assert isinstance(model[3], TCFPLinear)  # pyright: ignore[reportIndexIssue]

    def test_convert_preserves_weights(self) -> None:
        model = self._make_simple_model().to(DEVICE)
        original_weight = model[1].weight.data.clone()  # pyright: ignore[reportIndexIssue]
        convert_to_tcfp(model, mode=TCFPMode.TCFP12)
        assert torch.equal(model[1].weight.data, original_weight)  # pyright: ignore[reportIndexIssue]

    def test_skip_patterns(self) -> None:
        model = nn.ModuleDict(
            {
                "encoder": nn.Linear(128, 64),
                "head": nn.Linear(64, 10),
            }
        ).to(DEVICE)
        convert_to_tcfp(model, skip_patterns=("head",))
        assert isinstance(model["encoder"], TCFPLinear)
        assert isinstance(model["head"], nn.Linear)  # Should NOT be converted

    def test_indivisible_features_skipped(self) -> None:
        """Layers with in_features not divisible by block_size are skipped."""
        model = nn.Sequential(
            nn.Linear(33, 64),  # 33 not divisible by 32
            nn.Linear(64, 32),  # 64 is fine
        ).to(DEVICE)
        convert_to_tcfp(model, block_size=32)
        assert isinstance(model[0], nn.Linear)  # Skipped
        assert isinstance(model[1], TCFPLinear)  # Converted

    def test_converted_model_forward(self) -> None:
        """Converted model should still produce valid outputs."""
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
        ).to(DEVICE)
        convert_to_tcfp(model, mode=TCFPMode.TCFP12, block_size=32)

        x = torch.randn(4, 128, device=DEVICE)
        out = model(x)
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()

    def test_convert_with_enhanced_options(self) -> None:
        """convert_to_tcfp should pass TCFP-12 enhancement kwargs."""
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.Linear(64, 32),
        ).to(DEVICE)
        convert_to_tcfp(
            model,
            mode=TCFPMode.TCFP12,
            nf4_residual=True,
            nf_aware_scaling=True,
            stochastic_rounding=True,
            warmup_steps=5,
        )
        assert isinstance(model[0], TCFPLinear)
        assert model[0].nf4_residual is True
        assert model[0].nf_aware_scaling is True
        assert model[0].stochastic_rounding is True
        assert model[0]._warmup_steps == 5


# ── Enhanced Module Features ────────────────────────────────────────────


class TestWarmup:
    def test_warmup_skips_quantization(self) -> None:
        """During warmup, output should match FP32 linear."""
        layer = TCFPLinear(64, 32, mode=TCFPMode.TCFP12, warmup_steps=3).to(DEVICE)
        ref = nn.Linear(64, 32).to(DEVICE)
        ref.weight = layer.weight
        ref.bias = layer.bias

        x = torch.randn(4, 64, device=DEVICE)
        # First call is warmup — should match FP32
        out_warmup = layer(x)
        out_ref = ref(x)
        assert torch.allclose(out_warmup, out_ref, atol=1e-6), (
            "Warmup output doesn't match FP32 linear"
        )

    def test_warmup_ends(self) -> None:
        """After warmup_steps, quantization should resume."""
        layer = TCFPLinear(64, 32, mode=TCFPMode.TCFP12, warmup_steps=2).to(DEVICE)
        ref = nn.Linear(64, 32).to(DEVICE)
        ref.weight = layer.weight
        ref.bias = layer.bias

        x = torch.randn(4, 64, device=DEVICE)

        # Consume warmup steps
        layer(x)
        layer(x)

        # Third call — quantization should be active
        out_quant = layer(x)
        out_ref = ref(x)
        # Quantized output should differ from FP32
        assert not torch.allclose(out_quant, out_ref, atol=1e-6), (
            "Quantization not active after warmup ended"
        )


class TestLazyEmbedding:
    def test_lazy_embedding_shape(self) -> None:
        """Lazy embedding should produce correct output shape."""
        emb = TCFPEmbedding(1000, 128, mode=TCFPMode.TCFP12).to(DEVICE)
        idx = torch.randint(0, 1000, (4, 16), device=DEVICE)
        out = emb(idx)
        assert out.shape == (4, 16, 128)

    def test_lazy_embedding_gradient(self) -> None:
        """Lazy embedding should support backpropagation."""
        emb = TCFPEmbedding(100, 64, mode=TCFPMode.TCFP12).to(DEVICE)
        idx = torch.randint(0, 100, (4,), device=DEVICE)
        out = emb(idx)
        loss = out.sum()
        loss.backward()
        assert emb.weight.grad is not None


# ---------------------------------------------------------------------------
# Feature 3: torch.compile
# ---------------------------------------------------------------------------


class TestCompile:
    def test_compile_does_not_trace(self) -> None:
        """torch.compile should treat TCFPLinear as opaque (no graph breaks)."""
        layer = TCFPLinear(128, 64, mode=TCFPMode.TCFP12).to(DEVICE)
        compiled = torch.compile(layer, backend="eager")
        x = torch.randn(4, 128, device=DEVICE)
        out = compiled(x)
        assert out.shape == (4, 64)


# ---------------------------------------------------------------------------
# Feature 4: BF16 Fallback
# ---------------------------------------------------------------------------


class TestBF16Fallback:
    def test_bf16_fallback_output(self) -> None:
        """After fallback_to_bf16(), output matches F.linear exactly."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer._param_name = "test"
        x = torch.randn(4, 128, device=DEVICE)

        layer.fallback_to_bf16()
        out = layer(x)
        ref = torch.nn.functional.linear(x, layer.weight, layer.bias)
        assert torch.equal(out, ref)

    def test_restore_tcfp(self) -> None:
        """After restore_tcfp(), quantisation resumes."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer._param_name = "test"
        x = torch.randn(4, 128, device=DEVICE)

        # Go to BF16 and back
        layer.fallback_to_bf16()
        assert layer._bf16_fallback
        layer.restore_tcfp()
        assert not layer._bf16_fallback

        # Forward should now use quantisation (output differs from F.linear)
        out_tcfp = layer(x)
        ref_bf16 = torch.nn.functional.linear(x, layer.weight, layer.bias)
        # They should NOT be exactly equal (quantisation changes output)
        assert not torch.equal(out_tcfp, ref_bf16)

    def test_bf16_fallback_no_ef_mutation(self) -> None:
        """EF buffer untouched during BF16 fallback."""
        layer = TCFPLinear(
            128, 64, use_tensor_cores=True, error_feedback=True,
        ).to(DEVICE)
        layer._param_name = "test"

        # Prime EF
        x0 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        layer(x0).sum().backward()
        ef_before = layer._error_state._buffers["test"].clone()  # pyright: ignore[reportOptionalMemberAccess]

        # BF16 fallback — EF should not change
        layer.fallback_to_bf16()
        x1 = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        layer(x1).sum().backward()

        ef_after = layer._error_state._buffers["test"]  # pyright: ignore[reportOptionalMemberAccess]
        assert torch.equal(ef_before, ef_after)


# ---------------------------------------------------------------------------
# Feature 5: Preflight Diagnostic
# ---------------------------------------------------------------------------


class TestDiagnose:
    def test_diagnose_basic(self) -> None:
        """Simple model: layer counts correct."""
        model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        report = diagnose(model)
        assert report.total_linear_layers == 2
        assert report.total_params == 128 * 64 + 64 * 32

    def test_diagnose_tc_eligibility(self) -> None:
        """Mix of TC-eligible and non-eligible layers."""
        model = nn.Sequential(
            nn.Linear(128, 64),   # TC-eligible (both div by 16)
            nn.Linear(100, 50),   # NOT TC-eligible (100 % 16 != 0)
        )
        report = diagnose(model)
        assert report.tc_eligible_layers == 1
        assert report.fallback_layers == 1

    def test_diagnose_skip_patterns(self) -> None:
        """Layers matching skip_patterns are excluded."""
        model = nn.ModuleDict({
            "encoder": nn.Linear(128, 64),
            "head": nn.Linear(64, 10),
        })
        report = diagnose(model, skip_patterns=("head",))
        assert report.total_linear_layers == 1

    def test_diagnose_vram_estimate(self) -> None:
        """Byte calculations are correct."""
        model = nn.Sequential(nn.Linear(128, 64))
        report = diagnose(model)
        n = 128 * 64
        assert report.vram_fp32_bytes == n * 4
        assert report.vram_fp8_bytes == n * 2  # W_hi + W_lo
        assert report.vram_ef_overhead_bytes == n * 4  # EF buffer
        # Savings: FP32(4B) → FP8(2B) + EF(4B) = net -2B (negative savings)
        # Actually: savings = (4N - (2N + 4N)) / 4N = -50%
        assert report.estimated_savings_pct < 0

    def test_diagnose_snr_without_sample_batch(self) -> None:
        """SNR fields are None when sample_batch is not provided."""
        model = nn.Sequential(nn.Linear(128, 64))
        report = diagnose(model)
        assert report.min_snr_db is None
        assert report.mean_snr_db is None
        for ld in report.layer_details:
            assert ld.snr_db is None

    def test_diagnose_snr_with_sample_batch(self) -> None:
        """SNR is computed when sample_batch is provided."""
        model = nn.Sequential(nn.Linear(128, 64))
        dummy = torch.randn(2, 128)  # content unused, just a gate
        report = diagnose(model, sample_batch=dummy)
        assert report.min_snr_db is not None
        assert report.mean_snr_db is not None
        # TCFP-12 dual decomposition should give decent SNR (>25 dB)
        assert report.min_snr_db > 25.0
        for ld in report.layer_details:
            assert ld.snr_db is not None
            assert ld.snr_db > 25.0

    def test_diagnose_snr_block_scaled(self) -> None:
        """Block-scaled SNR path works when specified."""
        model = nn.Sequential(nn.Linear(128, 64))
        dummy = torch.randn(2, 128)
        # scale_block_size=64 — dims are compatible (128 % 64 == 0, 64 % 64 == 0)
        report = diagnose(model, scale_block_size=64, sample_batch=dummy)
        assert report.min_snr_db is not None
        # Per-tensor path (CPU, no Triton) still produces valid SNR
        assert report.min_snr_db > 20.0

    def test_diagnose_snr_warning_on_low_snr(self) -> None:
        """Warning fires when a layer reports SNR below 30 dB."""
        from unittest.mock import patch

        model = nn.Sequential(nn.Linear(128, 64))
        dummy = torch.randn(2, 128)
        # Mock _compute_snr_db to return a value below the 30 dB threshold
        with patch("tcfp.nn._compute_snr_db", return_value=15.0):
            import warnings as _w
            with _w.catch_warnings(record=True) as caught:
                _w.simplefilter("always")
                diagnose(model, sample_batch=dummy)
        snr_warnings = [w for w in caught if "Low quantization SNR" in str(w.message)]
        assert len(snr_warnings) >= 1

    def test_diagnose_snr_identity_weights(self) -> None:
        """Identity-like weights give high SNR (>40 dB)."""
        layer = nn.Linear(64, 64)
        model = nn.Sequential(layer)
        with torch.no_grad():
            layer.weight.copy_(torch.eye(64))
        dummy = torch.randn(2, 64)
        report = diagnose(model, sample_batch=dummy)
        assert report.min_snr_db is not None
        assert report.min_snr_db > 40.0


# ---------------------------------------------------------------------------
# Feature 6: Inference Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_replaces_modules(self) -> None:
        """After export, no TCFP modules remain."""
        model = nn.Sequential(
            TCFPLinear(128, 64),
            TCFPLayerNorm(64),
            TCFPLinear(64, 32),
        )
        export(model)
        for _name, mod in model.named_modules():
            assert not isinstance(mod, (TCFPLinear, TCFPLayerNorm, TCFPEmbedding))

    def test_export_preserves_weights(self) -> None:
        """Weight tensors are shared (same data_ptr)."""
        layer = TCFPLinear(128, 64)
        w_ptr = layer.weight.data_ptr()
        model = nn.Sequential(layer)
        export(model)
        exported_layer = model[0]
        assert exported_layer.weight.data_ptr() == w_ptr  # pyright: ignore[reportCallIssue]

    def test_export_output_matches(self) -> None:
        """Exported model output matches F.linear."""
        model = nn.Sequential(TCFPLinear(128, 64))
        model.to(DEVICE)
        w = model[0].weight.clone()  # pyright: ignore[reportCallIssue]
        b = model[0].bias.clone() if model[0].bias is not None else None  # pyright: ignore[reportCallIssue]
        export(model)
        x = torch.randn(4, 128, device=DEVICE)
        out = model(x)
        ref = torch.nn.functional.linear(x, w, b)
        assert torch.equal(out, ref)

    def test_export_roundtrip(self) -> None:
        """convert_to_tcfp then export gives same weights as original."""
        model = nn.Sequential(nn.Linear(128, 64), nn.Linear(64, 32))
        w0 = model[0].weight.data_ptr()  # pyright: ignore[reportCallIssue]
        w1 = model[1].weight.data_ptr()  # pyright: ignore[reportCallIssue]
        convert_to_tcfp(model)
        export(model)
        assert model[0].weight.data_ptr() == w0  # pyright: ignore[reportCallIssue]
        assert model[1].weight.data_ptr() == w1  # pyright: ignore[reportCallIssue]
        assert isinstance(model[0], nn.Linear)
        assert not isinstance(model[0], TCFPLinear)
