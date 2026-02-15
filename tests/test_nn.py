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
    ste_fake_quantize,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"


# ── STE Fake Quantize ────────────────────────────────────────────────────


class TestSTEFakeQuantize:
    def test_forward_changes_values(self) -> None:
        x = torch.randn(4, 64, device=DEVICE, requires_grad=True)
        out = ste_fake_quantize(x, TCFPMode.TCFP8)
        assert not torch.allclose(x, out, atol=1e-7)

    def test_gradient_passes_through(self) -> None:
        """STE should pass gradients through unchanged."""
        x = torch.randn(4, 64, device=DEVICE, requires_grad=True)
        out = ste_fake_quantize(x, TCFPMode.TCFP8)
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
        layer = TCFPLinear(128, 64, mode=TCFPMode.TCFP8).to(DEVICE)
        x = torch.randn(4, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (4, 64)

    def test_batched_forward(self) -> None:
        layer = TCFPLinear(128, 64, mode=TCFPMode.TCFP8).to(DEVICE)
        x = torch.randn(2, 8, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (2, 8, 64)

    def test_backward(self) -> None:
        layer = TCFPLinear(128, 64, mode=TCFPMode.TCFP8).to(DEVICE)
        x = torch.randn(4, 128, device=DEVICE, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        assert layer.weight.grad is not None
        assert layer.weight.grad.shape == (64, 128)
        assert x.grad is not None

    def test_no_bias(self) -> None:
        layer = TCFPLinear(64, 32, bias=False, mode=TCFPMode.TCFP8).to(DEVICE)
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
        layer = TCFPLayerNorm(128, mode=TCFPMode.TCFP8).to(DEVICE)
        x = torch.randn(4, 8, 128, device=DEVICE)
        out = layer(x)
        assert out.shape == (4, 8, 128)

    def test_normalized_output(self) -> None:
        """Output should be approximately normalized along the last dim."""
        layer = TCFPLayerNorm(128, mode=TCFPMode.TCFP8).to(DEVICE)
        x = torch.randn(4, 128, device=DEVICE) * 10 + 5
        out = layer(x)
        # Should be roughly zero-mean, unit-var after normalization
        # (quantized weight ≈ ones, bias = zeros)
        assert out.mean(dim=-1).abs().max() < 1.0
        assert (out.std(dim=-1) - 1.0).abs().max() < 1.0


# ── TCFPEmbedding ────────────────────────────────────────────────────────


class TestTCFPEmbedding:
    def test_forward_shape(self) -> None:
        emb = TCFPEmbedding(1000, 128, mode=TCFPMode.TCFP8).to(DEVICE)
        idx = torch.randint(0, 1000, (4, 16), device=DEVICE)
        out = emb(idx)
        assert out.shape == (4, 16, 128)

    def test_padding_idx(self) -> None:
        emb = TCFPEmbedding(100, 64, padding_idx=0, mode=TCFPMode.TCFP8).to(DEVICE)
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
        convert_to_tcfp(model, mode=TCFPMode.TCFP8, block_size=32)

        # Check types
        assert isinstance(model[0], TCFPEmbedding)
        assert isinstance(model[1], TCFPLinear)
        assert isinstance(model[2], TCFPLayerNorm)
        assert isinstance(model[3], TCFPLinear)

    def test_convert_preserves_weights(self) -> None:
        model = self._make_simple_model().to(DEVICE)
        original_weight = model[1].weight.data.clone()
        convert_to_tcfp(model, mode=TCFPMode.TCFP8)
        assert torch.equal(model[1].weight.data, original_weight)

    def test_skip_patterns(self) -> None:
        model = nn.ModuleDict({
            "encoder": nn.Linear(128, 64),
            "head": nn.Linear(64, 10),
        }).to(DEVICE)
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
        convert_to_tcfp(model, mode=TCFPMode.TCFP8, block_size=32)

        x = torch.randn(4, 128, device=DEVICE)
        out = model(x)
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()
