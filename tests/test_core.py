"""Tests for tcfp.core — FP8 utilities, block scaling, tensor core matmul."""

from __future__ import annotations

import pytest
import torch

from tcfp.core import (
    DEFAULT_BLOCK_SIZE,
    FP8_E4M3_MAX,
    FP8_E5M2_MAX,
    FP8Config,
    TCFPMode,
    apply_block_scales,
    bits_per_value,
    compute_block_scales,
    fp8_matmul,
    reverse_block_scales,
    to_fp8_e4m3,
    to_fp8_e5m2,
)

# All tests require CUDA
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)

DEVICE = "cuda"


# ── FP8 E4M3 Casting ─────────────────────────────────────────────────────


class TestFP8E4M3:
    def test_roundtrip_small_values(self) -> None:
        """Values within E4M3 range survive round-trip."""
        x = torch.randn(128, device=DEVICE) * 10
        fp8, inv_scale = to_fp8_e4m3(x)
        assert fp8.dtype == torch.float8_e4m3fn
        reconstructed = fp8.float() * inv_scale
        # E4M3 has 3 mantissa bits → ~12.5% relative error max
        assert torch.allclose(x, reconstructed, atol=0.0, rtol=0.15)

    def test_scale_computation(self) -> None:
        """Auto-scale should map max(|x|) near FP8_E4M3_MAX."""
        x = torch.randn(1024, device=DEVICE) * 100
        fp8, inv_scale = to_fp8_e4m3(x)
        # The largest FP8 value should be close to FP8_E4M3_MAX
        max_fp8 = fp8.float().abs().max().item()
        assert max_fp8 <= FP8_E4M3_MAX
        # Should use most of the range
        assert max_fp8 >= FP8_E4M3_MAX * 0.5

    def test_custom_scale(self) -> None:
        """Providing an explicit scale works correctly."""
        x = torch.ones(64, device=DEVICE) * 100.0
        scale = torch.tensor(FP8_E4M3_MAX / 100.0, device=DEVICE)
        fp8, inv_scale = to_fp8_e4m3(x, scale=scale)
        reconstructed = fp8.float() * inv_scale
        assert torch.allclose(x, reconstructed, atol=1.0)

    def test_output_shapes(self) -> None:
        x = torch.randn(32, 64, device=DEVICE)
        fp8, inv_scale = to_fp8_e4m3(x)
        assert fp8.shape == (32, 64)
        assert inv_scale.ndim == 1  # scalar-ish


# ── FP8 E5M2 Casting ─────────────────────────────────────────────────────


class TestFP8E5M2:
    def test_roundtrip(self) -> None:
        x = torch.randn(128, device=DEVICE) * 100
        fp8, inv_scale = to_fp8_e5m2(x)
        assert fp8.dtype == torch.float8_e5m2
        reconstructed = fp8.float() * inv_scale
        # E5M2 has only 2 mantissa bits → larger relative error
        assert torch.allclose(x, reconstructed, atol=0.0, rtol=0.30)

    def test_larger_range(self) -> None:
        """E5M2 supports a larger range than E4M3."""
        x = torch.tensor([1000.0, -1000.0], device=DEVICE)
        fp8, inv_scale = to_fp8_e5m2(x)
        reconstructed = fp8.float() * inv_scale
        assert reconstructed[0] > 0
        assert reconstructed[1] < 0


# ── Block Scale Computation ──────────────────────────────────────────────


class TestBlockScales:
    def test_shape(self) -> None:
        x = torch.randn(16, 128, device=DEVICE)
        scales = compute_block_scales(x, block_size=32)
        assert scales.shape == (16, 4)  # 128 / 32 = 4 blocks

    def test_power_of_two(self) -> None:
        """Block scales must be exact powers of 2."""
        x = torch.randn(4, 64, device=DEVICE)
        scales = compute_block_scales(x, block_size=32)
        # log2 of a power of 2 is an integer
        log2_scales = torch.log2(scales)
        assert torch.allclose(log2_scales, log2_scales.round(), atol=1e-5)

    def test_scaled_values_in_range(self) -> None:
        """After dividing by block scales, values fit in FP8 range."""
        x = torch.randn(8, 64, device=DEVICE) * 1000
        scales = compute_block_scales(x, block_size=32)
        scaled = apply_block_scales(x, scales, block_size=32)
        assert scaled.abs().max() <= FP8_E4M3_MAX * 2  # small margin from power-of-2

    def test_apply_reverse_roundtrip(self) -> None:
        """apply_block_scales → reverse_block_scales is identity."""
        x = torch.randn(8, 64, device=DEVICE)
        scales = compute_block_scales(x, block_size=32)
        scaled = apply_block_scales(x, scales, block_size=32)
        recovered = reverse_block_scales(scaled, scales, block_size=32)
        assert torch.allclose(x, recovered, atol=1e-6)

    def test_different_block_sizes(self) -> None:
        for bs in [16, 32, 64, 128]:
            x = torch.randn(4, 128, device=DEVICE)
            scales = compute_block_scales(x, block_size=bs)
            assert scales.shape == (4, 128 // bs)


# ── FP8 Tensor Core Matmul ──────────────────────────────────────────────


class TestFP8Matmul:
    def test_basic_matmul(self) -> None:
        """FP8 matmul produces reasonable result compared to FP32."""
        M, K, N = 64, 128, 32
        a = torch.randn(M, K, device=DEVICE)
        b = torch.randn(K, N, device=DEVICE)

        # Reference
        ref = a @ b

        # FP8
        a_fp8, sa = to_fp8_e4m3(a)
        b_fp8, sb = to_fp8_e4m3(b)
        result = fp8_matmul(a_fp8, b_fp8, sa, sb)

        # Should be in the right ballpark (FP8 matmul has quantization noise)
        rel_error = (result - ref).norm() / ref.norm()
        assert rel_error < 0.15, f"Relative error too high: {rel_error:.4f}"

    def test_output_shape(self) -> None:
        M, K, N = 32, 64, 16
        a = torch.randn(M, K, device=DEVICE)
        b = torch.randn(K, N, device=DEVICE)
        a_fp8, sa = to_fp8_e4m3(a)
        b_fp8, sb = to_fp8_e4m3(b)
        result = fp8_matmul(a_fp8, b_fp8, sa, sb)
        assert result.shape == (M, N)

    def test_output_dtype(self) -> None:
        M, K, N = 32, 64, 16
        a = torch.randn(M, K, device=DEVICE)
        b = torch.randn(K, N, device=DEVICE)
        a_fp8, sa = to_fp8_e4m3(a)
        b_fp8, sb = to_fp8_e4m3(b)
        result = fp8_matmul(a_fp8, b_fp8, sa, sb, out_dtype=torch.float32)
        assert result.dtype == torch.float32


# ── Config & Utilities ───────────────────────────────────────────────────


class TestConfig:
    def test_valid_block_sizes(self) -> None:
        for bs in [16, 32, 64, 128]:
            cfg = FP8Config(block_size=bs)
            assert cfg.block_size == bs

    def test_invalid_block_size(self) -> None:
        with pytest.raises(ValueError, match="block_size"):
            FP8Config(block_size=48)

    def test_bits_per_value(self) -> None:
        assert bits_per_value(TCFPMode.TCFP8) == 8.25
        assert bits_per_value(TCFPMode.TCFP12) == 12.5
        assert bits_per_value(TCFPMode.TCFP16) == 16.5

    def test_default_config(self) -> None:
        cfg = FP8Config()
        assert cfg.mode == TCFPMode.TCFP8
        assert cfg.block_size == 32
        assert cfg.e4m3_for_weights is True
        assert cfg.e5m2_for_grads is True
