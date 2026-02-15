"""
TCFP-12: FP8 + 4-Bit Residual Hybrid
======================================

Each value is represented as an FP8 E4M3 main value plus a 4-bit
residual correction, with shared block scales for both components.

The key insight: after quantizing to FP8, the residual (error) is
typically small and well-structured. A 4-bit correction captures
most of this residual, boosting effective mantissa precision from
3 bits to ~5-6 bits.

Encoding per block of B values:
  1. Compute block scale S (power-of-2)
  2. Scale: x_scaled = x / S
  3. Main: x_hi = fp8_e4m3(x_scaled)              → 8 bits
  4. Residual: r = x_scaled - dequant(x_hi)
  5. Residual scale: S_r (power-of-2, per block)
  6. Correction: x_lo = int4(r / S_r)             → 4 bits
  7. Store: [S:8bit][S_r:8bit] + [x_hi:8bit][x_lo:4bit] × B

Reconstruction: x ≈ S * (dequant(x_hi) + S_r * (x_lo / 7))

Storage: 12 bits/value + 16/B bits overhead → ~12.5 bits/value (B=32)
        25% less than BF16, 56% more than FP8

Compute: FP8 tensor core matmul for the main term + cheap scalar
         correction. The correction is O(M*N) adds, not O(M*N*K).

Effective precision: ~5-6 mantissa bits (vs 3 for FP8, 7 for BF16)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tcfp.core import (
    DEFAULT_BLOCK_SIZE,
    FP8_E4M3_MAX,
    FP8_E4M3_MIN,
    compute_block_scales,
)


# 4-bit residual: signed integer in [-7, +7] (15 levels + zero)
RESIDUAL_4BIT_MAX: int = 7
RESIDUAL_4BIT_LEVELS: int = 15  # 2^4 - 1


@dataclass
class TCFP12Tensor:
    """
    TCFP-12 quantised representation.

    Attributes:
        main_fp8: FP8 E4M3 tensor (main component).
        residual_int4: int8 tensor with values in [-7, +7] (4-bit residuals).
        main_scales: Per-block scales for the main FP8 values.
        residual_scales: Per-block scales for the 4-bit residuals.
        block_size: Block size used.
        original_shape: Shape of the original tensor.
    """

    main_fp8: torch.Tensor         # float8_e4m3fn
    residual_int4: torch.Tensor    # int8, values in [-7, +7]
    main_scales: torch.Tensor      # float32
    residual_scales: torch.Tensor  # float32
    block_size: int = DEFAULT_BLOCK_SIZE
    original_shape: tuple[int, ...] = field(default_factory=tuple)

    @property
    def bits_per_value(self) -> float:
        return 12.0 + 2 * 8.0 / self.block_size

    @property
    def device(self) -> torch.device:
        return self.main_fp8.device


def quantize_tcfp12(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> TCFP12Tensor:
    """
    Quantize a tensor to TCFP-12 format (FP8 + 4-bit residual).

    Args:
        tensor: Input tensor, shape ``(..., D)`` where D % block_size == 0.
        block_size: Block size for shared scaling.

    Returns:
        TCFP12Tensor with main FP8 and 4-bit residual components.
    """
    original_shape = tensor.shape
    t = tensor.float()
    *batch_dims, D = t.shape
    num_blocks = D // block_size

    # Step 1: Compute main block scales
    main_scales = compute_block_scales(t, block_size, FP8_E4M3_MAX)

    # Step 2: Scale and cast to FP8
    blocked = t.reshape(*batch_dims, num_blocks, block_size)
    scaled = blocked / main_scales.unsqueeze(-1).clamp(min=1e-45)
    scaled_flat = scaled.reshape(*batch_dims, D)
    clamped = scaled_flat.clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    main_fp8 = clamped.to(torch.float8_e4m3fn)

    # Step 3: Compute residual
    main_dequant = main_fp8.float()  # Still in scaled space
    residual = scaled_flat - main_dequant

    # Step 4: Compute residual block scales
    residual_blocked = residual.reshape(*batch_dims, num_blocks, block_size)
    residual_amax = residual_blocked.abs().amax(dim=-1).clamp(min=1e-12)
    # Power-of-2 scale for residual
    raw_exp = torch.ceil(torch.log2(residual_amax / RESIDUAL_4BIT_MAX))
    residual_scales = torch.exp2(raw_exp)

    # Step 5: Quantize residual to 4 bits [-7, +7]
    residual_scaled = residual_blocked / residual_scales.unsqueeze(-1).clamp(min=1e-45)
    residual_int = residual_scaled.round().clamp(-RESIDUAL_4BIT_MAX, RESIDUAL_4BIT_MAX).to(torch.int8)
    residual_int_flat = residual_int.reshape(*batch_dims, D)

    return TCFP12Tensor(
        main_fp8=main_fp8,
        residual_int4=residual_int_flat,
        main_scales=main_scales,
        residual_scales=residual_scales,
        block_size=block_size,
        original_shape=original_shape,
    )


def dequantize_tcfp12(tcfp12: TCFP12Tensor) -> torch.Tensor:
    """
    Dequantize a TCFP-12 tensor back to float32.

    Reconstruction: x ≈ main_scale * (fp8_value + residual_scale * (int4 / 7))

    Args:
        tcfp12: TCFP12Tensor to dequantize.

    Returns:
        Reconstructed float32 tensor.
    """
    *batch_dims, D = tcfp12.main_fp8.shape
    num_blocks = D // tcfp12.block_size

    # Reconstruct main component (in scaled space)
    main_values = tcfp12.main_fp8.float()

    # Reconstruct residual (in scaled space)
    residual_blocked = tcfp12.residual_int4.float().reshape(
        *batch_dims, num_blocks, tcfp12.block_size
    )
    residual_values = residual_blocked * tcfp12.residual_scales.unsqueeze(-1)
    residual_flat = residual_values.reshape(*batch_dims, D)

    # Combine and reverse main scaling
    combined = main_values + residual_flat
    combined_blocked = combined.reshape(*batch_dims, num_blocks, tcfp12.block_size)
    reconstructed = combined_blocked * tcfp12.main_scales.unsqueeze(-1)

    return reconstructed.reshape(tcfp12.original_shape)


def fake_quantize_tcfp12(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """
    Fake-quantize: quantize to TCFP-12 and immediately dequantize.

    Args:
        tensor: Input tensor.
        block_size: Block size.

    Returns:
        Fake-quantized tensor (same shape, float32).
    """
    tcfp12 = quantize_tcfp12(tensor, block_size)
    return dequantize_tcfp12(tcfp12)
