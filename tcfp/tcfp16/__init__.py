"""
TCFP-16: Residual FP8 (Double FP8)
====================================

Each value is represented as two FP8 E4M3 values: a high-order term
and a low-order residual, each with their own block scale.

This is the highest-precision TCFP mode. It achieves ~6-7 effective
mantissa bits (approaching BF16's 7) while using FP8 tensor cores
for all heavy computation.

Encoding per block of B values:
  1. Compute main block scale S_hi (power-of-2)
  2. Scale: x_scaled = x / S_hi
  3. Main: x_hi = fp8_e4m3(x_scaled)                    → 8 bits
  4. Residual: r = x_scaled - dequant(x_hi)
  5. Compute residual block scale S_lo (power-of-2)
  6. Scale residual: r_scaled = r / S_lo
  7. Correction: x_lo = fp8_e4m3(r_scaled)               → 8 bits
  8. Store: [S_hi:8bit][S_lo:8bit] + [x_hi:8bit][x_lo:8bit] × B

Reconstruction: x ≈ S_hi * (dequant(x_hi) + S_lo * dequant(x_lo))

Matmul Strategy (C = A @ B):
  Using linearity: C ≈ T1 + T2 + T3
    T1 = (S_ah * S_bh) * (A_hi @ B_hi)     ← FP8 tensor core
    T2 = (S_ah * S_bl) * (A_hi @ B_lo)     ← FP8 tensor core
    T3 = (S_al * S_bh) * (A_lo @ B_hi)     ← FP8 tensor core
  (T4 = A_lo @ B_lo is negligible and skipped)

Storage: 16 bits/value + 16/B bits overhead → ~16.5 bits/value (B=32)
         Same as BF16 storage, but with FP8 tensor core compute.

Compute: 3× FP8 tensor core matmuls + scalar adds.
         ~1.5× throughput of BF16 matmul on H100/B200 (since 3 FP8 at
         2× speed ≈ 1.5× effective, minus the add overhead).

Effective precision: ~6-7 mantissa bits (vs 3 for FP8, 7 for BF16, 10 for FP32)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tcfp.core import (
    DEFAULT_BLOCK_SIZE,
    FP8_E4M3_MAX,
    FP8_E4M3_MIN,
    compute_block_scales,
    fp8_matmul,
)


@dataclass
class TCFP16Tensor:
    """
    TCFP-16 quantised representation (double FP8).

    Attributes:
        hi_fp8: FP8 E4M3 tensor (main high-order component).
        lo_fp8: FP8 E4M3 tensor (residual low-order component).
        hi_scales: Per-block scales for the high-order component.
        lo_scales: Per-block scales for the low-order component.
        block_size: Block size used.
        original_shape: Shape of the original tensor.
    """

    hi_fp8: torch.Tensor  # float8_e4m3fn, shape (..., D)
    lo_fp8: torch.Tensor  # float8_e4m3fn, shape (..., D)
    hi_scales: torch.Tensor  # float32, shape (..., D // block_size)
    lo_scales: torch.Tensor  # float32, shape (..., D // block_size)
    block_size: int = DEFAULT_BLOCK_SIZE
    original_shape: tuple[int, ...] = field(default_factory=tuple)

    @property
    def bits_per_value(self) -> float:
        return 16.0 + 2 * 8.0 / self.block_size

    @property
    def device(self) -> torch.device:
        return self.hi_fp8.device


def quantize_tcfp16(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> TCFP16Tensor:
    """
    Quantize a tensor to TCFP-16 format (double FP8).

    Args:
        tensor: Input tensor, shape ``(..., D)`` where D % block_size == 0.
        block_size: Block size for shared scaling.

    Returns:
        TCFP16Tensor with hi and lo FP8 components.
    """
    original_shape = tensor.shape
    t = tensor.float()
    *batch_dims, D = t.shape
    num_blocks = D // block_size

    # Step 1: Compute high-order block scales
    hi_scales = compute_block_scales(t, block_size, FP8_E4M3_MAX)

    # Step 2: Scale and cast main component to FP8
    blocked = t.reshape(*batch_dims, num_blocks, block_size)
    hi_scaled = blocked / hi_scales.unsqueeze(-1).clamp(min=1e-45)
    hi_scaled_flat = hi_scaled.reshape(*batch_dims, D)
    hi_clamped = hi_scaled_flat.clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    hi_fp8 = hi_clamped.to(torch.float8_e4m3fn)

    # Step 3: Compute residual in scaled space
    hi_dequant = hi_fp8.float()
    residual = hi_scaled_flat - hi_dequant

    # Step 4: Compute residual block scales
    residual_blocked = residual.reshape(*batch_dims, num_blocks, block_size)
    residual_amax = residual_blocked.abs().amax(dim=-1).clamp(min=1e-12)
    raw_exp = torch.ceil(torch.log2(residual_amax / FP8_E4M3_MAX))
    lo_scales = torch.exp2(raw_exp)

    # Step 5: Scale and cast residual to FP8
    lo_scaled = residual_blocked / lo_scales.unsqueeze(-1).clamp(min=1e-45)
    lo_scaled_flat = lo_scaled.reshape(*batch_dims, D)
    lo_clamped = lo_scaled_flat.clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    lo_fp8 = lo_clamped.to(torch.float8_e4m3fn)

    return TCFP16Tensor(
        hi_fp8=hi_fp8,
        lo_fp8=lo_fp8,
        hi_scales=hi_scales,
        lo_scales=lo_scales,
        block_size=block_size,
        original_shape=original_shape,
    )


def dequantize_tcfp16(tcfp16: TCFP16Tensor) -> torch.Tensor:
    """
    Dequantize a TCFP-16 tensor back to float32.

    Reconstruction: x ≈ S_hi * (dequant(x_hi) + S_lo * dequant(x_lo))

    Args:
        tcfp16: TCFP16Tensor to dequantize.

    Returns:
        Reconstructed float32 tensor.
    """
    *batch_dims, D = tcfp16.hi_fp8.shape
    num_blocks = D // tcfp16.block_size

    # High-order component (in scaled space)
    hi_values = tcfp16.hi_fp8.float()

    # Low-order component (in scaled space)
    lo_values_blocked = tcfp16.lo_fp8.float().reshape(*batch_dims, num_blocks, tcfp16.block_size)
    lo_reconstructed = lo_values_blocked * tcfp16.lo_scales.unsqueeze(-1)
    lo_flat = lo_reconstructed.reshape(*batch_dims, D)

    # Combine in scaled space
    combined = hi_values + lo_flat

    # Reverse main scaling
    combined_blocked = combined.reshape(*batch_dims, num_blocks, tcfp16.block_size)
    result = combined_blocked * tcfp16.hi_scales.unsqueeze(-1)

    return result.reshape(tcfp16.original_shape)


def fake_quantize_tcfp16(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """
    Fake-quantize: quantize to TCFP-16 and immediately dequantize.

    Args:
        tensor: Input tensor.
        block_size: Block size.

    Returns:
        Fake-quantized tensor (same shape, float32).
    """
    tcfp16 = quantize_tcfp16(tensor, block_size)
    return dequantize_tcfp16(tcfp16)


# ---------------------------------------------------------------------------
# TCFP-16 Matmul (3× FP8 Tensor Core)
# ---------------------------------------------------------------------------


def tcfp16_matmul(
    a: TCFP16Tensor,
    b: TCFP16Tensor,
) -> torch.Tensor:
    """
    TCFP-16 matrix multiplication using 3 FP8 tensor core matmuls.

    Computes C ≈ T1 + T2 + T3 where:
      T1 = A_hi @ B_hi  (main term)
      T2 = A_hi @ B_lo  (cross term 1)
      T3 = A_lo @ B_hi  (cross term 2)

    Each term uses FP8 tensor cores with per-tensor scaling.

    Note: This operates on 2D tensors (M, K) @ (K, N).
    Block scales must be pre-applied by converting to per-tensor scales.

    Args:
        a: TCFP16Tensor of shape (M, K).
        b: TCFP16Tensor of shape (K, N).

    Returns:
        Result matrix (M, N) in float32.
    """
    assert a.hi_fp8.ndim == 2 and b.hi_fp8.ndim == 2

    # For tensor core matmul, we need per-tensor scales
    # Our block scales need to be incorporated differently:
    # We dequantize first, then use per-tensor FP8 casting
    # This is the simpler but correct approach

    # Dequantize both to get full-precision, then do the 3-term FP8 matmul
    # with per-tensor scaling for each dequantized component

    M, K = a.hi_fp8.shape
    K2, N = b.hi_fp8.shape
    assert K == K2

    # Dequantize components with their block scales
    a_hi = _dequant_component(a.hi_fp8, a.hi_scales, a.block_size)
    _a_lo = _dequant_component(a.lo_fp8, a.lo_scales, a.block_size)
    # For lo, also need to apply hi_scale since lo is in hi-scaled space
    # Actually lo_scales are relative to the hi-scaled space
    a_lo_full = _dequant_component_nested(a.lo_fp8, a.lo_scales, a.hi_scales, a.block_size)

    b_hi = _dequant_component(b.hi_fp8, b.hi_scales, b.block_size)
    b_lo_full = _dequant_component_nested(b.lo_fp8, b.lo_scales, b.hi_scales, b.block_size)

    # Per-tensor scale and cast for each component
    a_hi_fp8, sa_hi = _to_fp8_per_tensor(a_hi)
    a_lo_fp8, sa_lo = _to_fp8_per_tensor(a_lo_full)
    b_hi_fp8, sb_hi = _to_fp8_per_tensor(b_hi)
    b_lo_fp8, sb_lo = _to_fp8_per_tensor(b_lo_full)

    # T1: A_hi @ B_hi (dominant term)
    t1 = fp8_matmul(a_hi_fp8, b_hi_fp8, sa_hi, sb_hi)

    # T2: A_hi @ B_lo (cross term)
    t2 = fp8_matmul(a_hi_fp8, b_lo_fp8, sa_hi, sb_lo)

    # T3: A_lo @ B_hi (cross term)
    t3 = fp8_matmul(a_lo_fp8, b_hi_fp8, sa_lo, sb_hi)

    # Skip T4 = A_lo @ B_lo (negligible)
    return t1 + t2 + t3


def _dequant_component(
    fp8_data: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Dequantize an FP8 component with block scales."""
    values = fp8_data.float()
    *batch_dims, D = values.shape
    num_blocks = D // block_size
    blocked = values.reshape(*batch_dims, num_blocks, block_size)
    result = blocked * scales.unsqueeze(-1)
    return result.reshape(*batch_dims, D)


def _dequant_component_nested(
    fp8_data: torch.Tensor,
    inner_scales: torch.Tensor,
    outer_scales: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Dequantize with nested scales: outer * (inner * fp8)."""
    values = fp8_data.float()
    *batch_dims, D = values.shape
    num_blocks = D // block_size
    blocked = values.reshape(*batch_dims, num_blocks, block_size)
    # Apply inner scale (residual scale) then outer scale (main scale)
    result = blocked * inner_scales.unsqueeze(-1) * outer_scales.unsqueeze(-1)
    return result.reshape(*batch_dims, D)


def _to_fp8_per_tensor(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cast to FP8 E4M3 with per-tensor scale for tensor core matmul."""
    amax = tensor.abs().max().clamp(min=1e-12)
    scale = (FP8_E4M3_MAX / amax).float()
    scaled = (tensor * scale).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    fp8 = scaled.to(torch.float8_e4m3fn)
    inv_scale = torch.ones(1, device=tensor.device, dtype=torch.float32) / scale
    return fp8, inv_scale
