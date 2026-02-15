"""
TCFP Core — FP8 Utilities and Tensor Core Matmul Interface
============================================================

Provides the fundamental FP8 operations that all three TCFP modes build on:

- FP8 E4M3/E5M2 casting with proper clamping
- ``torch._scaled_mm`` wrapper for FP8 tensor core GEMM
- Block-scale computation (power-of-2)
- Format constants and configuration

All TCFP modes ultimately decompose their matmuls into one or more calls
to ``fp8_matmul``, which dispatches to hardware FP8 tensor cores.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Literal

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# FP8 E4M3: sign(1) + exponent(4) + mantissa(3) = 8 bits
# Range: ±448, precision: 3 mantissa bits
FP8_E4M3_MAX: float = 448.0
FP8_E4M3_MIN: float = -448.0
FP8_E4M3_SMALLEST_NORMAL: float = 2**-6  # 0.015625

# FP8 E5M2: sign(1) + exponent(5) + mantissa(2) = 8 bits
# Range: ±57344, precision: 2 mantissa bits
FP8_E5M2_MAX: float = 57344.0
FP8_E5M2_MIN: float = -57344.0

# Default block size for shared scaling
DEFAULT_BLOCK_SIZE: int = 32
VALID_BLOCK_SIZES: tuple[int, ...] = (16, 32, 64, 128)

# Exponent bias for block scales
SCALE_EXPONENT_BIAS: int = 127


class TCFPMode(Enum):
    """Available TCFP precision modes."""

    TCFP8 = auto()   # Enhanced block-scaled FP8 (~8.25 bits)
    TCFP12 = auto()  # FP8 + 4-bit residual (~12.25 bits)
    TCFP16 = auto()  # Residual FP8 / double FP8 (~16.5 bits)


@dataclass
class FP8Config:
    """
    Configuration for TCFP operations.

    Attributes:
        mode: Precision mode (TCFP8, TCFP12, TCFP16).
        block_size: Number of values sharing a block scale.
        e4m3_for_weights: Use E4M3 for weights (higher precision, lower range).
        e5m2_for_grads: Use E5M2 for gradients (higher range, lower precision).
        error_feedback: Enable error feedback for TCFP-8 mode.
        stochastic_rounding: Use stochastic rounding for gradients.
    """

    mode: TCFPMode = TCFPMode.TCFP8
    block_size: int = DEFAULT_BLOCK_SIZE
    e4m3_for_weights: bool = True
    e5m2_for_grads: bool = True
    error_feedback: bool = True
    stochastic_rounding: bool = False

    def __post_init__(self) -> None:
        if self.block_size not in VALID_BLOCK_SIZES:
            raise ValueError(
                f"block_size must be one of {VALID_BLOCK_SIZES}, got {self.block_size}"
            )


# ---------------------------------------------------------------------------
# FP8 Casting
# ---------------------------------------------------------------------------


def to_fp8_e4m3(
    tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cast a tensor to FP8 E4M3 format with per-tensor or provided scale.

    Args:
        tensor: Input tensor (any float dtype).
        scale: Optional pre-computed scale. If None, computed from tensor max.

    Returns:
        (fp8_tensor, scale): The FP8 tensor and the scale used.
    """
    if scale is None:
        amax = tensor.abs().max().clamp(min=1e-12)
        scale = (FP8_E4M3_MAX / amax).to(torch.float32)

    scaled = (tensor.float() * scale).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    fp8 = scaled.to(torch.float8_e4m3fn)
    inv_scale = torch.ones(1, device=tensor.device, dtype=torch.float32) / scale
    return fp8, inv_scale


def to_fp8_e5m2(
    tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cast a tensor to FP8 E5M2 format with per-tensor or provided scale.

    Args:
        tensor: Input tensor (any float dtype).
        scale: Optional pre-computed scale. If None, computed from tensor max.

    Returns:
        (fp8_tensor, scale): The FP8 tensor and the scale used.
    """
    if scale is None:
        amax = tensor.abs().max().clamp(min=1e-12)
        scale = (FP8_E5M2_MAX / amax).to(torch.float32)

    scaled = (tensor.float() * scale).clamp(FP8_E5M2_MIN, FP8_E5M2_MAX)
    fp8 = scaled.to(torch.float8_e5m2)
    inv_scale = torch.ones(1, device=tensor.device, dtype=torch.float32) / scale
    return fp8, inv_scale


# ---------------------------------------------------------------------------
# Block Scale Computation
# ---------------------------------------------------------------------------


def compute_block_scales(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    fp8_max: float = FP8_E4M3_MAX,
) -> torch.Tensor:
    """
    Compute power-of-2 block scales for a 2D tensor.

    Each block of ``block_size`` contiguous elements along the last dimension
    gets a shared scale = ``2^ceil(log2(amax / fp8_max))``.

    Args:
        tensor: Input tensor of shape ``(..., D)`` where D is divisible by block_size.
        block_size: Block size.
        fp8_max: Maximum representable value in the target FP8 format.

    Returns:
        Block scales of shape ``(..., D // block_size)``.
    """
    *batch_dims, D = tensor.shape
    assert D % block_size == 0, f"D ({D}) must be divisible by block_size ({block_size})"

    num_blocks = D // block_size
    # Reshape to (..., num_blocks, block_size)
    blocked = tensor.reshape(*batch_dims, num_blocks, block_size)
    # Per-block amax
    amax = blocked.abs().amax(dim=-1).clamp(min=1e-12)
    # Power-of-2 scale: 2^ceil(log2(amax / fp8_max))
    # This is the scale to divide by before casting to FP8
    raw_exp = torch.ceil(torch.log2(amax / fp8_max))
    scales = torch.exp2(raw_exp)
    return scales


def apply_block_scales(
    tensor: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """
    Divide a tensor by its block scales (preparing for FP8 casting).

    Args:
        tensor: Input tensor of shape ``(..., D)``.
        scales: Block scales of shape ``(..., D // block_size)``.
        block_size: Block size.

    Returns:
        Scaled tensor of the same shape.
    """
    *batch_dims, D = tensor.shape
    num_blocks = D // block_size
    blocked = tensor.reshape(*batch_dims, num_blocks, block_size)
    # Expand scales to match block elements
    scaled = blocked / scales.unsqueeze(-1)
    return scaled.reshape(*batch_dims, D)


def reverse_block_scales(
    tensor: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """
    Multiply a tensor by its block scales (reconstructing from FP8).

    Args:
        tensor: Dequantized tensor of shape ``(..., D)``.
        scales: Block scales of shape ``(..., D // block_size)``.
        block_size: Block size.

    Returns:
        Reconstructed tensor of the same shape.
    """
    *batch_dims, D = tensor.shape
    num_blocks = D // block_size
    blocked = tensor.reshape(*batch_dims, num_blocks, block_size)
    reconstructed = blocked * scales.unsqueeze(-1)
    return reconstructed.reshape(*batch_dims, D)


# ---------------------------------------------------------------------------
# FP8 Tensor Core Matmul
# ---------------------------------------------------------------------------


def fp8_matmul(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    FP8 matrix multiplication using tensor cores via ``torch._scaled_mm``.

    Computes: ``(a_fp8 / scale_a) @ (b_fp8 / scale_b)``
    where the FP8 multiply-accumulate happens on tensor cores.

    Args:
        a_fp8: Left matrix in FP8 format, shape ``(M, K)``.
        b_fp8: Right matrix in FP8 format, shape ``(K, N)`` — will be transposed internally.
        scale_a: Scale factor for A (scalar or per-row).
        scale_b: Scale factor for B (scalar or per-col).
        out_dtype: Output dtype (default float32).

    Returns:
        Result matrix ``(M, N)`` in out_dtype.
    """
    # _scaled_mm computes a @ b directly (no implicit transpose).
    # cuBLASLt requires: a = row-major (M,K), b = column-major (K,N).
    # Convert b to column-major via t().contiguous().t()
    b_col = b_fp8.t().contiguous().t()
    result = torch._scaled_mm(
        a_fp8,
        b_col,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
    )
    return result


def fp8_matmul_nn(
    a_fp8: torch.Tensor,
    b_fp8_t: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    FP8 matmul where B is already transposed (shape ``(N, K)``).

    Converts B back to column-major (K, N) for ``_scaled_mm``.
    """
    # b_fp8_t is (N, K) row-major; we need (K, N) column-major
    # b_fp8_t.t() gives (K, N) column-major view ← exactly what _scaled_mm wants
    b_col = b_fp8_t.t()
    result = torch._scaled_mm(
        a_fp8,
        b_col,
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=out_dtype,
    )
    return result


# ---------------------------------------------------------------------------
# Utility: bits-per-value calculation
# ---------------------------------------------------------------------------


def bits_per_value(mode: TCFPMode, block_size: int = DEFAULT_BLOCK_SIZE) -> float:
    """
    Calculate effective bits per value for a given TCFP mode.

    Args:
        mode: TCFP precision mode.
        block_size: Block size (affects amortised overhead).

    Returns:
        Effective bits per value.
    """
    scale_overhead = 8.0 / block_size  # 8-bit block scale amortised

    if mode == TCFPMode.TCFP8:
        # 8 bits per value + shared block scale
        return 8.0 + scale_overhead
    elif mode == TCFPMode.TCFP12:
        # 8 bits FP8 + 4 bits residual + shared block scale (two scales)
        return 8.0 + 4.0 + 2 * scale_overhead
    elif mode == TCFPMode.TCFP16:
        # 8 bits hi + 8 bits lo + two block scales
        return 16.0 + 2 * scale_overhead
    else:
        raise ValueError(f"Unknown mode: {mode}")
