"""
TCFP-12: FP8 + 4-Bit Residual Hybrid (Enhanced)
=================================================

Each value is represented as an FP8 E4M3 main value plus a 4-bit
residual correction, with shared block scales for both components.

The key insight: after quantizing to FP8, the residual (error) is
typically small and well-structured. A 4-bit correction captures
most of this residual, boosting effective mantissa precision from
3 bits to ~5-6 bits.

Encoding per block of B values:
  1. Compute block scale S (power-of-2)
  2. Scale: x_scaled = x / S
  3. Main: x_hi = fp8_e4m3(x_scaled)              -> 8 bits
  4. Residual: r = x_scaled - dequant(x_hi)
  5. Residual scale: S_r (power-of-2, per block)
  6. Correction: x_lo = int4(r / S_r)             -> 4 bits
  7. Store: [S:8bit][S_r:8bit] + [x_hi:8bit][x_lo:4bit] x B

Reconstruction: x ~ S * (dequant(x_hi) + S_r * (x_lo / 7))

Storage: 12 bits/value + 16/B bits overhead -> ~12.5 bits/value (B=32)
        25% less than BF16, 56% more than FP8

Compute: FP8 tensor core matmul for the main term + cheap scalar
         correction. The correction is O(M*N) adds, not O(M*N*K).

Effective precision: ~5-6 mantissa bits (vs 3 for FP8, 7 for BF16)

Enhanced features (all optional):
  - NF4 residual: NormalFloat-optimised 4-bit codebook for residuals
  - NF-aware scaling: sigma-based block scales for Gaussian weights
  - Stochastic rounding: unbiased rounding for training convergence
  - Error feedback: per-parameter drift compensation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from tcfp.core import (
    DEFAULT_BLOCK_SIZE,
    FP8_E4M3_MAX,
    FP8_E4M3_MIN,
    ErrorFeedbackState,
    compute_block_scales,
    compute_nf_aware_scales,
    stochastic_round_to_fp8_e4m3,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 4-bit residual: signed integer in [-7, +7] (15 levels + zero)
RESIDUAL_4BIT_MAX: int = 7
RESIDUAL_4BIT_LEVELS: int = 15  # 2^4 - 1


# ---------------------------------------------------------------------------
# NF4 Codebook (NormalFloat 4-bit)
# ---------------------------------------------------------------------------


def build_nf4_codebook() -> torch.Tensor:
    """
    Build a 16-level NormalFloat codebook for 4-bit residual quantisation.

    Levels are placed at quantiles of N(0,1) using the inverse error
    function: ``erfinv((2*i + 1) / (2*N))`` for i in 0..N-1,
    then normalized to [-1, 1].

    This concentrates more quantisation levels near zero where FP8
    residuals are densest, yielding lower MSE than uniform spacing.

    Returns:
        Sorted tensor of 16 float32 values in [-1, 1].
    """
    n_positive = 8  # 4 bits = 16 levels, 8 positive + 8 negative (symmetric)
    positive_levels = torch.zeros(n_positive, dtype=torch.float64)
    for i in range(n_positive):
        # Quantile positions for the positive half of N(0,1)
        p = (i + 0.5) / n_positive
        q = torch.tensor(p, dtype=torch.float64)
        positive_levels[i] = math.sqrt(2.0) * torch.erfinv(q).item()

    # Normalize so max = 1.0
    positive_levels = positive_levels / positive_levels[-1]

    # Build symmetric codebook: [-levels_reversed, +levels]
    negative_levels = -positive_levels.flip(0)
    codebook = torch.cat([negative_levels, positive_levels])
    return codebook.float()


# Precomputed module-level constants
NF4_CODEBOOK: torch.Tensor = build_nf4_codebook()
NF4_MIDPOINTS: torch.Tensor = (NF4_CODEBOOK[:-1] + NF4_CODEBOOK[1:]) / 2.0

# Number of NF4 levels
NF4_NUM_LEVELS: int = 16


# ---------------------------------------------------------------------------
# TCFP-12 Tensor
# ---------------------------------------------------------------------------


@dataclass
class TCFP12Tensor:
    """
    TCFP-12 quantised representation.

    Attributes:
        main_fp8: FP8 E4M3 tensor (main component).
        residual_int4: int8 tensor — either [-7, +7] linear or [0, 15] NF4 indices.
        main_scales: Per-block scales for the main FP8 values.
        residual_scales: Per-block scales for the 4-bit residuals.
        block_size: Block size used.
        original_shape: Shape of the original tensor.
        nf4_residual: Whether residual uses NF4 codebook encoding.
    """

    main_fp8: torch.Tensor  # float8_e4m3fn
    residual_int4: torch.Tensor  # int8, values in [-7, +7] or [0, 15]
    main_scales: torch.Tensor  # float32
    residual_scales: torch.Tensor  # float32
    block_size: int = DEFAULT_BLOCK_SIZE
    original_shape: tuple[int, ...] = field(default_factory=tuple)
    nf4_residual: bool = False

    @property
    def bits_per_value(self) -> float:
        return 12.0 + 2 * 8.0 / self.block_size

    @property
    def device(self) -> torch.device:
        return self.main_fp8.device


# ---------------------------------------------------------------------------
# Quantize
# ---------------------------------------------------------------------------


def quantize_tcfp12(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    *,
    nf4_residual: bool = False,
    nf_aware_scaling: bool = False,
    sigma_factor: float = 3.0,
    stochastic_rounding: bool = False,
    error_state: ErrorFeedbackState | None = None,
    param_name: str = "default",
) -> TCFP12Tensor:
    """
    Quantize a tensor to TCFP-12 format (FP8 + 4-bit residual).

    Args:
        tensor: Input tensor, shape ``(..., D)`` where D % block_size == 0.
        block_size: Block size for shared scaling.
        nf4_residual: Use NormalFloat-4 codebook for residuals (better for
            normally-distributed residuals).
        nf_aware_scaling: Use sigma-based block scales instead of amax
            (better for Gaussian weights, clips ~0.3% outliers).
        sigma_factor: Number of standard deviations for NF-aware scaling.
        stochastic_rounding: Use stochastic rounding for FP8 cast and
            residual quantisation (unbiased, better for training).
        error_state: Optional error feedback state for drift compensation.
        param_name: Parameter name for error feedback tracking.

    Returns:
        TCFP12Tensor with main FP8 and 4-bit residual components.
    """
    original_shape = tensor.shape
    t = tensor.float()
    *batch_dims, D = t.shape
    assert D % block_size == 0, (
        f"Last dimension ({D}) must be divisible by block_size ({block_size})"
    )
    num_blocks = D // block_size

    # Error feedback: add accumulated error from previous step
    if error_state is not None:
        error_buf = error_state.get_error(param_name, t)
        t = t + error_buf

    # Step 1: Compute main block scales
    if nf_aware_scaling:
        main_scales = compute_nf_aware_scales(t, block_size, sigma_factor)
    else:
        main_scales = compute_block_scales(t, block_size, FP8_E4M3_MAX)

    # EMA smoothing of scales if error feedback is active
    if error_state is not None:
        main_scales = error_state.get_amax_ema(param_name + ".main_scales", main_scales)

    # Step 2: Scale and cast to FP8
    blocked = t.reshape(*batch_dims, num_blocks, block_size)
    scaled = blocked / main_scales.unsqueeze(-1).clamp(min=1e-12)
    scaled_flat = scaled.reshape(*batch_dims, D)
    clamped = scaled_flat.clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)

    if stochastic_rounding:
        main_fp8 = stochastic_round_to_fp8_e4m3(clamped)
    else:
        main_fp8 = clamped.to(torch.float8_e4m3fn)

    # Step 3: Compute residual (in scaled space)
    main_dequant = main_fp8.float()
    residual = scaled_flat - main_dequant

    # Step 4: Compute residual block scales
    residual_blocked = residual.reshape(*batch_dims, num_blocks, block_size)
    residual_amax = residual_blocked.abs().amax(dim=-1).clamp(min=1e-12)

    if nf4_residual:
        # NF4: scale so that amax maps to 1.0 (codebook range is [-1, 1])
        raw_exp = torch.ceil(torch.log2(residual_amax))
        residual_scales = torch.exp2(raw_exp)
    else:
        # Linear int4: scale so that amax maps to 7
        raw_exp = torch.ceil(torch.log2(residual_amax / RESIDUAL_4BIT_MAX))
        residual_scales = torch.exp2(raw_exp)

    # Step 5: Quantize residual
    residual_scaled = residual_blocked / residual_scales.unsqueeze(-1).clamp(min=1e-12)

    if nf4_residual:
        # NF4 codebook quantisation via bucketize
        midpoints = NF4_MIDPOINTS.to(residual_scaled.device)

        if stochastic_rounding:
            # Dither before bucketize: shift by up to half a codebook step
            # Average codebook step ~ 2/16 = 0.125
            dither = (torch.rand_like(residual_scaled) - 0.5) * (2.0 / NF4_NUM_LEVELS)
            residual_dithered = (residual_scaled + dither).clamp(-1.0, 1.0)
            indices = torch.bucketize(residual_dithered, midpoints).to(torch.int8)
        else:
            residual_clamped = residual_scaled.clamp(-1.0, 1.0)
            indices = torch.bucketize(residual_clamped, midpoints).to(torch.int8)

        residual_int_flat = indices.reshape(*batch_dims, D)
    else:
        # Linear int4 quantisation
        if stochastic_rounding:
            dither = torch.rand_like(residual_scaled) - 0.5
            residual_int = (
                (residual_scaled + dither)
                .round()
                .clamp(-RESIDUAL_4BIT_MAX, RESIDUAL_4BIT_MAX)
                .to(torch.int8)
            )
        else:
            residual_int = (
                residual_scaled.round().clamp(-RESIDUAL_4BIT_MAX, RESIDUAL_4BIT_MAX).to(torch.int8)
            )
        residual_int_flat = residual_int.reshape(*batch_dims, D)

    result = TCFP12Tensor(
        main_fp8=main_fp8,
        residual_int4=residual_int_flat,
        main_scales=main_scales,
        residual_scales=residual_scales,
        block_size=block_size,
        original_shape=original_shape,
        nf4_residual=nf4_residual,
    )

    # Error feedback: compute and store error for next step
    if error_state is not None:
        reconstructed = dequantize_tcfp12(result)
        quantisation_error = tensor.float() - reconstructed
        error_state.update_error(param_name, quantisation_error)

    return result


# ---------------------------------------------------------------------------
# Dequantize
# ---------------------------------------------------------------------------


def dequantize_tcfp12(tcfp12: TCFP12Tensor) -> torch.Tensor:
    """
    Dequantize a TCFP-12 tensor back to float32.

    Reconstruction: x ~ main_scale * (fp8_value + residual_scale * residual)

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
    if tcfp12.nf4_residual:
        # NF4: look up codebook values from indices
        codebook = NF4_CODEBOOK.to(tcfp12.device)
        indices = tcfp12.residual_int4.long().reshape(*batch_dims, num_blocks, tcfp12.block_size)
        residual_values_blocked = codebook[indices] * tcfp12.residual_scales.unsqueeze(-1)
    else:
        # Linear int4: direct multiplication
        residual_blocked = tcfp12.residual_int4.float().reshape(
            *batch_dims, num_blocks, tcfp12.block_size
        )
        residual_values_blocked = residual_blocked * tcfp12.residual_scales.unsqueeze(-1)

    residual_flat = residual_values_blocked.reshape(*batch_dims, D)

    # Combine and reverse main scaling
    combined = main_values + residual_flat
    combined_blocked = combined.reshape(*batch_dims, num_blocks, tcfp12.block_size)
    reconstructed = combined_blocked * tcfp12.main_scales.unsqueeze(-1)

    return reconstructed.reshape(tcfp12.original_shape)


# ---------------------------------------------------------------------------
# Fake Quantize
# ---------------------------------------------------------------------------


def fake_quantize_tcfp12(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    *,
    nf4_residual: bool = False,
    nf_aware_scaling: bool = False,
    sigma_factor: float = 3.0,
    stochastic_rounding: bool = False,
) -> torch.Tensor:
    """
    Fake-quantize: quantize to TCFP-12 and immediately dequantize.

    This is the training hot path -- used in the forward pass to simulate
    quantisation effects while keeping gradients flowing via STE.

    Args:
        tensor: Input tensor.
        block_size: Block size.
        nf4_residual: Use NF4 codebook for residuals.
        nf_aware_scaling: Use sigma-based block scales.
        sigma_factor: Sigma multiplier for NF-aware scaling.
        stochastic_rounding: Use stochastic rounding.

    Returns:
        Fake-quantized tensor (same shape, float32).
    """
    tcfp12 = quantize_tcfp12(
        tensor,
        block_size,
        nf4_residual=nf4_residual,
        nf_aware_scaling=nf_aware_scaling,
        sigma_factor=sigma_factor,
        stochastic_rounding=stochastic_rounding,
    )
    return dequantize_tcfp12(tcfp12)
