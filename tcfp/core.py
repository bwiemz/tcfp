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

from dataclasses import dataclass
from enum import Enum, auto

import torch

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

    TCFP12 = auto()  # FP8 + 4-bit residual (~12.25 bits)


@dataclass
class FP8Config:
    """
    Configuration for TCFP operations.

    Attributes:
        mode: Precision mode (TCFP12).
        block_size: Number of values sharing a block scale.
        e4m3_for_weights: Use E4M3 for weights (higher precision, lower range).
        e5m2_for_grads: Use E5M2 for gradients (higher range, lower precision).
        error_feedback: Enable error feedback.
        stochastic_rounding: Use stochastic rounding for gradients.
    """

    mode: TCFPMode = TCFPMode.TCFP12
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


def to_fp8_e4m3_nf_aware(
    tensor: torch.Tensor,
    sigma_factor: float = 3.0,
    stochastic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cast to FP8 E4M3 with per-tensor NF-aware (sigma-based) scaling.

    Uses ``sigma_factor * std(tensor)`` as the effective max instead of
    ``abs().max()``, reducing quantisation noise for Gaussian-distributed
    data at the cost of clipping ~0.3 % of outliers (at 3-sigma).

    Args:
        tensor: Input tensor (any float dtype).
        sigma_factor: Number of standard deviations to cover (default 3.0).
        stochastic: Use stochastic rounding via :func:`stochastic_round_to_fp8_e4m3`.

    Returns:
        ``(fp8_tensor, inv_scale)`` where ``inv_scale`` has shape ``(1,)``
        for direct compatibility with ``torch._scaled_mm``.
    """
    t = tensor.float()
    std = t.std().clamp(min=1e-12)
    # Fall back to amax for near-constant tensors
    target_max = t.abs().max().clamp(min=1e-12) if std < 1e-6 else sigma_factor * std

    scale = (FP8_E4M3_MAX / target_max).to(torch.float32)
    scaled = (t * scale).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)

    fp8 = stochastic_round_to_fp8_e4m3(scaled) if stochastic else scaled.to(torch.float8_e4m3fn)

    inv_scale = torch.ones(1, device=tensor.device, dtype=torch.float32) / scale
    return fp8, inv_scale


def ensure_column_major(t: torch.Tensor) -> torch.Tensor:
    """
    Ensure a 2-D tensor is in column-major layout for ``torch._scaled_mm``.

    ``_scaled_mm`` requires its second operand (mat2) to have column-major
    strides ``(1, K)``.  If the tensor is already column-major this is a
    no-op; otherwise a contiguous copy is made.

    Args:
        t: 2-D tensor of shape ``(K, N)``.

    Returns:
        Column-major view (or copy) of *t*.
    """
    assert t.ndim == 2, f"ensure_column_major requires 2-D tensor, got {t.ndim}-D"
    if t.stride(0) == 1 and t.stride(1) == t.shape[0]:
        return t
    return t.t().contiguous().t()


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


def compute_nf_aware_scales(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    sigma_factor: float = 3.0,
) -> torch.Tensor:
    """
    Compute block scales optimised for Gaussian-distributed values.

    Instead of scaling to the block maximum (which wastes dynamic range
    on outliers), we scale to ``sigma_factor * std`` of the block.
    Values beyond this range are clipped — but for Gaussian data, only
    ~0.3% of values exceed 3 sigma, and clipping them causes minimal MSE.

    Args:
        tensor: Input tensor, shape ``(..., D)``.
        block_size: Block size.
        sigma_factor: Number of standard deviations to cover (default 3.0).

    Returns:
        Block scales of shape ``(..., D // block_size)``.
    """
    *batch_dims, D = tensor.shape
    num_blocks = D // block_size
    blocked = tensor.reshape(*batch_dims, num_blocks, block_size)

    # Per-block standard deviation
    block_std = blocked.float().std(dim=-1, correction=0).clamp(min=1e-12)
    block_amax = blocked.float().abs().amax(dim=-1).clamp(min=1e-12)
    target_max = sigma_factor * block_std
    # Fall back to amax for near-constant blocks (std ~ 0)
    target_max = torch.where(block_std < 1e-6, block_amax, target_max)

    # Power-of-2 scale
    raw_exp = torch.ceil(torch.log2(target_max / FP8_E4M3_MAX))
    scales = torch.exp2(raw_exp)
    return scales


# ---------------------------------------------------------------------------
# Error Feedback State
# ---------------------------------------------------------------------------


class ErrorFeedbackState:
    """
    Per-parameter error feedback buffers for TCFP quantisation.

    Tracks the quantisation error from each step and adds it back
    to the next quantisation input, ensuring unbiased long-term rounding.
    """

    def __init__(
        self,
        ema_decay: float = 0.999,
        residual_validation_interval: int = 100,
    ) -> None:
        self._buffers: dict[str, torch.Tensor] = {}
        self._amax_ema: dict[str, torch.Tensor] = {}
        self._ema_decay: float = ema_decay
        # Delayed scaling state (Predictive Dual-Track)
        self._delayed_amax: dict[str, torch.Tensor] = {}
        self._residual_ratio: dict[str, torch.Tensor] = {}
        self._step_count: dict[str, int] = {}
        self._residual_validation_interval: int = residual_validation_interval

    def get_error(self, name: str, like: torch.Tensor) -> torch.Tensor:
        if name not in self._buffers or self._buffers[name].shape != like.shape:
            self._buffers[name] = torch.zeros_like(like)
        return self._buffers[name]

    def update_error(self, name: str, error: torch.Tensor) -> None:
        self._buffers[name] = error.detach()

    def get_amax_ema(self, name: str, current_amax: torch.Tensor) -> torch.Tensor:
        """Get smoothed amax via exponential moving average."""
        if name not in self._amax_ema:
            self._amax_ema[name] = current_amax.detach().clone()
        else:
            self._amax_ema[name] = (
                self._ema_decay * self._amax_ema[name]
                + (1 - self._ema_decay) * current_amax.detach()
            )
        return self._amax_ema[name]

    # -- Delayed scaling (Predictive Dual-Track) -------------------------

    def get_delayed_amax(
        self, name: str, current_amax: torch.Tensor,
    ) -> torch.Tensor:
        """Return EMA-smoothed amax for delayed scaling.

        On first call for a given *name*, returns *current_amax* unchanged
        (no delay on step 0).  Subsequent calls return the EMA blend of the
        previous delayed value and the new *current_amax*.
        """
        if name not in self._delayed_amax:
            self._delayed_amax[name] = current_amax.detach().clone()
            return current_amax
        prev = self._delayed_amax[name]
        self._delayed_amax[name] = (
            self._ema_decay * prev
            + (1 - self._ema_decay) * current_amax.detach()
        )
        return self._delayed_amax[name]

    def predict_residual_amax(
        self, name: str, main_amax: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the residual (W_lo) amax from the main (W_hi) amax.

        Uses a tracked EMA of the ``amax_lo / amax_hi`` ratio.  If the
        ratio has not been tracked yet, falls back to the theoretical
        bound for FP8 E4M3 (3 mantissa bits → max relative error ≈ 1/8).
        """
        ratio_key = f"{name}.ratio"
        if ratio_key not in self._residual_ratio:
            return (main_amax / 8.0).clamp(min=1e-12)
        return (main_amax * self._residual_ratio[ratio_key]).clamp(min=1e-12)

    def update_residual_ratio(
        self,
        name: str,
        main_amax: torch.Tensor,
        actual_residual_amax: torch.Tensor,
    ) -> None:
        """Update the EMA of ``amax_lo / amax_hi``."""
        ratio = (
            actual_residual_amax / main_amax.clamp(min=1e-12)
        ).detach()
        ratio_key = f"{name}.ratio"
        if ratio_key not in self._residual_ratio:
            self._residual_ratio[ratio_key] = ratio
        else:
            self._residual_ratio[ratio_key] = (
                self._ema_decay * self._residual_ratio[ratio_key]
                + (1 - self._ema_decay) * ratio
            )

    def should_validate_residual(self, name: str) -> bool:
        """Return ``True`` on the first call and every *interval* steps.

        Used to periodically compute the actual residual amax and
        recalibrate the predictive ratio.  The interval is set via
        ``residual_validation_interval`` in the constructor.
        """
        key = f"{name}.steps"
        count = self._step_count.get(key, 0) + 1
        self._step_count[key] = count
        return count == 1 or count % self._residual_validation_interval == 0

    def reset(self) -> None:
        self._buffers.clear()
        self._amax_ema.clear()
        self._delayed_amax.clear()
        self._residual_ratio.clear()
        self._step_count.clear()


# ---------------------------------------------------------------------------
# Stochastic Rounding
# ---------------------------------------------------------------------------


def stochastic_round_to_fp8_e4m3(tensor: torch.Tensor) -> torch.Tensor:
    """
    Cast to FP8 E4M3 with stochastic rounding.

    For each value, computes the floor and ceil FP8 representations,
    then probabilistically selects between them proportional to proximity.
    This ensures ``E[dequant(quant(x))] = x`` (unbiased).

    Args:
        tensor: Tensor already in FP8 E4M3 representable range (clamped).

    Returns:
        FP8 E4M3 tensor with stochastic rounding applied.
    """
    # Floor cast: truncate toward zero
    lo = tensor.to(torch.float8_e4m3fn)
    lo_f = lo.float()

    # Compute the gap to the next representable FP8 value
    # Use the magnitude-based ULP: for E4M3, ULP = 2^(exponent - 3)
    # Simpler approach: cast (tensor + epsilon) to find the next value
    eps_direction = torch.sign(tensor - lo_f)
    # For values exactly on an FP8 grid point, no rounding needed
    exact_mask = tensor == lo_f

    # Compute next FP8 value in the direction of the residual
    # Add a small nudge in the residual direction, then cast
    nudge = eps_direction * FP8_E4M3_SMALLEST_NORMAL * 0.5
    hi = (lo_f + nudge).to(torch.float8_e4m3fn)
    # If hi == lo (at boundaries), try a larger nudge
    same_mask = (hi.float() == lo_f) & ~exact_mask
    if same_mask.any():
        abs_lo = lo_f.abs().clamp(min=FP8_E4M3_SMALLEST_NORMAL)
        # ULP is approximately value * 2^(-mantissa_bits) = value * 0.125
        larger_nudge = eps_direction * abs_lo * 0.25
        hi_retry = (lo_f + larger_nudge).to(torch.float8_e4m3fn)
        hi = torch.where(same_mask, hi_retry, hi)

    hi_f = hi.float()

    # Probability of rounding up = (value - lo) / (hi - lo)
    gap = (hi_f - lo_f).abs().clamp(min=1e-45)
    frac = ((tensor - lo_f).abs() / gap).clamp(0.0, 1.0)

    # Stochastic selection
    rand = torch.rand_like(frac)
    result = torch.where(exact_mask, lo, torch.where(rand < frac, hi, lo))
    return result


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

    if mode == TCFPMode.TCFP12:
        # 8 bits FP8 + 4 bits residual + shared block scale (two scales)
        return 8.0 + 4.0 + 2 * scale_overhead
    else:
        raise ValueError(f"Unknown mode: {mode}")
