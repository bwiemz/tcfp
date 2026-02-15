"""
TCFP-8: Enhanced Block-Scaled FP8
==================================

Standard FP8 E4M3 values with NormalFloat-aware block scaling and
optional error feedback. Uses FP8 tensor cores at full speed.

Key innovations over naive FP8:
  1. **NormalFloat-aware scaling**: Block scales are chosen to minimise MSE
     for Gaussian-distributed values (which neural network weights are),
     rather than simply using the block maximum.
  2. **Error feedback**: Accumulated quantisation error from previous steps
     is added back before the next quantisation, ensuring long-term
     unbiased rounding.
  3. **Amax EMA**: Exponential moving average of block maxima for stable
     scaling across training steps (like delayed scaling, but simpler).

Storage: 8 bits/value + 8-bit block scale per block → ~8.25 bits/value (B=32)
Compute: Standard FP8 tensor core matmul — zero overhead vs naive FP8
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


# ---------------------------------------------------------------------------
# NormalFloat-aware scaling
# ---------------------------------------------------------------------------


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
    ~0.3% of values exceed 3σ, and clipping them causes minimal MSE.

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
    # Fall back to amax for near-constant blocks (std ≈ 0)
    target_max = torch.where(block_std < 1e-6, block_amax, target_max)

    # Power-of-2 scale
    raw_exp = torch.ceil(torch.log2(target_max / FP8_E4M3_MAX))
    scales = torch.exp2(raw_exp)
    return scales


def compute_amax_scales(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """
    Standard amax-based block scaling (fallback, same as naive FP8).
    """
    return compute_block_scales(tensor, block_size, FP8_E4M3_MAX)


# ---------------------------------------------------------------------------
# Error Feedback State
# ---------------------------------------------------------------------------


class ErrorFeedbackState:
    """
    Per-parameter error feedback buffers for TCFP-8.

    Tracks the quantisation error from each step and adds it back
    to the next quantisation input, ensuring unbiased long-term rounding.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}
        self._amax_ema: dict[str, torch.Tensor] = {}
        self._ema_decay: float = 0.999

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

    def reset(self) -> None:
        self._buffers.clear()
        self._amax_ema.clear()


# ---------------------------------------------------------------------------
# Quantize / Dequantize
# ---------------------------------------------------------------------------


@dataclass
class TCFP8Tensor:
    """
    TCFP-8 quantised representation.

    Attributes:
        data_fp8: FP8 E4M3 tensor (same shape as original).
        block_scales: Per-block power-of-2 scales.
        block_size: Block size used.
        original_shape: Shape of the original tensor.
    """

    data_fp8: torch.Tensor        # float8_e4m3fn
    block_scales: torch.Tensor    # float32, shape (..., D // block_size)
    block_size: int = DEFAULT_BLOCK_SIZE
    original_shape: tuple[int, ...] = field(default_factory=tuple)

    @property
    def bits_per_value(self) -> float:
        return 8.0 + 8.0 / self.block_size

    @property
    def device(self) -> torch.device:
        return self.data_fp8.device


def quantize_tcfp8(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    nf_aware: bool = True,
    sigma_factor: float = 3.0,
    error_state: ErrorFeedbackState | None = None,
    param_name: str = "default",
) -> TCFP8Tensor:
    """
    Quantize a tensor to TCFP-8 format.

    Args:
        tensor: Input tensor, shape ``(..., D)`` where D % block_size == 0.
        block_size: Block size for shared scaling.
        nf_aware: Use NormalFloat-aware scaling (recommended for weights).
        sigma_factor: Sigma multiplier for NF-aware scaling.
        error_state: Optional error feedback state for drift compensation.
        param_name: Parameter name for error feedback tracking.

    Returns:
        TCFP8Tensor containing FP8 data and block scales.
    """
    original_shape = tensor.shape
    t = tensor.float()

    # Apply error feedback if available
    if error_state is not None:
        error_buf = error_state.get_error(param_name, t)
        t = t + error_buf

    # Compute block scales
    if nf_aware:
        scales = compute_nf_aware_scales(t, block_size, sigma_factor)
    else:
        scales = compute_amax_scales(t, block_size)

    # EMA smoothing if error state available
    if error_state is not None:
        scales = error_state.get_amax_ema(param_name + ".scales", scales)

    # Apply block scaling
    *batch_dims, D = t.shape
    num_blocks = D // block_size
    blocked = t.reshape(*batch_dims, num_blocks, block_size)
    scaled = blocked / scales.unsqueeze(-1).clamp(min=1e-45)
    scaled_flat = scaled.reshape(*batch_dims, D)

    # Cast to FP8 E4M3 (clamp to valid range)
    clamped = scaled_flat.clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    data_fp8 = clamped.to(torch.float8_e4m3fn)

    # Compute and store error for feedback
    if error_state is not None:
        reconstructed = dequantize_tcfp8(
            TCFP8Tensor(data_fp8, scales, block_size, original_shape)
        )
        quantisation_error = tensor.float() - reconstructed
        error_state.update_error(param_name, quantisation_error)

    return TCFP8Tensor(
        data_fp8=data_fp8,
        block_scales=scales,
        block_size=block_size,
        original_shape=original_shape,
    )


def dequantize_tcfp8(tcfp8: TCFP8Tensor) -> torch.Tensor:
    """
    Dequantize a TCFP-8 tensor back to float32.

    Args:
        tcfp8: TCFP8Tensor to dequantize.

    Returns:
        Reconstructed float32 tensor.
    """
    # Convert FP8 back to float
    values = tcfp8.data_fp8.float()

    # Reverse block scaling
    *batch_dims, D = values.shape
    num_blocks = D // tcfp8.block_size
    blocked = values.reshape(*batch_dims, num_blocks, tcfp8.block_size)
    reconstructed = blocked * tcfp8.block_scales.unsqueeze(-1)

    return reconstructed.reshape(tcfp8.original_shape)


def fake_quantize_tcfp8(
    tensor: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
    nf_aware: bool = True,
    sigma_factor: float = 3.0,
) -> torch.Tensor:
    """
    Fake-quantize: quantize to TCFP-8 and immediately dequantize.

    This is the training hot path — used in the forward pass to simulate
    quantisation effects while keeping gradients flowing via STE.

    Args:
        tensor: Input tensor.
        block_size: Block size.
        nf_aware: Use NormalFloat-aware scaling.
        sigma_factor: Sigma multiplier for NF-aware scaling.

    Returns:
        Fake-quantized tensor (same shape, float32).
    """
    tcfp8 = quantize_tcfp8(tensor, block_size, nf_aware, sigma_factor)
    return dequantize_tcfp8(tcfp8)
