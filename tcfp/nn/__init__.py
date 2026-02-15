"""
TCFP Neural Network Modules
=============================

Drop-in replacements for standard PyTorch layers that use TCFP
quantisation in the forward pass with STE (Straight-Through Estimator)
for gradient flow.

These modules store weights in FP32 (master copy) and apply fake-quantise
on every forward pass, then use FP8 tensor cores for the matmul.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcfp.core import (
    DEFAULT_BLOCK_SIZE,
    FP8_E4M3_MAX,
    FP8_E4M3_MIN,
    ErrorFeedbackState,
    TCFPMode,
    ensure_column_major,
    stochastic_round_to_fp8_e4m3,
    to_fp8_e4m3,
    to_fp8_e4m3_nf_aware,
    to_fp8_e5m2,
)
from tcfp.tcfp8 import fake_quantize_tcfp8
from tcfp.tcfp12 import fake_quantize_tcfp12
from tcfp.tcfp16 import fake_quantize_tcfp16

# ---------------------------------------------------------------------------
# Autograd: STE Fake-Quantize
# ---------------------------------------------------------------------------


class _STEFakeQuantize(torch.autograd.Function):
    """Straight-Through Estimator: forward quantizes, backward passes through."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        mode: TCFPMode,
        block_size: int,
        nf4_residual: bool,
        nf_aware_scaling: bool,
        sigma_factor: float,
        stochastic_rounding: bool,
    ) -> torch.Tensor:
        if mode == TCFPMode.TCFP8:
            return fake_quantize_tcfp8(x, block_size=block_size)
        elif mode == TCFPMode.TCFP12:
            return fake_quantize_tcfp12(
                x,
                block_size=block_size,
                nf4_residual=nf4_residual,
                nf_aware_scaling=nf_aware_scaling,
                sigma_factor=sigma_factor,
                stochastic_rounding=stochastic_rounding,
            )
        elif mode == TCFPMode.TCFP16:
            return fake_quantize_tcfp16(x, block_size=block_size)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        # STE: pass gradient through unchanged
        return grad_output, None, None, None, None, None, None


def ste_fake_quantize(
    x: torch.Tensor,
    mode: TCFPMode = TCFPMode.TCFP8,
    block_size: int = DEFAULT_BLOCK_SIZE,
    *,
    nf4_residual: bool = False,
    nf_aware_scaling: bool = False,
    sigma_factor: float = 3.0,
    stochastic_rounding: bool = False,
) -> torch.Tensor:
    """
    Apply fake-quantize with STE gradient.

    The TCFP-12 enhancement kwargs (nf4_residual, nf_aware_scaling,
    sigma_factor, stochastic_rounding) are only used when mode is TCFP12.
    """
    result: torch.Tensor = _STEFakeQuantize.apply(  # pyright: ignore[reportAssignmentType]
        x,
        mode,
        block_size,
        nf4_residual,
        nf_aware_scaling,
        sigma_factor,
        stochastic_rounding,
    )
    return result


# ---------------------------------------------------------------------------
# Autograd: FP8 Tensor Core Linear
# ---------------------------------------------------------------------------


class _FP8LinearFunction(torch.autograd.Function):
    """
    Linear layer using FP8 tensor cores.

    Forward: quantize weight -> FP8 matmul (tensor cores)
    Backward: STE for weight, standard gradient for input
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        mode: TCFPMode,
        block_size: int,
        nf4_residual: bool,
        nf_aware_scaling: bool,
        sigma_factor: float,
        stochastic_rounding: bool,
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight, bias)  # pyright: ignore[reportArgumentType]
        ctx.mode = mode  # pyright: ignore[reportAttributeAccessIssue]
        ctx.block_size = block_size  # pyright: ignore[reportAttributeAccessIssue]
        ctx.nf4_residual = nf4_residual  # pyright: ignore[reportAttributeAccessIssue]
        ctx.nf_aware_scaling = nf_aware_scaling  # pyright: ignore[reportAttributeAccessIssue]
        ctx.sigma_factor = sigma_factor  # pyright: ignore[reportAttributeAccessIssue]
        ctx.stochastic_rounding = stochastic_rounding  # pyright: ignore[reportAttributeAccessIssue]

        # Fake-quantize weight
        w_q = ste_fake_quantize(
            weight,
            mode,
            block_size,
            nf4_residual=nf4_residual,
            nf_aware_scaling=nf_aware_scaling,
            sigma_factor=sigma_factor,
            stochastic_rounding=stochastic_rounding,
        )

        # Standard linear with quantized weight
        output = F.linear(input, w_q, bias)
        return output

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        input, weight, bias = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]

        # Gradient w.r.t. input (use quantized weight for consistency)
        w_q = ste_fake_quantize(
            weight,
            ctx.mode,  # pyright: ignore[reportAttributeAccessIssue]
            ctx.block_size,  # pyright: ignore[reportAttributeAccessIssue]
            nf4_residual=ctx.nf4_residual,  # pyright: ignore[reportAttributeAccessIssue]
            nf_aware_scaling=ctx.nf_aware_scaling,  # pyright: ignore[reportAttributeAccessIssue]
            sigma_factor=ctx.sigma_factor,  # pyright: ignore[reportAttributeAccessIssue]
            stochastic_rounding=ctx.stochastic_rounding,  # pyright: ignore[reportAttributeAccessIssue]
        )
        grad_input = grad_output @ w_q

        # Gradient w.r.t. weight
        if grad_output.ndim == 3:
            # Batched: (B, T, out) -> reshape
            _B, _T, out_dim = grad_output.shape
            grad_weight = grad_output.reshape(-1, out_dim).t() @ input.reshape(-1, input.shape[-1])
        else:
            grad_weight = grad_output.t() @ input

        # Gradient w.r.t. bias
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# ---------------------------------------------------------------------------
# Autograd: FP8 Tensor Core GEMM (real _scaled_mm)
# ---------------------------------------------------------------------------


class _FP8TensorCoreFunction(torch.autograd.Function):
    """
    Linear layer using real FP8 tensor core GEMMs via ``torch._scaled_mm``.

    Forward: quantise both activation + weight to FP8, run FP8 GEMM.
    Backward: quantise grad to FP8 E5M2, two more FP8 GEMMs for dX and dW.
    """

    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        nf_aware_scaling: bool,
        sigma_factor: float,
        stochastic_rounding: bool,
        hp_grad_weight: bool,
        error_state: ErrorFeedbackState | None,
        param_name: str,
    ) -> torch.Tensor:
        # --- 1. Error feedback on weight ---
        w = weight.float()
        if error_state is not None:
            error_buf = error_state.get_error(param_name, w)
            w = w + error_buf

        # --- 2. Quantise weight to FP8 E4M3 ---
        if nf_aware_scaling:
            w_fp8, w_inv = to_fp8_e4m3_nf_aware(w, sigma_factor, stochastic_rounding)
        else:
            w_fp8, w_inv = to_fp8_e4m3(w)

        # --- 3. Error feedback: store quantisation error ---
        if error_state is not None:
            w_dequant = w_fp8.float() * w_inv
            quant_error = weight.float() - w_dequant
            error_state.update_error(param_name, quant_error)

        # --- 4. Flatten input to 2D ---
        input_shape = input.shape
        input_2d = input.reshape(-1, input.shape[-1]) if input.ndim == 3 else input

        # --- 5. Quantise activation to FP8 E4M3 ---
        act_fp8, act_inv = to_fp8_e4m3(input_2d)

        # --- 6. FP8 GEMM: output = act @ weight.T ---
        # w_fp8 is (D_out, D_in) row-major; w_fp8.t() is (D_in, D_out) col-major
        output_2d = torch._scaled_mm(
            act_fp8,
            w_fp8.t(),
            scale_a=act_inv,
            scale_b=w_inv,
            out_dtype=torch.float32,
            use_fast_accum=False,
        )

        # --- 7. Add bias and reshape ---
        if bias is not None:
            output_2d = output_2d + bias

        output = output_2d.reshape(*input_shape[:-1], output_2d.shape[-1])

        # --- 8. Save for backward ---
        ctx.hp_grad_weight = hp_grad_weight  # pyright: ignore[reportAttributeAccessIssue]
        ctx.input_shape = input_shape  # pyright: ignore[reportAttributeAccessIssue]
        ctx.has_bias = bias is not None  # pyright: ignore[reportAttributeAccessIssue]
        saved = [input_2d if hp_grad_weight else act_fp8, w_fp8, w_inv, act_inv]
        if bias is not None:
            saved.append(bias)
        ctx.save_for_backward(*saved)

        return output

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        None, None, None, None, None, None,
    ]:
        saved = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
        saved_act, w_fp8, w_inv, act_inv = saved[:4]
        has_bias: bool = ctx.has_bias  # pyright: ignore[reportAttributeAccessIssue]
        hp_grad_weight: bool = ctx.hp_grad_weight  # pyright: ignore[reportAttributeAccessIssue]
        input_shape: tuple[int, ...] = ctx.input_shape  # pyright: ignore[reportAttributeAccessIssue]

        # --- 1. Flatten grad to 2D ---
        grad_2d = (
            grad_output.reshape(-1, grad_output.shape[-1])
            if grad_output.ndim == 3
            else grad_output
        )

        # --- 2. Quantise grad to FP8 E5M2 (wider range for gradients) ---
        grad_fp8, grad_inv = to_fp8_e5m2(grad_2d)

        # --- 3. dX = grad_output @ weight ---
        # grad_fp8: (M, D_out) row-major, w_fp8: (D_out, D_in) needs col-major
        grad_input_2d = torch._scaled_mm(
            grad_fp8,
            ensure_column_major(w_fp8),
            scale_a=grad_inv,
            scale_b=w_inv,
            out_dtype=torch.float32,
            use_fast_accum=False,
        )

        # Reshape to match input
        grad_input = grad_input_2d.reshape(input_shape)

        # --- 4. dW = grad.T @ activation ---
        M = grad_2d.shape[0]
        if hp_grad_weight:
            # High-precision: FP32 matmul
            grad_weight = grad_2d.t() @ saved_act
        elif M % 16 != 0:
            # _scaled_mm requires inner dim (M) divisible by 16.
            # Fall back to FP32 for small/unaligned batch sizes.
            grad_weight = grad_2d.t() @ (saved_act.float() * act_inv)
        else:
            # FP8 GEMM for dW
            # grad.T: (D_out, M) — need contiguous row-major
            grad_t_fp8 = grad_fp8.t().contiguous()
            # act_fp8: (M, D_in) — need col-major
            grad_weight = torch._scaled_mm(
                grad_t_fp8,
                ensure_column_major(saved_act),
                scale_a=grad_inv,
                scale_b=act_inv,
                out_dtype=torch.float32,
                use_fast_accum=False,
            )

        # --- 5. dBias ---
        grad_bias = grad_2d.sum(dim=0) if has_bias else None

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None, None, None, None, None, None,
        )


# ---------------------------------------------------------------------------
# Autograd: TCFP-12 Tensor Core GEMM (2-GEMM residual decomposition)
# ---------------------------------------------------------------------------


def _delayed_quantise_w_hi(
    w: torch.Tensor,
    state: ErrorFeedbackState,
    name: str,
    nf_aware: bool,
    sigma_factor: float,
    stochastic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantise main weight component with delayed (EMA) scaling.

    Returns ``(w_hi_fp8, w_hi_inv, current_amax_hi)`` where
    *current_amax_hi* is the effective amax used for residual prediction.
    """
    if nf_aware:
        t = w.float()
        std = t.std().clamp(min=1e-12)
        target_max = (
            t.abs().max().clamp(min=1e-12)
            if std < 1e-6
            else sigma_factor * std
        )
        delayed = state.get_delayed_amax(name + ".w_hi", target_max)
        scale = (FP8_E4M3_MAX / delayed).to(torch.float32)
        scaled = (t * scale).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
        fp8 = (
            stochastic_round_to_fp8_e4m3(scaled)
            if stochastic
            else scaled.to(torch.float8_e4m3fn)
        )
        inv = torch.ones(1, device=w.device, dtype=torch.float32) / scale
        return fp8, inv, delayed
    else:
        current = w.abs().max().clamp(min=1e-12)
        delayed = state.get_delayed_amax(name + ".w_hi", current)
        scale = (FP8_E4M3_MAX / delayed).to(torch.float32)
        fp8, inv = to_fp8_e4m3(w, scale=scale)
        return fp8, inv, delayed


def _delayed_quantise_w_lo(
    residual: torch.Tensor,
    state: ErrorFeedbackState,
    name: str,
    main_amax: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantise residual weight with predictive scaling.

    On validation steps (first call + every 100), computes the actual
    residual amax and recalibrates the prediction ratio.  On other steps,
    the residual amax is predicted from the main component's amax —
    eliminating one full amax reduction.
    """
    if state.should_validate_residual(name):
        actual = residual.abs().max().clamp(min=1e-12)
        state.update_residual_ratio(name, main_amax, actual)
        scale = (FP8_E4M3_MAX / actual).to(torch.float32)
    else:
        predicted = state.predict_residual_amax(name, main_amax)
        scale = (FP8_E4M3_MAX / predicted).to(torch.float32)
    return to_fp8_e4m3(residual, scale=scale)


class _TCFP12TensorCoreFunction(torch.autograd.Function):
    """
    TCFP-12 on tensor cores via 2-GEMM residual FP8 decomposition.

    Decomposes weight into two FP8 components: ``W ≈ W_hi + W_lo``, then
    computes ``output = (A @ W_hi) + (A @ W_lo)`` using two FP8 GEMMs.
    This achieves near-BF16 quality at better-than-BF16 speed.

    Forward:  2 FP8 GEMMs (act @ W_hi + act @ W_lo)
    Backward: 2 FP8 GEMMs for dX + 1 for dW = 5 total
    """

    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        nf_aware_scaling: bool,
        sigma_factor: float,
        stochastic_rounding: bool,
        hp_grad_weight: bool,
        error_state: ErrorFeedbackState | None,
        param_name: str,
        delayed_scaling: bool = False,
    ) -> torch.Tensor:
        # --- 1. Error feedback on weight ---
        w = weight.float()
        if error_state is not None:
            error_buf = error_state.get_error(param_name, w)
            w = w + error_buf

        # --- 2. Quantise W_hi (main component) ---
        current_amax_hi: torch.Tensor | None = None
        if delayed_scaling and error_state is not None:
            w_hi_fp8, w_hi_inv, current_amax_hi = _delayed_quantise_w_hi(
                w, error_state, param_name, nf_aware_scaling,
                sigma_factor, stochastic_rounding,
            )
        elif nf_aware_scaling:
            w_hi_fp8, w_hi_inv = to_fp8_e4m3_nf_aware(
                w, sigma_factor, stochastic_rounding
            )
        else:
            w_hi_fp8, w_hi_inv = to_fp8_e4m3(w)

        # --- 3. Compute residual and quantise W_lo ---
        residual = w - w_hi_fp8.float() * w_hi_inv
        if current_amax_hi is not None and error_state is not None:
            w_lo_fp8, w_lo_inv = _delayed_quantise_w_lo(
                residual, error_state, param_name, current_amax_hi,
            )
        else:
            w_lo_fp8, w_lo_inv = to_fp8_e4m3(residual)

        # --- 4. Error feedback: store total quantisation error ---
        if error_state is not None:
            w_reconstructed = (
                w_hi_fp8.float() * w_hi_inv + w_lo_fp8.float() * w_lo_inv
            )
            quant_error = weight.float() - w_reconstructed
            error_state.update_error(param_name, quant_error)

        # --- 5. Flatten input to 2D, quantise activation once ---
        input_shape = input.shape
        input_2d = (
            input.reshape(-1, input.shape[-1]) if input.ndim == 3 else input
        )
        act_fp8, act_inv = to_fp8_e4m3(input_2d)  # always fresh

        # --- 6. Two FP8 GEMMs: output = (act @ W_hi.T) + (act @ W_lo.T) ---
        output_hi = torch._scaled_mm(
            act_fp8,
            w_hi_fp8.t(),
            scale_a=act_inv,
            scale_b=w_hi_inv,
            out_dtype=torch.float32,
            use_fast_accum=False,
        )
        output_lo = torch._scaled_mm(
            act_fp8,
            w_lo_fp8.t(),
            scale_a=act_inv,
            scale_b=w_lo_inv,
            out_dtype=torch.float32,
            use_fast_accum=False,
        )
        output_2d = output_hi + output_lo

        # --- 7. Add bias and reshape ---
        if bias is not None:
            output_2d = output_2d + bias

        output = output_2d.reshape(*input_shape[:-1], output_2d.shape[-1])

        # --- 8. Save for backward ---
        ctx.hp_grad_weight = hp_grad_weight  # pyright: ignore[reportAttributeAccessIssue]
        ctx.input_shape = input_shape  # pyright: ignore[reportAttributeAccessIssue]
        ctx.has_bias = bias is not None  # pyright: ignore[reportAttributeAccessIssue]
        ctx.delayed_scaling = delayed_scaling  # pyright: ignore[reportAttributeAccessIssue]
        ctx.error_state = error_state  # pyright: ignore[reportAttributeAccessIssue]
        ctx.param_name = param_name  # pyright: ignore[reportAttributeAccessIssue]
        saved = [
            input_2d if hp_grad_weight else act_fp8,
            w_hi_fp8,
            w_lo_fp8,
            w_hi_inv,
            w_lo_inv,
            act_inv,
        ]
        if bias is not None:
            saved.append(bias)
        ctx.save_for_backward(*saved)

        return output

    @staticmethod
    def backward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        None, None, None, None, None, None, None,
    ]:
        saved = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
        saved_act = saved[0]
        w_hi_fp8, w_lo_fp8, w_hi_inv, w_lo_inv, act_inv = saved[1:6]
        has_bias: bool = ctx.has_bias  # pyright: ignore[reportAttributeAccessIssue]
        hp_grad_weight: bool = ctx.hp_grad_weight  # pyright: ignore[reportAttributeAccessIssue]
        input_shape: tuple[int, ...] = ctx.input_shape  # pyright: ignore[reportAttributeAccessIssue]
        # --- 1. Flatten grad to 2D ---
        grad_2d = (
            grad_output.reshape(-1, grad_output.shape[-1])
            if grad_output.ndim == 3
            else grad_output
        )

        # --- 2. Quantise grad to FP8 E5M2 (always fresh) ---
        # Gradients change too rapidly during training for EMA-delayed
        # scaling — precision loss from stale scales degrades convergence.
        grad_fp8, grad_inv = to_fp8_e5m2(grad_2d)

        # --- 3. dX = grad @ W_hi + grad @ W_lo (2 GEMMs) ---
        grad_input_hi = torch._scaled_mm(
            grad_fp8,
            ensure_column_major(w_hi_fp8),
            scale_a=grad_inv,
            scale_b=w_hi_inv,
            out_dtype=torch.float32,
            use_fast_accum=False,
        )
        grad_input_lo = torch._scaled_mm(
            grad_fp8,
            ensure_column_major(w_lo_fp8),
            scale_a=grad_inv,
            scale_b=w_lo_inv,
            out_dtype=torch.float32,
            use_fast_accum=False,
        )
        grad_input = (grad_input_hi + grad_input_lo).reshape(input_shape)

        # --- 4. dW = grad.T @ activation (1 GEMM, no residual needed) ---
        M = grad_2d.shape[0]
        if hp_grad_weight:
            grad_weight = grad_2d.t() @ saved_act
        elif M % 16 != 0:
            # _scaled_mm requires inner dim divisible by 16
            grad_weight = grad_2d.t() @ (saved_act.float() * act_inv)
        else:
            grad_t_fp8 = grad_fp8.t().contiguous()
            grad_weight = torch._scaled_mm(
                grad_t_fp8,
                ensure_column_major(saved_act),
                scale_a=grad_inv,
                scale_b=act_inv,
                out_dtype=torch.float32,
                use_fast_accum=False,
            )

        # --- 5. dBias ---
        grad_bias = grad_2d.sum(dim=0) if has_bias else None

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None, None, None, None, None, None, None,
        )


# ---------------------------------------------------------------------------
# TCFPLinear — Drop-in nn.Linear replacement
# ---------------------------------------------------------------------------


class TCFPLinear(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` using TCFP quantisation.

    Stores weights in FP32 (master copy). Forward pass applies either:
    - **Fake-quantise path** (default): STE fake-quantise + ``F.linear``
    - **Tensor core path** (``use_tensor_cores=True``): real FP8 GEMMs
      via ``torch._scaled_mm`` on FP8 tensor cores

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        bias: Whether to include a bias term.
        mode: TCFP precision mode.
        block_size: Block size for quantisation.
        nf4_residual: Use NF4 codebook for residuals (TCFP-12 only).
        nf_aware_scaling: Use sigma-based block scales.
        stochastic_rounding: Use stochastic rounding.
        warmup_steps: Number of initial steps to skip quantisation.
        use_tensor_cores: Use real FP8 tensor core GEMMs.
        hp_grad_weight: Compute weight gradient in FP32 instead of FP8.
        error_feedback: Enable error feedback for tensor core path.
        delayed_scaling: Use predictive dual-track delayed scaling
            (TCFP-12 tensor core path only).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: TCFPMode = TCFPMode.TCFP8,
        block_size: int = DEFAULT_BLOCK_SIZE,
        *,
        nf4_residual: bool = False,
        nf_aware_scaling: bool = False,
        sigma_factor: float = 3.0,
        stochastic_rounding: bool = False,
        warmup_steps: int = 0,
        use_tensor_cores: bool = False,
        hp_grad_weight: bool = False,
        error_feedback: bool = True,
        delayed_scaling: bool = False,
    ) -> None:
        super().__init__()
        if nf4_residual and use_tensor_cores and mode == TCFPMode.TCFP12:
            warnings.warn(
                "nf4_residual is ignored in tensor core mode — NF4 cannot "
                "run on tensor cores. The residual uses FP8 E4M3 instead.",
                stacklevel=2,
            )
        if delayed_scaling and not use_tensor_cores:
            warnings.warn(
                "delayed_scaling requires use_tensor_cores=True and is "
                "ignored without it.",
                stacklevel=2,
            )
        if delayed_scaling and mode != TCFPMode.TCFP12:
            warnings.warn(
                "delayed_scaling is only supported for TCFP12 tensor core "
                "mode and is ignored for other modes.",
                stacklevel=2,
            )
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.block_size = block_size
        self.nf4_residual = nf4_residual
        self.nf_aware_scaling = nf_aware_scaling
        self.sigma_factor = sigma_factor
        self.stochastic_rounding = stochastic_rounding
        self._warmup_steps = warmup_steps
        self._warmup_remaining = warmup_steps
        self.use_tensor_cores = use_tensor_cores
        self.hp_grad_weight = hp_grad_weight
        self.delayed_scaling = delayed_scaling

        # Error feedback state for tensor core path
        self._error_state: ErrorFeedbackState | None = None
        if use_tensor_cores and error_feedback:
            self._error_state = ErrorFeedbackState()
        self._param_name: str = ""

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.shape[1]
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # During warmup, skip quantisation entirely
        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            return F.linear(x, self.weight, self.bias)

        # Tensor core path: real FP8 GEMMs via torch._scaled_mm
        if self.use_tensor_cores:
            if self.mode == TCFPMode.TCFP12:
                result: torch.Tensor = _TCFP12TensorCoreFunction.apply(  # pyright: ignore[reportAssignmentType]
                    x,
                    self.weight,
                    self.bias,
                    self.nf_aware_scaling,
                    self.sigma_factor,
                    self.stochastic_rounding,
                    self.hp_grad_weight,
                    self._error_state,
                    self._param_name,
                    self.delayed_scaling,
                )
            else:
                result = _FP8TensorCoreFunction.apply(  # pyright: ignore[reportAssignmentType]
                    x,
                    self.weight,
                    self.bias,
                    self.nf_aware_scaling,
                    self.sigma_factor,
                    self.stochastic_rounding,
                    self.hp_grad_weight,
                    self._error_state,
                    self._param_name,
                )
            return result

        # Fake-quantise path: STE + F.linear
        result = _FP8LinearFunction.apply(  # pyright: ignore[reportAssignmentType]
            x,
            self.weight,
            self.bias,
            self.mode,
            self.block_size,
            self.nf4_residual,
            self.nf_aware_scaling,
            self.sigma_factor,
            self.stochastic_rounding,
        )
        return result

    def extra_repr(self) -> str:
        parts = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"bias={self.bias is not None}",
            f"mode={self.mode.name}",
            f"block_size={self.block_size}",
        ]
        if self.use_tensor_cores:
            parts.append("use_tensor_cores=True")
        if self.hp_grad_weight:
            parts.append("hp_grad_weight=True")
        if self.nf4_residual:
            parts.append("nf4_residual=True")
        if self.nf_aware_scaling:
            parts.append("nf_aware_scaling=True")
        if self.stochastic_rounding:
            parts.append("stochastic_rounding=True")
        if self.delayed_scaling:
            parts.append("delayed_scaling=True")
        if self._warmup_steps > 0:
            parts.append(f"warmup_steps={self._warmup_steps}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# TCFPLayerNorm — Drop-in nn.LayerNorm replacement
# ---------------------------------------------------------------------------


class TCFPLayerNorm(nn.Module):
    """
    LayerNorm with TCFP-quantised internal weights.

    Note: Output is NOT quantised (LayerNorm output is used as-is
    for attention/MLP stability).
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        mode: TCFPMode = TCFPMode.TCFP8,
        block_size: int = DEFAULT_BLOCK_SIZE,
        *,
        nf4_residual: bool = False,
        nf_aware_scaling: bool = False,
        stochastic_rounding: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.mode = mode
        self.block_size = block_size
        self.nf4_residual = nf4_residual
        self.nf_aware_scaling = nf_aware_scaling
        self.stochastic_rounding = stochastic_rounding

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantise the weight (bias remains in full precision)
        w_q = ste_fake_quantize(
            self.weight.unsqueeze(0),
            self.mode,
            self.block_size,
            nf4_residual=self.nf4_residual,
            nf_aware_scaling=self.nf_aware_scaling,
            stochastic_rounding=self.stochastic_rounding,
        ).squeeze(0)
        return F.layer_norm(x, self.normalized_shape, w_q, self.bias, self.eps)


# ---------------------------------------------------------------------------
# TCFPEmbedding — Drop-in nn.Embedding replacement
# ---------------------------------------------------------------------------


class TCFPEmbedding(nn.Module):
    """
    Embedding with TCFP-quantised weight table.

    Uses lazy quantisation: only the looked-up rows are quantised,
    not the full embedding table.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        mode: TCFPMode = TCFPMode.TCFP8,
        block_size: int = DEFAULT_BLOCK_SIZE,
        *,
        nf4_residual: bool = False,
        nf_aware_scaling: bool = False,
        stochastic_rounding: bool = False,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.mode = mode
        self.block_size = block_size
        self.nf4_residual = nf4_residual
        self.nf_aware_scaling = nf_aware_scaling
        self.stochastic_rounding = stochastic_rounding

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Lazy quantisation: look up first, then quantise only accessed rows
        raw = F.embedding(indices, self.weight, padding_idx=self.padding_idx)
        return ste_fake_quantize(
            raw,
            self.mode,
            self.block_size,
            nf4_residual=self.nf4_residual,
            nf_aware_scaling=self.nf_aware_scaling,
            stochastic_rounding=self.stochastic_rounding,
        )


# ---------------------------------------------------------------------------
# Model Conversion
# ---------------------------------------------------------------------------


def convert_to_tcfp(
    model: nn.Module,
    mode: TCFPMode = TCFPMode.TCFP8,
    block_size: int = DEFAULT_BLOCK_SIZE,
    skip_patterns: tuple[str, ...] = ("head",),
    *,
    nf4_residual: bool = False,
    nf_aware_scaling: bool = False,
    stochastic_rounding: bool = False,
    warmup_steps: int = 0,
    use_tensor_cores: bool = False,
    hp_grad_weight: bool = False,
    error_feedback: bool = True,
    delayed_scaling: bool = False,
    _parent_path: str = "",
) -> nn.Module:
    """
    Convert a model's layers to TCFP equivalents in-place.

    Replaces:
      - nn.Linear -> TCFPLinear
      - nn.LayerNorm -> TCFPLayerNorm
      - nn.Embedding -> TCFPEmbedding

    Args:
        model: The model to convert.
        mode: TCFP precision mode.
        block_size: Block size.
        skip_patterns: Layer name patterns to skip (keep in FP32).
        nf4_residual: Use NF4 codebook for residuals (TCFP-12 only).
        nf_aware_scaling: Use sigma-based block scales.
        stochastic_rounding: Use stochastic rounding.
        warmup_steps: Number of initial steps to skip quantisation.
        use_tensor_cores: Use real FP8 tensor core GEMMs (TCFPLinear only).
        hp_grad_weight: Compute weight gradient in FP32 (tensor core path).
        error_feedback: Enable error feedback (tensor core path).
        delayed_scaling: Use predictive dual-track delayed scaling
            (TCFP-12 tensor core path only).

    Returns:
        The converted model (modified in-place).
    """
    for name, module in model.named_children():
        full_path = f"{_parent_path}.{name}" if _parent_path else name

        # Recurse first
        convert_to_tcfp(
            module,
            mode,
            block_size,
            skip_patterns,
            nf4_residual=nf4_residual,
            nf_aware_scaling=nf_aware_scaling,
            stochastic_rounding=stochastic_rounding,
            warmup_steps=warmup_steps,
            use_tensor_cores=use_tensor_cores,
            hp_grad_weight=hp_grad_weight,
            error_feedback=error_feedback,
            delayed_scaling=delayed_scaling,
            _parent_path=full_path,
        )

        # Skip patterns
        if any(pat in name for pat in skip_patterns):
            continue

        if isinstance(module, nn.Linear):
            if module.in_features % block_size != 0:
                continue
            # _scaled_mm requires both dims divisible by 16
            layer_tc = use_tensor_cores and (
                module.in_features % 16 == 0
                and module.out_features % 16 == 0
            )
            new_module = TCFPLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                mode=mode,
                block_size=block_size,
                nf4_residual=nf4_residual,
                nf_aware_scaling=nf_aware_scaling,
                stochastic_rounding=stochastic_rounding,
                warmup_steps=warmup_steps,
                use_tensor_cores=layer_tc,
                hp_grad_weight=hp_grad_weight,
                error_feedback=error_feedback,
                delayed_scaling=delayed_scaling and layer_tc,
            )
            new_module._param_name = full_path
            new_module.weight = module.weight  # pyright: ignore[reportAttributeAccessIssue]
            if module.bias is not None:
                new_module.bias = module.bias  # pyright: ignore[reportAttributeAccessIssue]
            setattr(model, name, new_module)

        elif isinstance(module, nn.LayerNorm):
            ns = module.normalized_shape
            if isinstance(ns, (list, tuple)) and ns[-1] % block_size != 0:
                continue
            new_module = TCFPLayerNorm(
                tuple(ns) if isinstance(ns, list) else ns,
                eps=module.eps,
                mode=mode,
                block_size=block_size,
                nf4_residual=nf4_residual,
                nf_aware_scaling=nf_aware_scaling,
                stochastic_rounding=stochastic_rounding,
            )
            new_module.weight = module.weight  # pyright: ignore[reportAttributeAccessIssue]
            new_module.bias = module.bias  # pyright: ignore[reportAttributeAccessIssue]
            setattr(model, name, new_module)

        elif isinstance(module, nn.Embedding):
            if module.embedding_dim % block_size != 0:
                continue
            new_module = TCFPEmbedding(
                module.num_embeddings,
                module.embedding_dim,
                padding_idx=module.padding_idx,
                mode=mode,
                block_size=block_size,
                nf4_residual=nf4_residual,
                nf_aware_scaling=nf_aware_scaling,
                stochastic_rounding=stochastic_rounding,
            )
            new_module.weight = module.weight  # pyright: ignore[reportAttributeAccessIssue]
            setattr(model, name, new_module)

    return model
