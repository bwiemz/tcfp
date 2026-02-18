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
    to_fp8_e4m3_sr,
    to_fp8_e5m2,
)
from tcfp.tcfp12 import fake_quantize_tcfp12

# Fused Triton kernels (optional — graceful fallback to torch._scaled_mm)
try:
    from tcfp.kernels import (
        block_dequantize_fp8,
        block_quantize_fp8,
        fused_act_quant_dual_gemm_forward,
        fused_block_scaled_dual_gemm_backward_dx,
        fused_dual_gemm_backward_dx,
        fused_dual_gemm_forward,
        fused_dual_quantize_fp8,
        fused_wtquant_block_scaled_dual_gemm_forward,
        is_triton_available,
    )

    _HAS_FUSED_KERNELS = is_triton_available()
except ImportError:
    _HAS_FUSED_KERNELS = False

# cuBLASLt C++ extension (optional — requires CUDA Toolkit)
try:
    from tcfp.cuda_ext import (
        cuda_ext_dual_gemm_forward,
        is_cuda_ext_available,
    )

    _HAS_CUDA_EXT = is_cuda_ext_available()
except ImportError:
    _HAS_CUDA_EXT = False

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
        if mode == TCFPMode.TCFP12:
            return fake_quantize_tcfp12(
                x,
                block_size=block_size,
                nf4_residual=nf4_residual,
                nf_aware_scaling=nf_aware_scaling,
                sigma_factor=sigma_factor,
                stochastic_rounding=stochastic_rounding,
            )
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
    mode: TCFPMode = TCFPMode.TCFP12,
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
    Backward: 2 FP8 GEMMs for dX + 1 for dW = 5 total (or 1+1=4 with ABD)
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
        use_fused_kernel: bool = True,
        use_cuda_ext: bool = True,
        abd: bool = False,
        srr: bool = False,
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
        elif srr:
            w_lo_fp8, w_lo_inv = to_fp8_e4m3_sr(residual)
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
        # 3-tier dispatch: cuBLASLt C++ ext > Triton fused > 2x _scaled_mm
        if _HAS_CUDA_EXT and use_cuda_ext:
            output_2d = cuda_ext_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
                act_fp8, w_hi_fp8, w_lo_fp8,
                act_inv, w_hi_inv, w_lo_inv,
            )
        elif _HAS_FUSED_KERNELS and use_fused_kernel:
            output_2d = fused_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
                act_fp8, w_hi_fp8, w_lo_fp8,
                act_inv, w_hi_inv, w_lo_inv,
            )
        else:
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
        ctx.use_fused_kernel = use_fused_kernel  # pyright: ignore[reportAttributeAccessIssue]
        ctx.abd = abd  # pyright: ignore[reportAttributeAccessIssue]
        if abd:
            # ABD: skip W_lo in backward — don't save it
            saved = [
                input_2d if hp_grad_weight else act_fp8,
                w_hi_fp8,
                w_hi_inv,
                act_inv,
            ]
        else:
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
        None, None, None, None, None, None, None, None, None, None, None,
    ]:
        saved = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
        has_bias: bool = ctx.has_bias  # pyright: ignore[reportAttributeAccessIssue]
        hp_grad_weight: bool = ctx.hp_grad_weight  # pyright: ignore[reportAttributeAccessIssue]
        input_shape: tuple[int, ...] = ctx.input_shape  # pyright: ignore[reportAttributeAccessIssue]
        use_fused: bool = ctx.use_fused_kernel  # pyright: ignore[reportAttributeAccessIssue]
        is_abd: bool = ctx.abd  # pyright: ignore[reportAttributeAccessIssue]
        if is_abd:
            saved_act, w_hi_fp8, w_hi_inv, act_inv = saved[:4]
            w_lo_fp8 = w_lo_inv = None
        else:
            saved_act = saved[0]
            w_hi_fp8, w_lo_fp8, w_hi_inv, w_lo_inv, act_inv = saved[1:6]
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

        # --- 3. dX: ABD uses single GEMM (grad @ W_hi), else dual ---
        # Note: cuBLASLt doesn't support E5M2 as A-type on SM 12.0+
        # (backward gradients are E5M2), so we skip cuBLASLt here.
        if is_abd:
            # ABD: dX = grad @ W_hi only (single GEMM)
            grad_input = torch._scaled_mm(
                grad_fp8,
                ensure_column_major(w_hi_fp8),
                scale_a=grad_inv,
                scale_b=w_hi_inv,
                out_dtype=torch.float32,
                use_fast_accum=False,
            ).reshape(input_shape)
        elif _HAS_FUSED_KERNELS and use_fused:
            # Non-ABD path: w_lo_fp8 / w_lo_inv are always Tensor here
            assert w_lo_fp8 is not None and w_lo_inv is not None
            grad_input = fused_dual_gemm_backward_dx(  # pyright: ignore[reportPossiblyUnboundVariable]
                grad_fp8, w_hi_fp8, w_lo_fp8,
                grad_inv, w_hi_inv, w_lo_inv,
            ).reshape(input_shape)
        else:
            # Non-ABD path: w_lo_fp8 / w_lo_inv are always Tensor here
            assert w_lo_fp8 is not None and w_lo_inv is not None
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
            grad_input = (
                grad_input_hi + grad_input_lo
            ).reshape(input_shape)

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
            None, None, None, None, None, None, None, None, None, None, None,
        )


# ---------------------------------------------------------------------------
# Autograd: TCFP-12 with per-block scaling on tensor cores
# ---------------------------------------------------------------------------


class _TCFP12BlockScaledFunction(torch.autograd.Function):
    """
    TCFP-12 on tensor cores with per-block FP8 scaling.

    Uses custom Triton GEMM kernels (block-scaled) instead of
    ``torch._scaled_mm`` which only supports scalar scales.
    """

    @staticmethod
    def forward(  # pyright: ignore[reportIncompatibleMethodOverride]
        ctx: torch.autograd.function.FunctionCtx,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        scale_block_size: int,
        hp_grad_weight: bool,
        error_state: ErrorFeedbackState | None,
        param_name: str,
        abd: bool = False,
        srr: bool = False,
    ) -> torch.Tensor:
        # --- 1. Get error-feedback buffer (lazy-allocated, FP32) ---
        error_buf: torch.Tensor | None = None
        if error_state is not None:
            # get_error creates buffer with same dtype as `like` — must be FP32
            error_buf = error_state.get_error(param_name, weight.float())

        # --- 2. Flatten input to 2D ---
        input_shape = input.shape
        input_2d = (
            input.reshape(-1, input.shape[-1]) if input.ndim == 3 else input
        )

        # --- 3. Act quantize + block-scaled dual GEMM ---
        # Dispatch strategy:
        #   use_fused_act (small N): quantise W separately, fuse act-quant+GEMM
        #   else (large N):          quantise act separately, fuse W-quant+GEMM
        #                            (eliminates W_hi/W_lo intermediate GMEM traffic)
        N = weight.shape[0]
        num_n_tiles = (N + 127) // 128  # approx (BLOCK_N varies by autotune)
        use_fused_act = num_n_tiles <= 4

        # Declare w_lo_fp8 / w_lo_scales as possibly-None for the ABD path.
        w_lo_fp8: torch.Tensor | None
        w_lo_scales: torch.Tensor | None

        if use_fused_act:
            # Pre-quantise W (separate kernel), then fuse act-quant into GEMM.
            w_hi_fp8, w_lo_fp8, w_hi_scales, w_lo_scales = fused_dual_quantize_fp8(  # pyright: ignore[reportPossiblyUnboundVariable]
                weight,
                block_size=scale_block_size,
                error_buf=error_buf,
                stochastic_lo=srr,
            )
            # error_buf updated in-place by fused_dual_quantize_fp8 —
            # ErrorFeedbackState.get_error returns a reference to the internal
            # buffer, so no explicit update_error() call is needed.

            output_2d, act_fp8, act_scales = fused_act_quant_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
                input_2d, w_hi_fp8, w_lo_fp8,
                w_hi_scales, w_lo_scales,
                block_size=scale_block_size,
                save_side_outputs=not hp_grad_weight,
            )
        else:
            # Quantise act separately, then fuse W-quant into GEMM.
            # fused_wtquant reads W_bf16 once and quantises W_hi / W_lo
            # in-register, eliminating the intermediate FP8 GMEM round-trip.
            act_fp8, act_scales = block_quantize_fp8(  # pyright: ignore[reportPossiblyUnboundVariable]
                input_2d, block_size=scale_block_size,
                fp8_dtype=torch.float8_e4m3fn,
            )
            (
                output_2d, w_hi_fp8, w_lo_fp8, w_hi_scales, w_lo_scales,
            ) = fused_wtquant_block_scaled_dual_gemm_forward(  # pyright: ignore[reportPossiblyUnboundVariable]
                act_fp8, weight, act_scales,
                block_size=scale_block_size,
                error_buf=error_buf,
                stochastic_lo=srr,
                save_w_lo=not abd,
            )
            # error_buf updated in-place by fused_wtquant (same protocol).

        # --- 7. Add bias and reshape ---
        if bias is not None:
            output_2d = output_2d + bias

        output = output_2d.reshape(*input_shape[:-1], output_2d.shape[-1])

        # --- 8. Save for backward ---
        ctx.hp_grad_weight = hp_grad_weight  # pyright: ignore[reportAttributeAccessIssue]
        ctx.input_shape = input_shape  # pyright: ignore[reportAttributeAccessIssue]
        ctx.has_bias = bias is not None  # pyright: ignore[reportAttributeAccessIssue]
        ctx.scale_block_size = scale_block_size  # pyright: ignore[reportAttributeAccessIssue]
        ctx.abd = abd  # pyright: ignore[reportAttributeAccessIssue]
        # act_scales is only needed in backward for the grad_weight path when
        # hp_grad_weight=False (dequant activation before matmul).  When
        # hp_grad_weight=True the BF16 input is used directly, so save a
        # 0-element placeholder to keep backward indexing stable and avoid
        # persisting the (K//bs, M) int8 tensor across the backward pass.
        if hp_grad_weight or act_scales is None:
            act_scales_save = torch.empty(0, device=input.device, dtype=torch.int8)
        else:
            act_scales_save = act_scales
        if abd:
            # ABD: skip W_lo in backward — don't save it
            saved = [
                input_2d if hp_grad_weight else act_fp8,
                w_hi_fp8, w_hi_scales, act_scales_save,
            ]
        else:
            # w_lo_fp8 / w_lo_scales are always real tensors here:
            #  use_fused_act path: fused_dual_quantize_fp8 always returns tensors
            #  wtquant path: save_w_lo=True (since abd=False) → returns tensors
            assert w_lo_fp8 is not None and w_lo_scales is not None
            saved = [
                input_2d if hp_grad_weight else act_fp8,
                w_hi_fp8, w_lo_fp8,
                w_hi_scales, w_lo_scales, act_scales_save,
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
        None, None, None, None, None, None,
    ]:
        saved = ctx.saved_tensors  # pyright: ignore[reportAttributeAccessIssue]
        has_bias: bool = ctx.has_bias  # pyright: ignore[reportAttributeAccessIssue]
        hp_grad_weight: bool = ctx.hp_grad_weight  # pyright: ignore[reportAttributeAccessIssue]
        input_shape: tuple[int, ...] = ctx.input_shape  # pyright: ignore[reportAttributeAccessIssue]
        bs: int = ctx.scale_block_size  # pyright: ignore[reportAttributeAccessIssue]
        is_abd: bool = ctx.abd  # pyright: ignore[reportAttributeAccessIssue]
        if is_abd:
            saved_act, w_hi_fp8, w_hi_scales, act_scales = saved[:4]
            w_lo_fp8 = w_lo_scales = None
        else:
            saved_act = saved[0]
            w_hi_fp8, w_lo_fp8 = saved[1], saved[2]
            w_hi_scales, w_lo_scales, act_scales = saved[3], saved[4], saved[5]

        # --- 1. Flatten grad to 2D ---
        grad_2d = (
            grad_output.reshape(-1, grad_output.shape[-1])
            if grad_output.ndim == 3
            else grad_output
        )

        # --- 2. Block-quantise grad to FP8 E5M2 ---
        grad_fp8, grad_scales = block_quantize_fp8(  # pyright: ignore[reportPossiblyUnboundVariable]
            grad_2d, block_size=bs, fp8_dtype=torch.float8_e5m2,
        )

        # --- 3. dX: ABD uses single GEMM (grad @ W_hi), else dual ---
        if is_abd:
            # ABD: dX = grad @ W_hi only (dequant + single FP32 matmul)
            w_hi_deq = block_dequantize_fp8(w_hi_fp8, w_hi_scales, bs)  # pyright: ignore[reportPossiblyUnboundVariable]
            grad_deq = block_dequantize_fp8(grad_fp8, grad_scales, bs)  # pyright: ignore[reportPossiblyUnboundVariable]
            grad_input = (grad_deq @ w_hi_deq).reshape(input_shape)
        else:
            # Non-ABD path: w_lo_fp8 / w_lo_scales are always Tensor here
            assert w_lo_fp8 is not None and w_lo_scales is not None
            grad_input = fused_block_scaled_dual_gemm_backward_dx(  # pyright: ignore[reportPossiblyUnboundVariable]
                grad_fp8, w_hi_fp8, w_lo_fp8,
                grad_scales, w_hi_scales, w_lo_scales,
                block_size=bs,
            ).reshape(input_shape)

        # --- 4. dW = grad.T @ activation (full precision) ---
        if hp_grad_weight:
            grad_weight = grad_2d.t() @ saved_act
        elif saved_act.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            saved_act_f = block_dequantize_fp8(saved_act, act_scales, bs)  # pyright: ignore[reportPossiblyUnboundVariable]
            grad_weight = grad_2d.t() @ saved_act_f
        else:
            grad_weight = grad_2d.t() @ saved_act

        # --- 5. dBias ---
        grad_bias = grad_2d.sum(dim=0) if has_bias else None

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None, None, None, None, None, None,
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
        use_fused_kernel: Use fused Triton 2-GEMM kernel when available
            (TCFP-12 tensor core path only, requires Triton).
        use_cuda_ext: Use cuBLASLt C++ extension for fused dual GEMM
            (TCFP-12 tensor core path only, requires CUDA Toolkit).
        scale_block_size: Per-block scaling block size (32, 64, or 128).
            ``None`` (default) uses per-tensor scaling. Requires Triton
            and ``use_tensor_cores=True``. Mutually exclusive with
            ``delayed_scaling``.
        abd: Asymmetric Backward Decomposition — drop W_lo from backward
            dgrad (``dX = grad @ W_hi`` instead of ``grad @ W_hi + grad @ W_lo``).
            Saves 1 GEMM per layer per backward pass. Tensor core path only.
        srr: Stochastic Residual Rounding — use stochastic rounding for
            W_lo quantization instead of round-to-nearest.  Eliminates the
            need for error feedback buffers (``E[SR(x)] = x``).  Mutually
            exclusive with ``delayed_scaling``.  Tensor core path only.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: TCFPMode = TCFPMode.TCFP12,
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
        use_fused_kernel: bool = True,
        use_cuda_ext: bool = True,
        scale_block_size: int | None = None,
        abd: bool = False,
        srr: bool = False,
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
        self.use_fused_kernel = use_fused_kernel and _HAS_FUSED_KERNELS
        if use_fused_kernel and not _HAS_FUSED_KERNELS:
            warnings.warn(
                "use_fused_kernel=True requested but Triton is not available. "
                "Falling back to two separate _scaled_mm calls.",
                stacklevel=2,
            )
        self.use_cuda_ext = use_cuda_ext and _HAS_CUDA_EXT
        if use_cuda_ext and not _HAS_CUDA_EXT:
            warnings.warn(
                "use_cuda_ext=True requested but cuBLASLt C++ extension is "
                "not available. Install CUDA Toolkit and set CUDA_HOME.",
                stacklevel=2,
            )
        # Per-block scaling validation
        if scale_block_size is not None:
            if scale_block_size not in (32, 64, 128):
                raise ValueError(
                    f"scale_block_size must be 32, 64, or 128, "
                    f"got {scale_block_size}"
                )
            if in_features % scale_block_size != 0:
                raise ValueError(
                    f"in_features ({in_features}) must be divisible by "
                    f"scale_block_size ({scale_block_size})"
                )
            if out_features % scale_block_size != 0:
                raise ValueError(
                    f"out_features ({out_features}) must be divisible by "
                    f"scale_block_size ({scale_block_size}). Backward pass "
                    f"block-quantizes gradients along the output dimension."
                )
            if not _HAS_FUSED_KERNELS:
                warnings.warn(
                    "scale_block_size requires Triton for block-scaled GEMM "
                    "kernels. Falling back to per-tensor scaling.",
                    stacklevel=2,
                )
                scale_block_size = None
            if not use_tensor_cores:
                warnings.warn(
                    "scale_block_size requires use_tensor_cores=True and is "
                    "ignored without it.",
                    stacklevel=2,
                )
                scale_block_size = None
            if delayed_scaling and scale_block_size is not None:
                warnings.warn(
                    "Per-block scaling (scale_block_size) and delayed_scaling "
                    "are mutually exclusive. Ignoring delayed_scaling.",
                    stacklevel=2,
                )
                delayed_scaling = False
        self.scale_block_size = scale_block_size
        self.delayed_scaling = delayed_scaling
        if abd and not use_tensor_cores:
            warnings.warn(
                "abd=True requires use_tensor_cores=True and is "
                "ignored without it.",
                stacklevel=2,
            )
            abd = False
        self.abd = abd
        if srr and not use_tensor_cores:
            warnings.warn(
                "srr=True requires use_tensor_cores=True and is "
                "ignored without it.",
                stacklevel=2,
            )
            srr = False
        if srr and delayed_scaling:
            warnings.warn(
                "srr=True is mutually exclusive with delayed_scaling=True. "
                "Disabling delayed_scaling.",
                stacklevel=2,
            )
            delayed_scaling = False
            self.delayed_scaling = False
        self.srr = srr

        # Error feedback state for tensor core path
        # SRR eliminates the need for error feedback (E[SR(x)] = x)
        self._error_state: ErrorFeedbackState | None = None
        if use_tensor_cores and error_feedback and not srr:
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
            if self.mode == TCFPMode.TCFP12 and self.scale_block_size is not None:
                result: torch.Tensor = _TCFP12BlockScaledFunction.apply(  # pyright: ignore[reportAssignmentType]
                    x,
                    self.weight,
                    self.bias,
                    self.scale_block_size,
                    self.hp_grad_weight,
                    self._error_state,
                    self._param_name,
                    self.abd,
                    self.srr,
                )
            else:
                result = _TCFP12TensorCoreFunction.apply(  # pyright: ignore[reportAssignmentType]
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
                    self.use_fused_kernel,
                    self.use_cuda_ext,
                    self.abd,
                    self.srr,
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
        if self.use_fused_kernel:
            parts.append("use_fused_kernel=True")
        if self.use_cuda_ext:
            parts.append("use_cuda_ext=True")
        if self.scale_block_size is not None:
            parts.append(f"scale_block_size={self.scale_block_size}")
        if self.abd:
            parts.append("abd=True")
        if self.srr:
            parts.append("srr=True")
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
        mode: TCFPMode = TCFPMode.TCFP12,
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
        mode: TCFPMode = TCFPMode.TCFP12,
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
    mode: TCFPMode = TCFPMode.TCFP12,
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
    use_fused_kernel: bool = True,
    use_cuda_ext: bool = True,
    scale_block_size: int | None = None,
    abd: bool = False,
    srr: bool = False,
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
        use_fused_kernel: Use fused Triton 2-GEMM kernel when available
            (TCFP-12 tensor core path only, requires Triton).
        use_cuda_ext: Use cuBLASLt C++ extension for fused dual GEMM
            (TCFP-12 tensor core path only, requires CUDA Toolkit).
        scale_block_size: Per-block scaling block size (32, 64, or 128).
            ``None`` uses per-tensor scaling. Requires Triton.
        abd: Asymmetric Backward Decomposition — drop W_lo from backward
            dgrad. Saves 1 GEMM per layer per backward pass.
        srr: Stochastic Residual Rounding — use stochastic rounding for
            W_lo quantization. Eliminates error feedback buffers.

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
            use_fused_kernel=use_fused_kernel,
            use_cuda_ext=use_cuda_ext,
            scale_block_size=scale_block_size,
            abd=abd,
            srr=srr,
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
            # Per-block scaling: skip for layers where in_features or
            # out_features isn't divisible by the block size
            layer_sbs = scale_block_size
            if layer_sbs is not None and (
                module.in_features % layer_sbs != 0
                or module.out_features % layer_sbs != 0
            ):
                layer_sbs = None
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
                use_fused_kernel=use_fused_kernel and layer_tc,
                use_cuda_ext=use_cuda_ext and layer_tc,
                scale_block_size=layer_sbs if layer_tc else None,
                abd=abd and layer_tc,
                srr=srr and layer_tc,
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
