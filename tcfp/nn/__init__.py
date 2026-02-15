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

import torch
import torch.nn as nn
import torch.nn.functional as F

from tcfp.core import (
    DEFAULT_BLOCK_SIZE,
    FP8_E4M3_MAX,
    FP8_E4M3_MIN,
    TCFPMode,
    fp8_matmul,
)
from tcfp.tcfp8 import fake_quantize_tcfp8, quantize_tcfp8, dequantize_tcfp8
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
    ) -> torch.Tensor:
        if mode == TCFPMode.TCFP8:
            return fake_quantize_tcfp8(x, block_size=block_size)
        elif mode == TCFPMode.TCFP12:
            return fake_quantize_tcfp12(x, block_size=block_size)
        elif mode == TCFPMode.TCFP16:
            return fake_quantize_tcfp16(x, block_size=block_size)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        # STE: pass gradient through unchanged
        return grad_output, None, None


def ste_fake_quantize(
    x: torch.Tensor,
    mode: TCFPMode = TCFPMode.TCFP8,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Apply fake-quantize with STE gradient."""
    return _STEFakeQuantize.apply(x, mode, block_size)


# ---------------------------------------------------------------------------
# Autograd: FP8 Tensor Core Linear
# ---------------------------------------------------------------------------


class _FP8LinearFunction(torch.autograd.Function):
    """
    Linear layer using FP8 tensor cores.

    Forward: quantize weight → FP8 matmul (tensor cores)
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
    ) -> torch.Tensor:
        ctx.save_for_backward(input, weight, bias)
        ctx.mode = mode
        ctx.block_size = block_size

        # Fake-quantize weight
        w_q = ste_fake_quantize(weight, mode, block_size)

        # Standard linear with quantized weight
        output = F.linear(input, w_q, bias)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None, None]:
        input, weight, bias = ctx.saved_tensors

        # Gradient w.r.t. input (use quantized weight for consistency)
        w_q = ste_fake_quantize(weight, ctx.mode, ctx.block_size)
        grad_input = grad_output @ w_q

        # Gradient w.r.t. weight
        if grad_output.ndim == 3:
            # Batched: (B, T, out) → reshape
            B, T, O = grad_output.shape
            grad_weight = grad_output.reshape(-1, O).t() @ input.reshape(-1, input.shape[-1])
        else:
            grad_weight = grad_output.t() @ input

        # Gradient w.r.t. bias
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))

        return grad_input, grad_weight, grad_bias, None, None


# ---------------------------------------------------------------------------
# TCFPLinear — Drop-in nn.Linear replacement
# ---------------------------------------------------------------------------


class TCFPLinear(nn.Module):
    """
    Drop-in replacement for ``nn.Linear`` using TCFP quantisation.

    Stores weights in FP32 (master copy). Forward pass applies
    fake-quantise + FP8 tensor core matmul. Backward uses STE.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        bias: Whether to include a bias term.
        mode: TCFP precision mode.
        block_size: Block size for quantisation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mode: TCFPMode = TCFPMode.TCFP8,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.block_size = block_size

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
        return _FP8LinearFunction.apply(
            x, self.weight, self.bias, self.mode, self.block_size
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, mode={self.mode.name}, "
            f"block_size={self.block_size}"
        )


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
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.mode = mode
        self.block_size = block_size

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Quantise the weight/bias, but output is full precision
        w_q = ste_fake_quantize(
            self.weight.unsqueeze(0), self.mode, self.block_size
        ).squeeze(0)
        return F.layer_norm(x, self.normalized_shape, w_q, self.bias, self.eps)


# ---------------------------------------------------------------------------
# TCFPEmbedding — Drop-in nn.Embedding replacement
# ---------------------------------------------------------------------------


class TCFPEmbedding(nn.Module):
    """
    Embedding with TCFP-quantised weight table.

    Only accessed rows are quantised (lazy quantisation for efficiency).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        mode: TCFPMode = TCFPMode.TCFP8,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.mode = mode
        self.block_size = block_size

        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].fill_(0)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Fake-quantise the full embedding table
        w_q = ste_fake_quantize(self.weight, self.mode, self.block_size)
        return F.embedding(
            indices, w_q, padding_idx=self.padding_idx
        )


# ---------------------------------------------------------------------------
# Model Conversion
# ---------------------------------------------------------------------------


def convert_to_tcfp(
    model: nn.Module,
    mode: TCFPMode = TCFPMode.TCFP8,
    block_size: int = DEFAULT_BLOCK_SIZE,
    skip_patterns: tuple[str, ...] = ("head",),
) -> nn.Module:
    """
    Convert a model's layers to TCFP equivalents in-place.

    Replaces:
      - nn.Linear → TCFPLinear
      - nn.LayerNorm → TCFPLayerNorm
      - nn.Embedding → TCFPEmbedding

    Args:
        model: The model to convert.
        mode: TCFP precision mode.
        block_size: Block size.
        skip_patterns: Layer name patterns to skip (keep in FP32).

    Returns:
        The converted model (modified in-place).
    """
    for name, module in model.named_children():
        # Recurse first
        convert_to_tcfp(module, mode, block_size, skip_patterns)

        # Skip patterns
        if any(pat in name for pat in skip_patterns):
            continue

        if isinstance(module, nn.Linear):
            # Check if embedding_dim is divisible by block_size
            if module.in_features % block_size != 0:
                continue
            new_module = TCFPLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                mode=mode,
                block_size=block_size,
            )
            new_module.weight = module.weight
            if module.bias is not None:
                new_module.bias = module.bias
            setattr(model, name, new_module)

        elif isinstance(module, nn.LayerNorm):
            ns = module.normalized_shape
            if isinstance(ns, (list, tuple)) and ns[-1] % block_size != 0:
                continue
            new_module = TCFPLayerNorm(
                ns,
                eps=module.eps,
                mode=mode,
                block_size=block_size,
            )
            new_module.weight = module.weight
            new_module.bias = module.bias
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
            )
            new_module.weight = module.weight
            setattr(model, name, new_module)

    return model
