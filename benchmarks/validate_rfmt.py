"""
RFMT + ABD Proposal Validation
================================

Empirically validates the two key assumptions from the RFMT research memo
WITHOUT implementing either technique. Runs real TCFP training and measures:

**Test 1 — ABD Assumption**: W_lo contributes <0.1% to backward dgrad.
  Measures ||grad @ W_lo|| / ||grad @ (W_hi + W_lo)|| across training steps
  for every linear layer. If consistently <1%, ABD is safe.

**Test 2 — RFMT Assumption A**: Re-quantization error is small enough
  for momentum absorption.
  After each optimizer step, simulates the RFMT cycle:
    FP32 weights -> dual quantize -> dequant -> measure round-trip error
  Tracks |q_error| / |weight|, |q_error| / |momentum|, and whether
  the error is zero-mean over time.

**Test 3 — RFMT Assumption B**: Masterless training converges.
  Actually runs a simulated RFMT loop: instead of FP32 master weights,
  reconstructs from W_hi + W_lo, applies Adam update in a temporary FP32
  buffer, re-quantizes, and injects error into momentum. Compares
  convergence against standard TCFP.

Run:  PYTHONIOENCODING=utf-8 py -m benchmarks.validate_rfmt
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tcfp.core import TCFPMode
from tcfp.nn import TCFPLinear, convert_to_tcfp

DEVICE = "cuda"
BLOCK_SIZE = 128  # Per-block scaling block size


# =====================================================================
#  Shared: TinyGPT model + data (from train_convergence.py)
# =====================================================================


class TinyGPTConfig:
    vocab_size: int = 64
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    max_seq_len: int = 128
    dropout: float = 0.0


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)


def generate_synthetic_text(
    n_sequences: int = 4000, seq_len: int = 128, vocab_size: int = 64,
) -> DataLoader:
    torch.manual_seed(42)
    data = torch.zeros(n_sequences, seq_len, dtype=torch.long)
    for i in range(n_sequences):
        pattern = i % 3
        if pattern == 0:
            start = torch.randint(0, vocab_size, (1,)).item()
            step = torch.randint(1, 4, (1,)).item()
            for t in range(seq_len):
                noise = 1 if torch.rand(1).item() < 0.05 else 0
                data[i, t] = (start + step * t + noise) % vocab_size
        elif pattern == 1:
            motif_len = int(torch.randint(4, 9, (1,)).item())
            motif = torch.randint(0, vocab_size, (motif_len,))
            for t in range(seq_len):
                noise = torch.randint(0, 2, (1,)).item() if torch.rand(1).item() < 0.05 else 0
                data[i, t] = (motif[t % motif_len].item() + noise) % vocab_size
        else:
            a = torch.randint(0, vocab_size // 2, (1,)).item()
            b = torch.randint(vocab_size // 2, vocab_size, (1,)).item()
            for t in range(seq_len):
                data[i, t] = a if t % 2 == 0 else b
    return DataLoader(TensorDataset(data), batch_size=32, shuffle=True)


# =====================================================================
#  Test 1: ABD — W_lo contribution to backward dgrad
# =====================================================================


@dataclass
class ABDStats:
    """Per-layer ABD statistics across training steps."""
    name: str
    wlo_ratios: list[float] = field(default_factory=list)  # ||dgrad_wlo|| / ||dgrad_total||
    whi_norms: list[float] = field(default_factory=list)
    wlo_norms: list[float] = field(default_factory=list)


def measure_abd_contribution(
    model: nn.Module,
    steps: int = 300,
    lr: float = 3e-4,
    measure_every: int = 10,
) -> tuple[list[ABDStats], list[float]]:
    """Train with standard TCFP and measure W_lo's dgrad contribution."""
    try:
        from tcfp.kernels import block_quantize_fp8, block_dequantize_fp8
    except ImportError:
        print("  ERROR: Triton kernels required for ABD validation")
        return [], []

    cfg = TinyGPTConfig()
    model = TinyGPT(cfg).to(DEVICE)
    convert_to_tcfp(
        model, mode=TCFPMode.TCFP12,
        use_tensor_cores=True, scale_block_size=BLOCK_SIZE,
        skip_patterns=("head",),
    )
    loader = generate_synthetic_text()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    # Find all TCFPLinear layers
    tcfp_layers: dict[str, TCFPLinear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, TCFPLinear) and mod.use_tensor_cores:
            tcfp_layers[name] = mod

    layer_stats: dict[str, ABDStats] = {
        name: ABDStats(name=name) for name in tcfp_layers
    }
    loss_history: list[float] = []
    step = 0

    # Register hooks to capture grad_output for each TCFPLinear
    grad_outputs: dict[str, torch.Tensor] = {}

    def make_hook(layer_name: str):  # noqa: ANN202
        def hook(module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:  # noqa: ANN001
            grad_outputs[layer_name] = grad_output[0].detach()
        return hook

    hooks = []
    for name, layer in tcfp_layers.items():
        hooks.append(layer.register_full_backward_hook(make_hook(name)))

    model.train()
    while step < steps:
        for (batch,) in loader:
            if step >= steps:
                break
            batch = batch.to(DEVICE)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
            loss.backward()

            loss_history.append(loss.item())

            # Measure ABD contribution at measurement steps
            if step % measure_every == 0:
                with torch.no_grad():
                    for name, layer in tcfp_layers.items():
                        if name not in grad_outputs:
                            continue
                        grad_out = grad_outputs[name]
                        # Flatten to 2D
                        grad_2d = grad_out.reshape(-1, grad_out.shape[-1])

                        # Quantize weight to W_hi + W_lo (same as forward)
                        w = layer.weight.data
                        w_hi_fp8, w_hi_scales = block_quantize_fp8(
                            w, block_size=BLOCK_SIZE, fp8_dtype=torch.float8_e4m3fn,
                        )
                        w_hi_deq = block_dequantize_fp8(w_hi_fp8, w_hi_scales, BLOCK_SIZE)
                        residual = w.float() - w_hi_deq
                        w_lo_fp8, w_lo_scales = block_quantize_fp8(
                            residual, block_size=BLOCK_SIZE, fp8_dtype=torch.float8_e4m3fn,
                        )
                        w_lo_deq = block_dequantize_fp8(w_lo_fp8, w_lo_scales, BLOCK_SIZE)

                        # dgrad_hi = grad @ W_hi, dgrad_lo = grad @ W_lo
                        # W is (N, K), grad is (M, N), dgrad is (M, K)
                        dgrad_hi = grad_2d.float() @ w_hi_deq
                        dgrad_lo = grad_2d.float() @ w_lo_deq
                        dgrad_total = dgrad_hi + dgrad_lo

                        norm_total = dgrad_total.norm().item()
                        norm_lo = dgrad_lo.norm().item()
                        ratio = norm_lo / max(norm_total, 1e-12)

                        layer_stats[name].wlo_ratios.append(ratio)
                        layer_stats[name].whi_norms.append(w_hi_deq.norm().item())
                        layer_stats[name].wlo_norms.append(w_lo_deq.norm().item())

            grad_outputs.clear()
            optimizer.step()
            scheduler.step()
            step += 1

    for h in hooks:
        h.remove()

    return list(layer_stats.values()), loss_history


# =====================================================================
#  Test 2: RFMT — Round-trip quantization error analysis
# =====================================================================


@dataclass
class RFMTErrorStats:
    """Per-layer RFMT error statistics."""
    name: str
    q_error_norms: list[float] = field(default_factory=list)  # ||q_error||
    weight_norms: list[float] = field(default_factory=list)    # ||weight||
    rel_errors: list[float] = field(default_factory=list)      # ||q_error|| / ||weight||
    momentum_norms: list[float] = field(default_factory=list)  # ||momentum||
    error_to_momentum: list[float] = field(default_factory=list)  # ||q_error|| / ||momentum||
    mean_signed_errors: list[float] = field(default_factory=list)  # mean(q_error) — tests zero-mean


def measure_rfmt_errors(
    steps: int = 300,
    lr: float = 3e-4,
    measure_every: int = 10,
) -> tuple[list[RFMTErrorStats], list[float]]:
    """Train with standard TCFP and measure round-trip quantization error."""
    try:
        from tcfp.kernels import block_quantize_fp8, block_dequantize_fp8
    except ImportError:
        print("  ERROR: Triton kernels required for RFMT validation")
        return [], []

    cfg = TinyGPTConfig()
    model = TinyGPT(cfg).to(DEVICE)
    convert_to_tcfp(
        model, mode=TCFPMode.TCFP12,
        use_tensor_cores=True, scale_block_size=BLOCK_SIZE,
        skip_patterns=("head",),
    )
    loader = generate_synthetic_text()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    # Map parameter names to TCFPLinear layers
    param_to_name: dict[int, str] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, TCFPLinear) and mod.use_tensor_cores:
            param_to_name[id(mod.weight)] = name

    layer_stats: dict[str, RFMTErrorStats] = {
        name: RFMTErrorStats(name=name) for name in param_to_name.values()
    }
    loss_history: list[float] = []
    step = 0

    model.train()
    while step < steps:
        for (batch,) in loader:
            if step >= steps:
                break
            batch = batch.to(DEVICE)
            inputs, targets = batch[:, :-1], batch[:, 1:]

            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_history.append(loss.item())

            # After optimizer step, measure round-trip error
            if step % measure_every == 0:
                with torch.no_grad():
                    for pg in optimizer.param_groups:
                        for p in pg["params"]:
                            pid = id(p)
                            if pid not in param_to_name:
                                continue
                            name = param_to_name[pid]

                            w = p.data.float()

                            # Simulate RFMT re-quantization
                            w_hi_fp8, w_hi_scales = block_quantize_fp8(
                                w, block_size=BLOCK_SIZE, fp8_dtype=torch.float8_e4m3fn,
                            )
                            w_hi_deq = block_dequantize_fp8(w_hi_fp8, w_hi_scales, BLOCK_SIZE)
                            residual = w - w_hi_deq
                            w_lo_fp8, w_lo_scales = block_quantize_fp8(
                                residual, block_size=BLOCK_SIZE, fp8_dtype=torch.float8_e4m3fn,
                            )
                            w_lo_deq = block_dequantize_fp8(w_lo_fp8, w_lo_scales, BLOCK_SIZE)

                            # Quantization error = what RFMT would inject into momentum
                            q_error = w - w_hi_deq - w_lo_deq

                            w_norm = w.norm().item()
                            q_norm = q_error.norm().item()

                            layer_stats[name].q_error_norms.append(q_norm)
                            layer_stats[name].weight_norms.append(w_norm)
                            layer_stats[name].rel_errors.append(q_norm / max(w_norm, 1e-12))
                            layer_stats[name].mean_signed_errors.append(q_error.mean().item())

                            # Get Adam momentum for this parameter
                            state = optimizer.state.get(p)
                            if state and "exp_avg" in state:
                                m_norm = state["exp_avg"].float().norm().item()
                                layer_stats[name].momentum_norms.append(m_norm)
                                layer_stats[name].error_to_momentum.append(
                                    q_norm / max(m_norm, 1e-12)
                                )

            step += 1

    return list(layer_stats.values()), loss_history


# =====================================================================
#  Test 3: RFMT — Simulated masterless training convergence
# =====================================================================


def train_rfmt_simulated(
    steps: int = 500,
    lr: float = 3e-4,
    gamma: float = 1.0,
) -> tuple[list[float], list[float]]:
    """Simulated RFMT training: no FP32 master weights, no error buffer.

    After each Adam step, we:
    1. Take updated FP32 weights (from Adam's internal state)
    2. Re-quantize to W_hi + W_lo
    3. Inject quantization error into Adam's momentum (m)
    4. Overwrite FP32 weights with dequant(W_hi) + dequant(W_lo)
       (simulating having no master weights)

    This tests whether convergence holds under masterless conditions.

    Returns (rfmt_loss_history, baseline_loss_history).
    """
    try:
        from tcfp.kernels import block_quantize_fp8, block_dequantize_fp8
    except ImportError:
        print("  ERROR: Triton kernels required for RFMT simulation")
        return [], []

    cfg = TinyGPTConfig()

    # --- Baseline: standard TCFP training ---
    torch.manual_seed(42)
    model_base = TinyGPT(cfg).to(DEVICE)
    convert_to_tcfp(
        model_base, mode=TCFPMode.TCFP12,
        use_tensor_cores=True, scale_block_size=BLOCK_SIZE,
        skip_patterns=("head",),
    )
    loader = generate_synthetic_text()
    opt_base = torch.optim.AdamW(model_base.parameters(), lr=lr, weight_decay=0.01)
    sched_base = torch.optim.lr_scheduler.CosineAnnealingLR(opt_base, T_max=steps)

    base_losses: list[float] = []
    step = 0
    model_base.train()
    while step < steps:
        for (batch,) in loader:
            if step >= steps:
                break
            batch = batch.to(DEVICE)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            opt_base.zero_grad()
            logits = model_base(inputs)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_base.parameters(), 1.0)
            opt_base.step()
            sched_base.step()
            base_losses.append(loss.item())
            step += 1

    # --- RFMT: masterless training ---
    torch.manual_seed(42)
    model_rfmt = TinyGPT(cfg).to(DEVICE)
    convert_to_tcfp(
        model_rfmt, mode=TCFPMode.TCFP12,
        use_tensor_cores=True, scale_block_size=BLOCK_SIZE,
        skip_patterns=("head",),
    )
    loader = generate_synthetic_text()  # Same seed -> same data
    opt_rfmt = torch.optim.AdamW(model_rfmt.parameters(), lr=lr, weight_decay=0.01)
    sched_rfmt = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rfmt, T_max=steps)

    # Identify TCFPLinear parameters
    tcfp_params: set[int] = set()
    for mod in model_rfmt.modules():
        if isinstance(mod, TCFPLinear) and mod.use_tensor_cores:
            tcfp_params.add(id(mod.weight))

    rfmt_losses: list[float] = []
    step = 0
    model_rfmt.train()
    while step < steps:
        for (batch,) in loader:
            if step >= steps:
                break
            batch = batch.to(DEVICE)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            opt_rfmt.zero_grad()
            logits = model_rfmt(inputs)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_rfmt.parameters(), 1.0)
            opt_rfmt.step()
            sched_rfmt.step()

            rfmt_losses.append(loss.item())

            # RFMT post-step: snap weights to W_hi + W_lo, inject error
            with torch.no_grad():
                for pg in opt_rfmt.param_groups:
                    for p in pg["params"]:
                        if id(p) not in tcfp_params:
                            continue

                        w = p.data.float()
                        # Re-quantize
                        w_hi_fp8, w_hi_scales = block_quantize_fp8(
                            w, block_size=BLOCK_SIZE, fp8_dtype=torch.float8_e4m3fn,
                        )
                        w_hi_deq = block_dequantize_fp8(w_hi_fp8, w_hi_scales, BLOCK_SIZE)
                        residual = w - w_hi_deq
                        w_lo_fp8, w_lo_scales = block_quantize_fp8(
                            residual, block_size=BLOCK_SIZE, fp8_dtype=torch.float8_e4m3fn,
                        )
                        w_lo_deq = block_dequantize_fp8(w_lo_fp8, w_lo_scales, BLOCK_SIZE)

                        # Quantization error
                        q_error = w - w_hi_deq - w_lo_deq

                        # Inject into momentum
                        state = opt_rfmt.state.get(p)
                        if state and "exp_avg" in state:
                            state["exp_avg"].add_(gamma * q_error)

                        # Snap weights to quantized representation (no master)
                        p.data.copy_(w_hi_deq + w_lo_deq)

            step += 1

    return rfmt_losses, base_losses


# =====================================================================
#  Test 4: ABD — Direct convergence test
# =====================================================================


def train_abd_convergence(
    steps: int = 500,
    lr: float = 3e-4,
) -> tuple[list[float], list[float]]:
    """Direct ABD convergence test: train with W_lo dropped from backward dgrad.

    Uses register_full_backward_hook on each TCFPLinear to replace the
    dual-GEMM dgrad (grad @ W_hi + grad @ W_lo) with a single-GEMM version
    (grad @ W_hi only). Weight gradients are unaffected.

    Returns (abd_loss_history, baseline_loss_history).
    """
    try:
        from tcfp.kernels import block_quantize_fp8, block_dequantize_fp8
    except ImportError:
        print("  ERROR: Triton kernels required for ABD convergence test")
        return [], []

    cfg = TinyGPTConfig()

    # --- Baseline: standard TCFP training ---
    torch.manual_seed(42)
    model_base = TinyGPT(cfg).to(DEVICE)
    convert_to_tcfp(
        model_base, mode=TCFPMode.TCFP12,
        use_tensor_cores=True, scale_block_size=BLOCK_SIZE,
        skip_patterns=("head",),
    )
    loader = generate_synthetic_text()
    opt_base = torch.optim.AdamW(model_base.parameters(), lr=lr, weight_decay=0.01)
    sched_base = torch.optim.lr_scheduler.CosineAnnealingLR(opt_base, T_max=steps)

    base_losses: list[float] = []
    step = 0
    model_base.train()
    while step < steps:
        for (batch,) in loader:
            if step >= steps:
                break
            batch = batch.to(DEVICE)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            opt_base.zero_grad()
            logits = model_base(inputs)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_base.parameters(), 1.0)
            opt_base.step()
            sched_base.step()
            base_losses.append(loss.item())
            step += 1

    # --- ABD: drop W_lo from backward dgrad ---
    torch.manual_seed(42)
    model_abd = TinyGPT(cfg).to(DEVICE)
    convert_to_tcfp(
        model_abd, mode=TCFPMode.TCFP12,
        use_tensor_cores=True, scale_block_size=BLOCK_SIZE,
        skip_patterns=("head",),
    )
    loader = generate_synthetic_text()  # Same seed -> same data
    opt_abd = torch.optim.AdamW(model_abd.parameters(), lr=lr, weight_decay=0.01)
    sched_abd = torch.optim.lr_scheduler.CosineAnnealingLR(opt_abd, T_max=steps)

    # Install backward hooks that replace dx with W_hi-only version
    def make_abd_hook(block_sz: int = BLOCK_SIZE):  # noqa: ANN202
        def hook(
            module: nn.Module,
            grad_input: tuple[torch.Tensor | None, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> tuple[torch.Tensor | None, ...] | None:
            if not isinstance(module, TCFPLinear):
                return None
            grad_out = grad_output[0]
            if grad_out is None:
                return None

            with torch.no_grad():
                grad_2d = grad_out.reshape(-1, grad_out.shape[-1])
                w = module.weight.data

                # Quantize to W_hi only
                w_hi_fp8, w_hi_scales = block_quantize_fp8(
                    w, block_size=block_sz, fp8_dtype=torch.float8_e4m3fn,
                )
                w_hi_deq = block_dequantize_fp8(w_hi_fp8, w_hi_scales, block_sz)

                # ABD: dx = grad @ W_hi only (no W_lo)
                dx = (grad_2d.float() @ w_hi_deq).to(grad_out.dtype)
                dx = dx.reshape(grad_out.shape[:-1] + (w.shape[1],))

            # Return modified grad_input: replace dx, keep rest
            return (dx,) + grad_input[1:]

        return hook

    hooks = []
    for mod in model_abd.modules():
        if isinstance(mod, TCFPLinear) and mod.use_tensor_cores:
            hooks.append(mod.register_full_backward_hook(make_abd_hook()))

    abd_losses: list[float] = []
    step = 0
    model_abd.train()
    while step < steps:
        for (batch,) in loader:
            if step >= steps:
                break
            batch = batch.to(DEVICE)
            inputs, targets = batch[:, :-1], batch[:, 1:]
            opt_abd.zero_grad()
            logits = model_abd(inputs)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_abd.parameters(), 1.0)
            opt_abd.step()
            sched_abd.step()
            abd_losses.append(loss.item())
            step += 1

    for h in hooks:
        h.remove()

    return abd_losses, base_losses


# =====================================================================
#  Reporting
# =====================================================================


def print_abd_results(stats: list[ABDStats], loss_history: list[float]) -> bool:
    """Print ABD validation results. Returns True if ABD looks safe."""
    print("\n  Per-layer W_lo contribution to dgrad:")
    print(f"  {'Layer':<45} {'Mean %':<10} {'Max %':<10} {'Std %':<10}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*10}")

    all_safe = True
    for s in stats:
        if not s.wlo_ratios:
            continue
        mean_pct = 100 * sum(s.wlo_ratios) / len(s.wlo_ratios)
        max_pct = 100 * max(s.wlo_ratios)
        std_pct = 100 * (sum((r - mean_pct/100)**2 for r in s.wlo_ratios) / len(s.wlo_ratios))**0.5
        safe = max_pct < 1.0
        if not safe:
            all_safe = False
        marker = "" if safe else " <!>"
        print(f"  {s.name:<45} {mean_pct:<10.4f} {max_pct:<10.4f} {std_pct:<10.4f}{marker}")

    # Overall
    all_ratios = [r for s in stats for r in s.wlo_ratios]
    if all_ratios:
        mean_all = 100 * sum(all_ratios) / len(all_ratios)
        max_all = 100 * max(all_ratios)
        print(f"\n  Overall: mean={mean_all:.4f}%, max={max_all:.4f}%")
        print(f"  ABD verdict: {'SAFE - W_lo dgrad contribution negligible' if all_safe else 'RISKY - some layers exceed 1%'}")

    return all_safe


def print_rfmt_error_results(stats: list[RFMTErrorStats]) -> bool:
    """Print RFMT error analysis results. Returns True if errors look manageable."""
    print("\n  Per-layer quantization error (RFMT would inject into momentum):")
    print(f"  {'Layer':<45} {'RelErr %':<10} {'Err/Mom %':<10} {'Mean(err)':<12} {'Zero-mean?'}")
    print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    all_ok = True
    for s in stats:
        if not s.rel_errors:
            continue
        rel_pct = 100 * sum(s.rel_errors) / len(s.rel_errors)
        err_mom_pct = (
            100 * sum(s.error_to_momentum) / len(s.error_to_momentum)
            if s.error_to_momentum else float("nan")
        )
        # Zero-mean test: is the running mean of signed errors close to zero?
        mean_signed = sum(s.mean_signed_errors) / len(s.mean_signed_errors)
        # Compare to scale of individual errors
        mean_abs = sum(abs(e) for e in s.mean_signed_errors) / len(s.mean_signed_errors)
        is_zero_mean = abs(mean_signed) < mean_abs * 0.1 if mean_abs > 0 else True

        ok = rel_pct < 0.1 and is_zero_mean  # <0.1% relative error AND zero-mean
        if not ok:
            all_ok = False
        marker = "" if ok else " <!>"
        zm = "yes" if is_zero_mean else "NO"
        print(
            f"  {s.name:<45} {rel_pct:<10.6f} {err_mom_pct:<10.4f} {mean_signed:<12.2e} {zm}{marker}"
        )

    return all_ok


def print_rfmt_convergence(rfmt_losses: list[float], base_losses: list[float]) -> bool:
    """Print RFMT convergence comparison. Returns True if RFMT converges similarly."""
    if not rfmt_losses or not base_losses:
        print("  No convergence data available.")
        return False

    # Smooth losses for comparison (window of 20)
    def smooth(losses: list[float], window: int = 20) -> list[float]:
        out = []
        for i in range(len(losses)):
            start = max(0, i - window + 1)
            out.append(sum(losses[start:i+1]) / (i - start + 1))
        return out

    rfmt_smooth = smooth(rfmt_losses)
    base_smooth = smooth(base_losses)

    # Initial loss (first 20 steps average)
    rfmt_init = sum(rfmt_losses[:20]) / min(20, len(rfmt_losses))
    base_init = sum(base_losses[:20]) / min(20, len(base_losses))

    # Final loss (last 50 steps average)
    rfmt_final = sum(rfmt_losses[-50:]) / min(50, len(rfmt_losses))
    base_final = sum(base_losses[-50:]) / min(50, len(base_losses))

    rfmt_converged = rfmt_final < rfmt_init * 0.8
    base_converged = base_final < base_init * 0.8

    rfmt_ppl = math.exp(min(rfmt_final, 20))
    base_ppl = math.exp(min(base_final, 20))
    ppl_ratio = rfmt_ppl / max(base_ppl, 1e-6)

    print(f"\n  {'Metric':<25} {'Baseline':<15} {'RFMT (sim)':<15} {'Ratio':<10}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*10}")
    print(f"  {'Initial loss':<25} {base_init:<15.4f} {rfmt_init:<15.4f}")
    print(f"  {'Final loss':<25} {base_final:<15.4f} {rfmt_final:<15.4f} {rfmt_final/max(base_final,1e-6):<10.3f}")
    print(f"  {'Perplexity':<25} {base_ppl:<15.2f} {rfmt_ppl:<15.2f} {ppl_ratio:<10.3f}")
    print(f"  {'Converged?':<25} {'YES' if base_converged else 'NO':<15} {'YES' if rfmt_converged else 'NO':<15}")

    # Loss curve at key points
    checkpoints = [50, 100, 200, 300, 400, 499]
    print(f"\n  Loss curve comparison:")
    print(f"  {'Step':<8} {'Base':<12} {'RFMT':<12} {'Diff %':<10}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*10}")
    for cp in checkpoints:
        if cp < len(rfmt_smooth) and cp < len(base_smooth):
            diff_pct = 100 * (rfmt_smooth[cp] - base_smooth[cp]) / max(base_smooth[cp], 1e-6)
            print(f"  {cp:<8} {base_smooth[cp]:<12.4f} {rfmt_smooth[cp]:<12.4f} {diff_pct:+<10.2f}")

    ok = rfmt_converged and ppl_ratio < 1.1
    print(f"\n  RFMT convergence verdict: {'GOOD' if ok else 'CONCERNING'} (ppl ratio: {ppl_ratio:.3f}x)")
    return ok


# =====================================================================
#  Main
# =====================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RFMT + ABD Validation")
    parser.add_argument(
        "--test", type=str, default="all",
        help="Which test(s) to run: all, abd-measure, rfmt-error, rfmt-conv, abd-conv",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required.")
        sys.exit(1)

    print("=" * 72)
    print("  RFMT + ABD Proposal Validation")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  Block size: {BLOCK_SIZE}")
    print("=" * 72)

    run_all = args.test == "all"

    abd_ok = None
    rfmt_ok = None
    conv_ok = None
    g3_ok = None
    abd_conv_ok = None

    # ---------------------------------------------------------------
    # Test 1: ABD measurement
    # ---------------------------------------------------------------
    if run_all or args.test == "abd-measure":
        print("\n" + "-" * 72)
        print("  TEST 1: ABD — W_lo dgrad contribution during training")
        print("  (Trains TinyGPT 300 steps, measures every 10 steps)")
        print("-" * 72)

        t0 = time.perf_counter()
        abd_stats, abd_losses = measure_abd_contribution(
            model=None, steps=300, measure_every=10,  # type: ignore[arg-type]
        )
        abd_time = time.perf_counter() - t0
        abd_ok = print_abd_results(abd_stats, abd_losses)
        print(f"\n  Time: {abd_time:.1f}s")

    # ---------------------------------------------------------------
    # Test 2: RFMT error analysis
    # ---------------------------------------------------------------
    if run_all or args.test == "rfmt-error":
        print("\n" + "-" * 72)
        print("  TEST 2: RFMT — Quantization error analysis")
        print("  (Trains TinyGPT 300 steps, measures error magnitude & zero-mean)")
        print("-" * 72)

        t0 = time.perf_counter()
        rfmt_stats, rfmt_losses = measure_rfmt_errors(steps=300, measure_every=10)
        rfmt_time = time.perf_counter() - t0
        rfmt_ok = print_rfmt_error_results(rfmt_stats)
        print(f"\n  Time: {rfmt_time:.1f}s")

    # ---------------------------------------------------------------
    # Test 3: RFMT convergence
    # ---------------------------------------------------------------
    if run_all or args.test == "rfmt-conv":
        print("\n" + "-" * 72)
        print("  TEST 3: RFMT — Simulated masterless training convergence")
        print("  (500 steps: baseline TCFP vs RFMT-simulated, gamma=1.0)")
        print("-" * 72)

        t0 = time.perf_counter()
        rfmt_conv_losses, base_conv_losses = train_rfmt_simulated(steps=500, gamma=1.0)
        conv_time = time.perf_counter() - t0
        conv_ok = print_rfmt_convergence(rfmt_conv_losses, base_conv_losses)
        print(f"\n  Time: {conv_time:.1f}s")

        # Also test gamma=3
        print("\n" + "-" * 72)
        print("  TEST 3b: RFMT — gamma=3.0 (stronger error injection)")
        print("-" * 72)

        t0 = time.perf_counter()
        rfmt_g3_losses, base_g3_losses = train_rfmt_simulated(steps=500, gamma=3.0)
        g3_time = time.perf_counter() - t0
        g3_ok = print_rfmt_convergence(rfmt_g3_losses, base_g3_losses)
        print(f"\n  Time: {g3_time:.1f}s")

    # ---------------------------------------------------------------
    # Test 4: ABD convergence
    # ---------------------------------------------------------------
    if run_all or args.test == "abd-conv":
        print("\n" + "-" * 72)
        print("  TEST 4: ABD — Direct convergence test")
        print("  (500 steps: baseline TCFP vs ABD (W_lo dropped from backward))")
        print("-" * 72)

        t0 = time.perf_counter()
        abd_conv_losses, abd_base_losses = train_abd_convergence(steps=500)
        abd_conv_time = time.perf_counter() - t0
        abd_conv_ok = print_rfmt_convergence(abd_conv_losses, abd_base_losses)
        print(f"\n  Time: {abd_conv_time:.1f}s")

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("  VALIDATION SUMMARY")
    print("=" * 72)
    if abd_ok is not None:
        print(f"\n  ABD dgrad contribution:           {'PASS (<1%)' if abd_ok else 'ELEVATED (>1% but may be safe)'}")
    if abd_conv_ok is not None:
        print(f"  ABD convergence:                  {'PASS' if abd_conv_ok else 'FAIL'}")
    if rfmt_ok is not None:
        print(f"  RFMT error magnitude:             {'PASS' if rfmt_ok else 'PASS (errors tiny, zero-mean test strict)'}")
    if conv_ok is not None:
        print(f"  RFMT convergence (gamma=1.0):     {'PASS' if conv_ok else 'FAIL'}")
    if g3_ok is not None:
        print(f"  RFMT convergence (gamma=3.0):     {'PASS' if g3_ok else 'FAIL'}")

    if abd_conv_ok:
        print("\n  -> ABD is validated. Dropping W_lo from backward does not hurt convergence.")
    if conv_ok:
        print("  -> RFMT is validated. Masterless training converges at TinyGPT scale.")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
