"""
SRR (Stochastic Residual Rounding) Validation
================================================

Validates whether stochastic rounding on W_lo can replace deterministic
quantization + FP32 error feedback buffer. SRR claims:

  E[dequant(SR(residual))] = residual  (exact per-step unbiasedness)

This eliminates the ErrorFeedbackState entirely (13 GB at 7B scale).

**Test 1 — SR Unbiasedness**: Verify stochastic rounding is truly unbiased
  for W_lo-scale values. Run many SR trials, check E[SR(x)] = x.

**Test 2 — Variance Analysis**: Measure SR variance at W_lo magnitudes
  and compare to mini-batch gradient noise.

**Test 3 — SRR Training Convergence**: Train TinyGPT with SRR
  (stochastic W_lo, no error buffer) vs standard TCFP (deterministic
  W_lo + error feedback). Compare loss curves.

**Test 4 — 10K Step Long Run**: Extended training to check for
  divergence or drift over many steps.

Run:  PYTHONIOENCODING=utf-8 py -m benchmarks.validate_srr [--test NAME] [--steps N]
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tcfp.core import (
    FP8_E4M3_MAX,
    FP8_E4M3_MIN,
    TCFPMode,
    stochastic_round_to_fp8_e4m3,
)
from tcfp.nn import TCFPLinear, convert_to_tcfp

DEVICE = "cuda"
BLOCK_SIZE = 128  # For no-EF test (block-scaled path)
# SRR convergence test uses per-tensor path (no block scaling)
# because the block-scaled path fuses quantization in a Triton kernel
# that can't be monkey-patched for SR.


# =====================================================================
#  Shared: TinyGPT model + data (from validate_rfmt.py)
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
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
        mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
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
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
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
    n_sequences: int = 4000,
    seq_len: int = 128,
    vocab_size: int = 64,
) -> DataLoader[tuple[torch.Tensor, ...]]:
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
                noise = (
                    torch.randint(0, 2, (1,)).item()
                    if torch.rand(1).item() < 0.05
                    else 0
                )
                data[i, t] = (motif[t % motif_len].item() + noise) % vocab_size
        else:
            a = torch.randint(0, vocab_size // 2, (1,)).item()
            b = torch.randint(vocab_size // 2, vocab_size, (1,)).item()
            for t in range(seq_len):
                data[i, t] = a if t % 2 == 0 else b
    return DataLoader(TensorDataset(data), batch_size=32, shuffle=True)


# =====================================================================
#  Test 1: SR Unbiasedness Verification
# =====================================================================


def test_sr_unbiasedness(n_trials: int = 10000) -> None:
    """Verify that stochastic rounding is unbiased for W_lo-scale values."""
    print("\n=== Test 1: Stochastic Rounding Unbiasedness ===\n")

    # W_lo values cluster around amax ~7.6e-6 (exponent ~ -17)
    # Test at multiple magnitudes
    test_magnitudes = [1e-4, 7.6e-6, 1e-6, 1e-8]

    for mag in test_magnitudes:
        # Create values at this magnitude, pre-scaled for FP8 range
        values = torch.randn(1024, device=DEVICE) * mag

        # Compute scale for FP8
        amax = values.abs().max().clamp(min=1e-12)
        scale = (FP8_E4M3_MAX / amax).to(torch.float32)
        inv_scale = 1.0 / scale

        # Scale to FP8 range
        scaled = (values.float() * scale).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)

        # Run many SR trials and average
        sr_sum = torch.zeros_like(scaled)
        for _ in range(n_trials):
            sr_result = stochastic_round_to_fp8_e4m3(scaled)
            sr_sum += sr_result.float()

        sr_mean = sr_sum / n_trials
        # Dequantize back
        sr_mean_deq = sr_mean * inv_scale
        rtn_result = scaled.to(torch.float8_e4m3fn).float() * inv_scale

        # Compare bias
        sr_bias = (sr_mean_deq - values).abs().mean()
        rtn_bias = (rtn_result - values).abs().mean()
        relative_sr_bias = sr_bias / values.abs().mean().clamp(min=1e-30)
        relative_rtn_bias = rtn_bias / values.abs().mean().clamp(min=1e-30)

        status = "PASS" if relative_sr_bias < 0.01 else "FAIL"
        print(f"  mag={mag:.0e}: SR bias={relative_sr_bias:.6f} "
              f"RTN bias={relative_rtn_bias:.6f} [{status}]")

    print()


# =====================================================================
#  Test 2: Variance Analysis
# =====================================================================


def test_variance_analysis(n_trials: int = 1000) -> None:
    """Measure SR variance at W_lo magnitudes, compare to gradient noise."""
    print("\n=== Test 2: SR Variance vs Gradient Noise ===\n")

    # Create a small TCFP model and measure gradient variance
    torch.manual_seed(42)
    cfg = TinyGPTConfig()
    model = TinyGPT(cfg).to(DEVICE)
    convert_to_tcfp(
        model, mode=TCFPMode.TCFP12,
        use_tensor_cores=True, scale_block_size=BLOCK_SIZE,
        skip_patterns=("head",),
    )
    loader = generate_synthetic_text(n_sequences=200, seq_len=64)

    # Measure gradient variance over a few mini-batches
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    grad_samples: list[torch.Tensor] = []

    for step, (batch,) in enumerate(loader):
        if step >= 10:
            break
        batch = batch.to(DEVICE)
        inp, tgt = batch[:, :-1], batch[:, 1:]
        optimizer.zero_grad()
        logits = model(inp)
        loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), tgt.reshape(-1))
        loss.backward()

        # Collect gradients from first TCFPLinear
        for mod in model.modules():
            if isinstance(mod, TCFPLinear):
                if mod.weight.grad is not None:
                    grad_samples.append(mod.weight.grad.detach().clone())
                break

    if len(grad_samples) < 2:
        print("  Not enough gradient samples collected")
        return

    grad_stack = torch.stack(grad_samples)
    grad_variance = grad_stack.var(dim=0).mean().item()
    grad_std = grad_variance**0.5

    # Measure SR variance at W_lo magnitude
    mag = 7.6e-6  # typical W_lo amax
    values = torch.randn(1024, device=DEVICE) * mag
    amax = values.abs().max().clamp(min=1e-12)
    scale = (FP8_E4M3_MAX / amax).to(torch.float32)
    inv_scale = 1.0 / scale
    scaled = (values.float() * scale).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)

    sr_results = []
    for _ in range(n_trials):
        sr = stochastic_round_to_fp8_e4m3(scaled).float() * inv_scale
        sr_results.append(sr)

    sr_stack = torch.stack(sr_results)
    sr_variance = sr_stack.var(dim=0).mean().item()
    sr_std = sr_variance**0.5

    ratio = sr_std / grad_std if grad_std > 0 else float("inf")
    status = "PASS" if ratio < 0.01 else "MARGINAL" if ratio < 0.1 else "FAIL"

    print(f"  Gradient std (per-element): {grad_std:.2e}")
    print(f"  SR std at W_lo mag ({mag:.0e}): {sr_std:.2e}")
    print(f"  Ratio (SR noise / grad noise): {ratio:.4f} [{status}]")
    print(f"    -> SR noise is {1/ratio:.0f}x smaller than gradient noise"
          if ratio > 0 else "")
    print()


# =====================================================================
#  Test 3: SRR Training Convergence (short)
# =====================================================================


def _to_fp8_e4m3_sr(
    tensor: torch.Tensor,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 E4M3 with stochastic rounding (SR version of to_fp8_e4m3)."""
    if scale is None:
        amax = tensor.abs().max().clamp(min=1e-12)
        scale = (FP8_E4M3_MAX / amax).to(torch.float32)

    scaled = (tensor.float() * scale).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    fp8 = stochastic_round_to_fp8_e4m3(scaled)
    inv_scale = (
        torch.ones(1, device=tensor.device, dtype=torch.float32) / scale
    )
    return fp8, inv_scale


def train_srr_convergence(
    steps: int = 500,
    lr: float = 3e-4,
) -> tuple[list[float], list[float]]:
    """Train with SRR (SR on W_lo, no error buffer) vs standard TCFP.

    SRR is simulated via hook: after each optimizer step, we re-quantize
    W_lo using stochastic rounding and disable error feedback.
    """
    print(f"\n=== Test 3: SRR Convergence ({steps} steps) ===\n")

    cfg = TinyGPTConfig()
    loader = generate_synthetic_text()

    # --- Standard TCFP (baseline) — per-tensor TC path ---
    torch.manual_seed(42)
    model_std = TinyGPT(cfg).to(DEVICE)
    convert_to_tcfp(
        model_std, mode=TCFPMode.TCFP12,
        use_tensor_cores=True,
        skip_patterns=("head",),
    )
    opt_std = torch.optim.AdamW(model_std.parameters(), lr=lr, weight_decay=0.01)
    sched_std = torch.optim.lr_scheduler.CosineAnnealingLR(opt_std, T_max=steps)

    # --- SRR model (no error feedback, SR on W_lo) — per-tensor TC path ---
    torch.manual_seed(42)
    model_srr = TinyGPT(cfg).to(DEVICE)
    convert_to_tcfp(
        model_srr, mode=TCFPMode.TCFP12,
        use_tensor_cores=True,
        skip_patterns=("head",),
        error_feedback=False,  # No error buffer
    )
    opt_srr = torch.optim.AdamW(model_srr.parameters(), lr=lr, weight_decay=0.01)
    sched_srr = torch.optim.lr_scheduler.CosineAnnealingLR(opt_srr, T_max=steps)

    # Monkey-patch to_fp8_e4m3 in tcfp.nn module to use SR on W_lo.
    # Per-tensor forward calls to_fp8_e4m3 per layer:
    #   Call 1: W_hi quantization (deterministic -- keep)
    #   Call 2: W_lo quantization (SR -- patch this one)
    #   Call 3: Activation quantization (deterministic -- keep)
    # We count calls: every 3rd call starting at 2 is W_lo.
    import tcfp.nn as nn_mod

    original_to_fp8_e4m3 = nn_mod.to_fp8_e4m3
    _call_count = {"n": 0}

    def srr_to_fp8_e4m3(
        tensor: torch.Tensor,
        scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _call_count["n"] += 1
        if _call_count["n"] % 3 == 2:  # 2nd of every 3 calls = W_lo
            return _to_fp8_e4m3_sr(tensor, scale)
        return original_to_fp8_e4m3(tensor, scale)

    losses_std: list[float] = []
    losses_srr: list[float] = []

    data_iter_std = iter(loader)
    data_iter_srr = iter(loader)

    t0 = time.time()
    for step in range(steps):
        # Standard training step
        try:
            (batch_std,) = next(data_iter_std)
        except StopIteration:
            data_iter_std = iter(loader)
            (batch_std,) = next(data_iter_std)

        batch_std = batch_std.to(DEVICE)
        inp_s, tgt_s = batch_std[:, :-1], batch_std[:, 1:]
        opt_std.zero_grad()
        logits_s = model_std(inp_s)
        loss_s = F.cross_entropy(
            logits_s.reshape(-1, cfg.vocab_size), tgt_s.reshape(-1)
        )
        loss_s.backward()
        opt_std.step()
        sched_std.step()
        losses_std.append(loss_s.item())

        # SRR training step (patch to_fp8_e4m3 for SR on W_lo)
        try:
            (batch_srr,) = next(data_iter_srr)
        except StopIteration:
            data_iter_srr = iter(loader)
            (batch_srr,) = next(data_iter_srr)

        batch_srr = batch_srr.to(DEVICE)
        inp_r, tgt_r = batch_srr[:, :-1], batch_srr[:, 1:]
        opt_srr.zero_grad()

        # Monkey-patch for this forward pass only
        _call_count["n"] = 0
        nn_mod.to_fp8_e4m3 = srr_to_fp8_e4m3  # type: ignore[attr-defined]

        logits_r = model_srr(inp_r)
        loss_r = F.cross_entropy(
            logits_r.reshape(-1, cfg.vocab_size), tgt_r.reshape(-1)
        )
        loss_r.backward()

        # Restore original
        nn_mod.to_fp8_e4m3 = original_to_fp8_e4m3  # type: ignore[attr-defined]

        opt_srr.step()
        sched_srr.step()
        losses_srr.append(loss_r.item())

        if (step + 1) % 100 == 0 or step == 0:
            elapsed = time.time() - t0
            print(
                f"  step {step + 1:4d}/{steps}  "
                f"std={losses_std[-1]:.4f}  srr={losses_srr[-1]:.4f}  "
                f"ratio={losses_srr[-1] / max(losses_std[-1], 1e-8):.3f}x  "
                f"({elapsed:.1f}s)"
            )

    # Compute final metrics
    window = min(50, steps // 4)
    avg_std = sum(losses_std[-window:]) / window
    avg_srr = sum(losses_srr[-window:]) / window
    ppl_std = 2**avg_std if avg_std < 20 else float("inf")
    ppl_srr = 2**avg_srr if avg_srr < 20 else float("inf")

    nan_srr = any(
        not torch.isfinite(torch.tensor(v)) for v in losses_srr
    )

    ratio = ppl_srr / ppl_std if ppl_std > 0 else float("inf")
    status = "FAIL (NaN)" if nan_srr else (
        "PASS" if ratio < 1.05 else "MARGINAL" if ratio < 1.2 else "FAIL"
    )

    print(f"\n  Result: ppl_std={ppl_std:.2f}  ppl_srr={ppl_srr:.2f}  "
          f"ratio={ratio:.4f}x  [{status}]")
    print()

    return losses_std, losses_srr


# =====================================================================
#  Test 4: No-Error-Feedback Baseline (simpler SRR proxy)
# =====================================================================


def train_no_error_feedback(
    steps: int = 500,
    lr: float = 3e-4,
) -> tuple[list[float], list[float]]:
    """Compare standard TCFP vs TCFP without error feedback.

    If no-error-feedback converges similarly to standard TCFP,
    then the error buffer provides minimal value and SRR
    (which replaces it with per-step SR unbiasedness) is viable.
    """
    print(f"\n=== Test 4: No-Error-Feedback Baseline ({steps} steps) ===\n")

    cfg = TinyGPTConfig()
    loader = generate_synthetic_text()

    # Standard TCFP (with error feedback)
    torch.manual_seed(42)
    model_ef = TinyGPT(cfg).to(DEVICE)
    convert_to_tcfp(
        model_ef, mode=TCFPMode.TCFP12,
        use_tensor_cores=True, scale_block_size=BLOCK_SIZE,
        skip_patterns=("head",), error_feedback=True,
    )
    opt_ef = torch.optim.AdamW(model_ef.parameters(), lr=lr, weight_decay=0.01)
    sched_ef = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ef, T_max=steps)

    # No error feedback
    torch.manual_seed(42)
    model_no = TinyGPT(cfg).to(DEVICE)
    convert_to_tcfp(
        model_no, mode=TCFPMode.TCFP12,
        use_tensor_cores=True, scale_block_size=BLOCK_SIZE,
        skip_patterns=("head",), error_feedback=False,
    )
    opt_no = torch.optim.AdamW(model_no.parameters(), lr=lr, weight_decay=0.01)
    sched_no = torch.optim.lr_scheduler.CosineAnnealingLR(opt_no, T_max=steps)

    losses_ef: list[float] = []
    losses_no: list[float] = []

    data_iter_ef = iter(loader)
    data_iter_no = iter(loader)

    t0 = time.time()
    for step in range(steps):
        # With error feedback
        try:
            (batch_ef,) = next(data_iter_ef)
        except StopIteration:
            data_iter_ef = iter(loader)
            (batch_ef,) = next(data_iter_ef)
        batch_ef = batch_ef.to(DEVICE)
        inp_e, tgt_e = batch_ef[:, :-1], batch_ef[:, 1:]
        opt_ef.zero_grad()
        logits_e = model_ef(inp_e)
        loss_e = F.cross_entropy(
            logits_e.reshape(-1, cfg.vocab_size), tgt_e.reshape(-1)
        )
        loss_e.backward()
        opt_ef.step()
        sched_ef.step()
        losses_ef.append(loss_e.item())

        # Without error feedback
        try:
            (batch_no,) = next(data_iter_no)
        except StopIteration:
            data_iter_no = iter(loader)
            (batch_no,) = next(data_iter_no)
        batch_no = batch_no.to(DEVICE)
        inp_n, tgt_n = batch_no[:, :-1], batch_no[:, 1:]
        opt_no.zero_grad()
        logits_n = model_no(inp_n)
        loss_n = F.cross_entropy(
            logits_n.reshape(-1, cfg.vocab_size), tgt_n.reshape(-1)
        )
        loss_n.backward()
        opt_no.step()
        sched_no.step()
        losses_no.append(loss_n.item())

        if (step + 1) % 100 == 0 or step == 0:
            elapsed = time.time() - t0
            print(
                f"  step {step + 1:4d}/{steps}  "
                f"ef={losses_ef[-1]:.4f}  no_ef={losses_no[-1]:.4f}  "
                f"ratio={losses_no[-1] / max(losses_ef[-1], 1e-8):.3f}x  "
                f"({elapsed:.1f}s)"
            )

    window = min(50, steps // 4)
    avg_ef = sum(losses_ef[-window:]) / window
    avg_no = sum(losses_no[-window:]) / window
    ppl_ef = 2**avg_ef if avg_ef < 20 else float("inf")
    ppl_no = 2**avg_no if avg_no < 20 else float("inf")

    ratio = ppl_no / ppl_ef if ppl_ef > 0 else float("inf")
    status = "PASS" if ratio < 1.05 else "MARGINAL" if ratio < 1.2 else "FAIL"

    print(f"\n  Result: ppl_ef={ppl_ef:.2f}  ppl_no={ppl_no:.2f}  "
          f"ratio={ratio:.4f}x  [{status}]")
    print(f"    -> Error feedback {'not needed' if status == 'PASS' else 'matters'} "
          f"at this scale")
    print()

    return losses_ef, losses_no


# =====================================================================
#  Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description="SRR Validation")
    parser.add_argument(
        "--test",
        choices=["all", "sr-bias", "variance", "srr-conv", "no-ef"],
        default="all",
        help="Which test to run",
    )
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    args = parser.parse_args()

    print("=" * 60)
    print("  SRR (Stochastic Residual Rounding) Validation")
    print("=" * 60)

    if args.test in ("all", "sr-bias"):
        test_sr_unbiasedness()

    if args.test in ("all", "variance"):
        test_variance_analysis()

    if args.test in ("all", "no-ef"):
        train_no_error_feedback(steps=args.steps)

    if args.test in ("all", "srr-conv"):
        train_srr_convergence(steps=args.steps)

    print("\nDone.")


if __name__ == "__main__":
    main()
