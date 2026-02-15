"""
TCFP Training Convergence Tests
=================================

Two real training tasks to validate that TCFP modes actually converge:

1. **MNIST** — Quick sanity check. 3-layer MLP, 10 epochs.
   If this fails, nothing else matters.

2. **TinyGPT** — Realistic transformer. 4-layer, 256-dim, causal LM on
   synthetic data. 500 steps. Tests gradient flow through attention,
   layernorm, embeddings — the full training stack.

For each task, we train with:
  - BF16 baseline (FP32 master weights, BF16 forward)
  - TCFP-8
  - TCFP-12  ← the one we care about most
  - TCFP-16

Reports: final loss, accuracy/perplexity, loss curve stability.

Run:  py -m examples.train_convergence
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from tcfp.core import TCFPMode
from tcfp.nn import convert_to_tcfp

DEVICE = "cuda"


# =====================================================================
#  Part 1: MNIST Convergence
# =====================================================================


class MNISTNet(nn.Module):
    """Simple 3-layer MLP for MNIST classification."""

    def __init__(self, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def generate_fake_mnist(
    n_train: int = 10000,
    n_test: int = 2000,
) -> tuple[DataLoader, DataLoader]:
    """Generate synthetic MNIST-like data (avoids torchvision dependency).

    Creates 784-dim data with strong class-dependent structure:
    each class has a unique spatial pattern (stripes, quadrants, etc.)
    with moderate noise. This is clearly learnable but non-trivial.
    """
    torch.manual_seed(42)

    # Create 10 distinct class prototypes with strong structure
    prototypes = torch.zeros(10, 784)
    for c in range(10):
        proto = torch.zeros(28, 28)
        if c == 0:
            proto[4:24, 8:20] = 1.0  # vertical bar center
        elif c == 1:
            proto[8:20, 4:24] = 1.0  # horizontal bar center
        elif c == 2:
            proto[:14, :14] = 1.0  # top-left quadrant
        elif c == 3:
            proto[:14, 14:] = 1.0  # top-right quadrant
        elif c == 4:
            proto[14:, :14] = 1.0  # bottom-left quadrant
        elif c == 5:
            proto[14:, 14:] = 1.0  # bottom-right quadrant
        elif c == 6:
            for i in range(28):
                proto[i, :] = 1.0 if i % 4 < 2 else 0.0  # horizontal stripes
        elif c == 7:
            for j in range(28):
                proto[:, j] = 1.0 if j % 4 < 2 else 0.0  # vertical stripes
        elif c == 8:
            for i in range(28):
                for j in range(28):
                    if (i + j) % 6 < 3:
                        proto[i, j] = 1.0  # diagonal stripes
        elif c == 9:
            for i in range(28):
                for j in range(28):
                    if abs(i - 14) + abs(j - 14) < 10:
                        proto[i, j] = 1.0  # diamond
        prototypes[c] = proto.reshape(784)

    def make_data(n: int) -> tuple[torch.Tensor, torch.Tensor]:
        labels = torch.randint(0, 10, (n,))
        images = prototypes[labels].clone()
        # Add noise: enough to make it non-trivial but pattern is clearly visible
        images += torch.randn_like(images) * 0.3
        return images, labels

    train_x, train_y = make_data(n_train)
    test_x, test_y = make_data(n_test)

    train_loader = DataLoader(
        TensorDataset(train_x, train_y), batch_size=128, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y), batch_size=256, shuffle=False
    )
    return train_loader, test_loader


def train_mnist(
    mode_name: str,
    mode: TCFPMode | None,
    epochs: int = 10,
    lr: float = 1e-3,
) -> dict:
    """Train MNIST and return metrics."""
    # Make dimensions divisible by 32: pad 784 → 800 internally
    # Actually 256 is divisible by 32, so hidden layers are fine.
    # 784 is NOT divisible by 32, so fc1 won't be converted — that's fine,
    # it acts as an unquantized input projection.

    model = MNISTNet(hidden=256).to(DEVICE)
    if mode is not None:
        convert_to_tcfp(model, mode=mode, block_size=32, skip_patterns=())

    train_loader, test_loader = generate_fake_mnist()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history: list[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images).argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    return {
        "mode": mode_name,
        "final_loss": loss_history[-1],
        "accuracy": accuracy,
        "loss_history": loss_history,
        "converged": loss_history[-1] < loss_history[0] * 0.5,  # Loss halved
    }


# =====================================================================
#  Part 2: TinyGPT Convergence
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
        q, k, v = qkv.unbind(dim=2)  # Each: (B, T, n_heads, d_head)
        q = q.transpose(1, 2)  # (B, n_heads, T, d_head)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
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
        # Weight tying
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
) -> DataLoader:
    """Generate synthetic text with strongly learnable patterns.

    Creates sequences with clear deterministic structure + small noise:
    - Pattern A: arithmetic sequences (tokens count up modulo vocab)
    - Pattern B: repeating short motifs
    - Pattern C: palindrome-like mirror patterns

    A good LM should easily drive perplexity well below random (=64).
    """
    torch.manual_seed(42)
    data = torch.zeros(n_sequences, seq_len, dtype=torch.long)

    for i in range(n_sequences):
        pattern = i % 3
        if pattern == 0:
            # Arithmetic: start + step*t (mod vocab_size)
            start = torch.randint(0, vocab_size, (1,)).item()
            step = torch.randint(1, 4, (1,)).item()
            for t in range(seq_len):
                noise = 1 if torch.rand(1).item() < 0.05 else 0  # 5% noise
                data[i, t] = (start + step * t + noise) % vocab_size
        elif pattern == 1:
            # Repeating motif of length 4-8
            motif_len = torch.randint(4, 9, (1,)).item()
            motif = torch.randint(0, vocab_size, (motif_len,))
            for t in range(seq_len):
                noise = torch.randint(0, 2, (1,)).item() if torch.rand(1).item() < 0.05 else 0
                data[i, t] = (motif[t % motif_len].item() + noise) % vocab_size
        else:
            # Alternating: A B A B A B...  where A,B are from different halves
            a = torch.randint(0, vocab_size // 2, (1,)).item()
            b = torch.randint(vocab_size // 2, vocab_size, (1,)).item()
            for t in range(seq_len):
                data[i, t] = a if t % 2 == 0 else b

    return DataLoader(TensorDataset(data), batch_size=32, shuffle=True)


def train_tinygpt(
    mode_name: str,
    mode: TCFPMode | None,
    steps: int = 1000,
    lr: float = 3e-4,
) -> dict:
    """Train TinyGPT and return metrics."""
    cfg = TinyGPTConfig()
    model = TinyGPT(cfg).to(DEVICE)

    if mode is not None:
        # Skip the head (tied with embeddings) to avoid double-conversion issues
        convert_to_tcfp(model, mode=mode, block_size=32, skip_patterns=("head",))

    n_params = sum(p.numel() for p in model.parameters())
    loader = generate_synthetic_text(n_sequences=4000, seq_len=cfg.max_seq_len)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    loss_history: list[float] = []
    step = 0
    model.train()

    while step < steps:
        for (batch,) in loader:
            if step >= steps:
                break
            batch = batch.to(DEVICE)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            loss_history.append(loss.item())
            step += 1

            if step % 100 == 0:
                print(f"    [{mode_name}] step {step}/{steps}, loss={loss.item():.4f}")

    # Final evaluation: compute perplexity on last 50 steps
    avg_final_loss = sum(loss_history[-50:]) / 50
    perplexity = math.exp(min(avg_final_loss, 20))  # Cap to avoid overflow

    # Check for NaN/explosion
    has_nan = any(math.isnan(l) or math.isinf(l) for l in loss_history)

    # Check loss decreased meaningfully (at least 20% from initial)
    early_avg = sum(loss_history[:20]) / 20
    converged = avg_final_loss < early_avg * 0.8 and not has_nan

    return {
        "mode": mode_name,
        "n_params": n_params,
        "final_loss": avg_final_loss,
        "perplexity": perplexity,
        "loss_history": loss_history,
        "has_nan": has_nan,
        "converged": converged,
        "initial_loss": early_avg,
    }


# =====================================================================
#  Main
# =====================================================================


def print_loss_sparkline(history: list[float], width: int = 40) -> str:
    """Tiny ASCII sparkline of loss curve."""
    if not history:
        return ""
    # Sample down to width points
    step = max(1, len(history) // width)
    sampled = [history[i] for i in range(0, len(history), step)][:width]
    lo, hi = min(sampled), max(sampled)
    if hi - lo < 1e-6:
        return "▁" * len(sampled)
    chars = " ▁▂▃▄▅▆▇█"
    return "".join(
        chars[min(len(chars) - 1, int((v - lo) / (hi - lo) * (len(chars) - 1)))]
        for v in sampled
    )


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA required for TCFP convergence tests.")
        sys.exit(1)

    print("=" * 72)
    print("  TCFP Training Convergence Tests")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  PyTorch: {torch.__version__}")
    print("=" * 72)

    configs = [
        ("BF16 baseline", None),
        ("TCFP-8", TCFPMode.TCFP8),
        ("TCFP-12", TCFPMode.TCFP12),
        ("TCFP-16", TCFPMode.TCFP16),
    ]

    # ── MNIST ─────────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  TEST 1: MNIST Classification (3-layer MLP, 10 epochs)")
    print("─" * 72)

    mnist_results = []
    for name, mode in configs:
        print(f"\n  Training {name}...")
        t0 = time.perf_counter()
        result = train_mnist(name, mode)
        elapsed = time.perf_counter() - t0
        result["time"] = elapsed
        mnist_results.append(result)
        status = "✓ CONVERGED" if result["converged"] else "✗ FAILED"
        print(f"    {status}: loss={result['final_loss']:.4f}, "
              f"acc={result['accuracy']:.1%}, time={elapsed:.1f}s")

    print(f"\n  {'Config':<18} {'Final Loss':<12} {'Accuracy':<10} {'Status':<14} {'Loss Curve'}")
    print(f"  {'-'*18} {'-'*12} {'-'*10} {'-'*14} {'-'*40}")
    for r in mnist_results:
        status = "CONVERGED" if r["converged"] else "FAILED"
        sparkline = print_loss_sparkline(r["loss_history"])
        print(f"  {r['mode']:<18} {r['final_loss']:<12.4f} {r['accuracy']:<10.1%} "
              f"{status:<14} {sparkline}")

    # ── TinyGPT ──────────────────────────────────────────────────────
    print("\n" + "─" * 72)
    print("  TEST 2: TinyGPT Language Model (4L, d=256, 500 steps)")
    print("─" * 72)

    gpt_results = []
    for name, mode in configs:
        print(f"\n  Training {name}...")
        t0 = time.perf_counter()
        result = train_tinygpt(name, mode, steps=500)
        elapsed = time.perf_counter() - t0
        result["time"] = elapsed
        gpt_results.append(result)
        status = "✓ CONVERGED" if result["converged"] else "✗ FAILED"
        nan_warn = " [NaN!]" if result["has_nan"] else ""
        print(f"    {status}: loss={result['final_loss']:.4f}, "
              f"ppl={result['perplexity']:.1f}, time={elapsed:.1f}s{nan_warn}")

    print(f"\n  {'Config':<18} {'Init Loss':<12} {'Final Loss':<12} {'PPL':<10} "
          f"{'Status':<14} {'Loss Curve'}")
    print(f"  {'-'*18} {'-'*12} {'-'*12} {'-'*10} {'-'*14} {'-'*40}")
    for r in gpt_results:
        status = "CONVERGED" if r["converged"] else "FAILED"
        if r["has_nan"]:
            status = "NaN/INF"
        sparkline = print_loss_sparkline(r["loss_history"])
        print(f"  {r['mode']:<18} {r['initial_loss']:<12.4f} {r['final_loss']:<12.4f} "
              f"{r['perplexity']:<10.1f} {status:<14} {sparkline}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    all_converged = all(r["converged"] for r in mnist_results + gpt_results)
    tcfp12_results = [r for r in mnist_results + gpt_results if r["mode"] == "TCFP-12"]
    bf16_results = [r for r in mnist_results + gpt_results if r["mode"] == "BF16 baseline"]

    print(f"\n  All modes converged: {'YES' if all_converged else 'NO'}")

    if tcfp12_results and bf16_results:
        # MNIST comparison
        tcfp12_mnist = tcfp12_results[0]
        bf16_mnist = bf16_results[0]
        acc_gap = bf16_mnist["accuracy"] - tcfp12_mnist["accuracy"]
        print(f"\n  TCFP-12 vs BF16 (MNIST):")
        print(f"    Accuracy gap: {acc_gap:+.1%} "
              f"({'BF16 better' if acc_gap > 0 else 'TCFP-12 better'})")

        # GPT comparison
        if len(tcfp12_results) > 1 and len(bf16_results) > 1:
            tcfp12_gpt = tcfp12_results[1]
            bf16_gpt = bf16_results[1]
            ppl_ratio = tcfp12_gpt["perplexity"] / bf16_gpt["perplexity"]
            print(f"\n  TCFP-12 vs BF16 (TinyGPT):")
            print(f"    Perplexity ratio: {ppl_ratio:.3f}x "
                  f"({'similar' if 0.9 < ppl_ratio < 1.1 else 'degraded' if ppl_ratio > 1 else 'better'})")
            print(f"    BF16 final loss:    {bf16_gpt['final_loss']:.4f}")
            print(f"    TCFP-12 final loss: {tcfp12_gpt['final_loss']:.4f}")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
