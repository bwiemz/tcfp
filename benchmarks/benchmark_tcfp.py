"""
TCFP Benchmark Suite
=====================

Measures quantisation quality (MSE, SNR) and throughput for all three
TCFP modes compared to BF16 and naive FP8 baselines.

Run:  py -m benchmarks.benchmark_tcfp
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from tcfp.core import (
    DEFAULT_BLOCK_SIZE,
    TCFPMode,
    bits_per_value,
    fp8_matmul,
    to_fp8_e4m3,
)
from tcfp.nn import TCFPLinear, convert_to_tcfp
from tcfp.tcfp8 import dequantize_tcfp8, fake_quantize_tcfp8, quantize_tcfp8
from tcfp.tcfp12 import dequantize_tcfp12, fake_quantize_tcfp12, quantize_tcfp12
from tcfp.tcfp16 import (
    dequantize_tcfp16,
    fake_quantize_tcfp16,
    quantize_tcfp16,
    tcfp16_matmul,
)

DEVICE = "cuda"
WARMUP = 5
ITERS = 50


# ── Utilities ─────────────────────────────────────────────────────────────


def snr_db(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Signal-to-noise ratio in dB."""
    signal = original.pow(2).mean()
    noise = (original - reconstructed).pow(2).mean()
    if noise < 1e-20:
        return float("inf")
    return (10 * torch.log10(signal / noise)).item()


def mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    return (original - reconstructed).pow(2).mean().item()


def max_abs_error(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    return (original - reconstructed).abs().max().item()


def time_fn(fn, warmup: int = WARMUP, iters: int = ITERS) -> float:
    """Time a function in milliseconds (GPU-sync'd)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return (elapsed / iters) * 1000  # ms


# ── Benchmark 1: Quantization Quality ────────────────────────────────────


def benchmark_quality() -> None:
    print("\n" + "=" * 70)
    print("  BENCHMARK 1: Quantization Quality (MSE / SNR / Max Error)")
    print("=" * 70)

    shapes = [(256, 512), (1024, 1024), (4096, 4096)]
    distributions = [
        ("Gaussian N(0,1)", lambda s: torch.randn(*s, device=DEVICE)),
        ("Uniform [-1,1]", lambda s: torch.rand(*s, device=DEVICE) * 2 - 1),
        ("Heavy-tail", lambda s: torch.randn(*s, device=DEVICE) * torch.randn(*s, device=DEVICE)),
    ]

    for dist_name, dist_fn in distributions:
        print(f"\n  Distribution: {dist_name}")
        print(f"  {'Shape':<16} {'Format':<14} {'Bits':<7} {'MSE':<12} {'SNR(dB)':<10} {'MaxErr':<10}")
        print(f"  {'-'*16} {'-'*14} {'-'*7} {'-'*12} {'-'*10} {'-'*10}")

        for shape in shapes:
            torch.manual_seed(42)
            x = dist_fn(shape)

            # BF16 baseline
            bf16_recon = x.bfloat16().float()
            print(
                f"  {str(shape):<16} {'BF16':<14} {'16.0':<7} "
                f"{mse(x, bf16_recon):<12.2e} {snr_db(x, bf16_recon):<10.1f} "
                f"{max_abs_error(x, bf16_recon):<10.2e}"
            )

            # Naive FP8 (per-tensor scale, no blocks)
            fp8, inv_s = to_fp8_e4m3(x)
            fp8_recon = fp8.float() * inv_s
            print(
                f"  {'':<16} {'Naive FP8':<14} {'8.0':<7} "
                f"{mse(x, fp8_recon):<12.2e} {snr_db(x, fp8_recon):<10.1f} "
                f"{max_abs_error(x, fp8_recon):<10.2e}"
            )

            # TCFP-8
            tcfp8_recon = fake_quantize_tcfp8(x)
            bpv8 = f"{bits_per_value(TCFPMode.TCFP8):.1f}"
            print(
                f"  {'':<16} {'TCFP-8':<14} {bpv8:<7} "
                f"{mse(x, tcfp8_recon):<12.2e} {snr_db(x, tcfp8_recon):<10.1f} "
                f"{max_abs_error(x, tcfp8_recon):<10.2e}"
            )

            # TCFP-12
            tcfp12_recon = fake_quantize_tcfp12(x)
            bpv12 = f"{bits_per_value(TCFPMode.TCFP12):.1f}"
            print(
                f"  {'':<16} {'TCFP-12':<14} {bpv12:<7} "
                f"{mse(x, tcfp12_recon):<12.2e} {snr_db(x, tcfp12_recon):<10.1f} "
                f"{max_abs_error(x, tcfp12_recon):<10.2e}"
            )

            # TCFP-16
            tcfp16_recon = fake_quantize_tcfp16(x)
            bpv16 = f"{bits_per_value(TCFPMode.TCFP16):.1f}"
            print(
                f"  {'':<16} {'TCFP-16':<14} {bpv16:<7} "
                f"{mse(x, tcfp16_recon):<12.2e} {snr_db(x, tcfp16_recon):<10.1f} "
                f"{max_abs_error(x, tcfp16_recon):<10.2e}"
            )
            print()


# ── Benchmark 2: Fake-Quantize Throughput ────────────────────────────────


def benchmark_fake_quantize_throughput() -> None:
    print("\n" + "=" * 70)
    print("  BENCHMARK 2: Fake-Quantize Throughput (ms per call)")
    print("=" * 70)

    shapes = [(256, 512), (1024, 1024), (4096, 4096)]

    print(f"\n  {'Shape':<16} {'BF16 cast':<12} {'TCFP-8':<12} {'TCFP-12':<12} {'TCFP-16':<12}")
    print(f"  {'-'*16} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    for shape in shapes:
        x = torch.randn(*shape, device=DEVICE)

        t_bf16 = time_fn(lambda: x.bfloat16().float())
        t_8 = time_fn(lambda: fake_quantize_tcfp8(x))
        t_12 = time_fn(lambda: fake_quantize_tcfp12(x))
        t_16 = time_fn(lambda: fake_quantize_tcfp16(x))

        print(
            f"  {str(shape):<16} {t_bf16:<12.3f} {t_8:<12.3f} "
            f"{t_12:<12.3f} {t_16:<12.3f}"
        )


# ── Benchmark 3: FP8 Matmul Throughput ───────────────────────────────────


def benchmark_matmul_throughput() -> None:
    print("\n" + "=" * 70)
    print("  BENCHMARK 3: Matmul Throughput (ms per call)")
    print("=" * 70)

    shapes = [
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]

    print(f"\n  {'(M,K,N)':<24} {'BF16 mm':<12} {'FP8 TC':<12} {'TCFP-16 3x':<12}")
    print(f"  {'-'*24} {'-'*12} {'-'*12} {'-'*12}")

    for M, K, N in shapes:
        a = torch.randn(M, K, device=DEVICE)
        b = torch.randn(K, N, device=DEVICE)

        # BF16
        a_bf = a.bfloat16()
        b_bf = b.bfloat16()
        t_bf16 = time_fn(lambda: a_bf @ b_bf)

        # Single FP8 tensor core
        a_fp8, sa = to_fp8_e4m3(a)
        b_fp8, sb = to_fp8_e4m3(b)
        t_fp8 = time_fn(lambda: fp8_matmul(a_fp8, b_fp8, sa, sb))

        # TCFP-16 (3× FP8)
        a_q16 = quantize_tcfp16(a)
        b_q16 = quantize_tcfp16(b)
        t_16 = time_fn(lambda: tcfp16_matmul(a_q16, b_q16))

        print(
            f"  {str((M,K,N)):<24} {t_bf16:<12.3f} {t_fp8:<12.3f} {t_16:<12.3f}"
        )


# ── Benchmark 4: Training Step (Mini MLP) ───────────────────────────────


def benchmark_training_step() -> None:
    print("\n" + "=" * 70)
    print("  BENCHMARK 4: Training Step Throughput (MLP forward+backward)")
    print("=" * 70)

    hidden = 1024
    batch = 64
    seq = 128

    class MiniMLP(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d, 4 * d)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(4 * d, d)
            self.ln = nn.LayerNorm(d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.ln(self.fc2(self.act(self.fc1(x))) + x)

    x = torch.randn(batch, seq, hidden, device=DEVICE)
    target = torch.randn_like(x)

    configs = [
        ("BF16 baseline", None),
        ("TCFP-8", TCFPMode.TCFP8),
        ("TCFP-12", TCFPMode.TCFP12),
        ("TCFP-16", TCFPMode.TCFP16),
    ]

    print(f"\n  MLP: {hidden}→{4*hidden}→{hidden}, batch={batch}, seq={seq}")
    print(f"  {'Config':<20} {'Forward+Bwd (ms)':<20} {'Params':<12}")
    print(f"  {'-'*20} {'-'*20} {'-'*12}")

    for name, mode in configs:
        model = MiniMLP(hidden).to(DEVICE)
        if mode is not None:
            convert_to_tcfp(model, mode=mode, block_size=32, skip_patterns=())
        model.train()

        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        n_params = sum(p.numel() for p in model.parameters())

        def step() -> None:
            opt.zero_grad()
            out = model(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            opt.step()

        t = time_fn(step, warmup=3, iters=20)
        print(f"  {name:<20} {t:<20.2f} {n_params:<12,}")


# ── Benchmark 5: VRAM Usage ─────────────────────────────────────────────


def benchmark_vram() -> None:
    print("\n" + "=" * 70)
    print("  BENCHMARK 5: Quantized Tensor VRAM (estimated)")
    print("=" * 70)

    # Simulate a 7B parameter model's weight storage
    param_count = 7_000_000_000  # 7B

    print(f"\n  Model size: {param_count / 1e9:.0f}B parameters")
    print(f"  {'Format':<14} {'Bits/val':<10} {'Estimated GB':<14}")
    print(f"  {'-'*14} {'-'*10} {'-'*14}")

    formats = [
        ("FP32", 32.0),
        ("BF16", 16.0),
        ("FP8", 8.0),
        ("TCFP-8", bits_per_value(TCFPMode.TCFP8)),
        ("TCFP-12", bits_per_value(TCFPMode.TCFP12)),
        ("TCFP-16", bits_per_value(TCFPMode.TCFP16)),
    ]

    for name, bpv in formats:
        gb = (param_count * bpv) / (8 * 1024**3)
        print(f"  {name:<14} {bpv:<10.2f} {gb:<14.2f}")


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("  TCFP Benchmark Suite")
    print(f"  GPU: {torch.cuda.get_device_name()}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print("=" * 70)

    benchmark_quality()
    benchmark_fake_quantize_throughput()
    benchmark_matmul_throughput()
    benchmark_training_step()
    benchmark_vram()

    print("\n" + "=" * 70)
    print("  Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
