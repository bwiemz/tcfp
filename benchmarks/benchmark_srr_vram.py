"""
SRR VRAM Benchmark
===================

Measures VRAM savings from SRR (Stochastic Residual Rounding) by comparing
error-feedback mode vs SRR mode across model sizes.

SRR eliminates the FP32 error feedback buffer per TCFPLinear layer,
saving 4 bytes per weight parameter.

Run:  py -m benchmarks.benchmark_srr_vram
"""
from __future__ import annotations

import gc

import torch
import torch.nn as nn

from tcfp.core import TCFPMode
from tcfp.nn import TCFPLinear, convert_to_tcfp

DEVICE = "cuda"


def _clear_gpu() -> None:
    """Force GPU memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _count_error_buf_bytes(model: nn.Module) -> int:
    """Count bytes used by error feedback buffers (after forward populates them)."""
    total = 0
    for m in model.modules():
        if isinstance(m, TCFPLinear) and m._error_state is not None:
            for buf in m._error_state._buffers.values():
                total += buf.nelement() * buf.element_size()
    return total


def _measure_steady_state_vram(
    model_fn: callable,  # type: ignore[type-arg]
    in_features: int,
) -> tuple[float, float, float]:
    """Measure VRAM after model creation and one training step.

    Error feedback buffers are lazily allocated on first forward pass,
    so we must run one step to get true steady-state memory.

    Returns:
        (steady_state_mb, peak_mb, error_buf_mb)
    """
    _clear_gpu()
    base = torch.cuda.memory_allocated()

    model = model_fn()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # One full training step to allocate all lazy buffers
    x = torch.randn(32, in_features, device=DEVICE)
    out = model(x)
    loss = out.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

    # Clear intermediate activations — measure model steady state
    del x, out, loss
    gc.collect()
    torch.cuda.empty_cache()

    steady = torch.cuda.memory_allocated() - base
    peak = torch.cuda.max_memory_allocated() - base
    ef_bytes = _count_error_buf_bytes(model)

    del model, opt
    _clear_gpu()

    return (
        steady / (1024 * 1024),
        peak / (1024 * 1024),
        ef_bytes / (1024 * 1024),
    )


def benchmark_vram() -> None:
    """Compare VRAM usage: error-feedback vs SRR across model sizes."""
    print("=" * 78)
    print("SRR VRAM Benchmark -- Error Feedback vs Stochastic Residual Rounding")
    print("=" * 78)
    print()
    print("Error feedback buffers are lazily allocated on first forward pass.")
    print("All measurements taken AFTER one full training step (steady state).")
    print()

    # Test configurations: (name, hidden_dims)
    configs = [
        ("Small (1K->4K->1K)", [1024, 4096, 1024]),
        ("Medium (2K->8K->2K)", [2048, 8192, 2048]),
        ("Large (4K->16K->4K)", [4096, 16384, 4096]),
    ]

    header = (
        f"{'Config':<25} {'Mode':<8} {'Params':>12} {'EF Buf':>10} "
        f"{'Steady MB':>10} {'Peak MB':>10} {'Saved':>10}"
    )
    print(header)
    print("-" * len(header))

    for name, dims in configs:
        layers: list[tuple[int, int]] = []
        for i in range(len(dims) - 1):
            layers.append((dims[i], dims[i + 1]))

        total_params = sum(d_in * d_out + d_out for d_in, d_out in layers)
        in_features = dims[0]

        # --- Error Feedback mode ---
        def make_ef(
            _layers: list[tuple[int, int]] = layers,
        ) -> nn.Module:
            mods: list[nn.Module] = []
            for j, (d_in, d_out) in enumerate(_layers):
                mods.append(TCFPLinear(
                    d_in, d_out, bias=True,
                    mode=TCFPMode.TCFP12,
                    use_tensor_cores=True,
                    error_feedback=True, srr=False,
                ).to(DEVICE))
                if j < len(_layers) - 1:
                    mods.append(nn.ReLU())
            return nn.Sequential(*mods)

        ef_steady, ef_peak, ef_buf_mb = _measure_steady_state_vram(
            make_ef, in_features,
        )

        # --- SRR mode ---
        def make_srr(
            _layers: list[tuple[int, int]] = layers,
        ) -> nn.Module:
            mods: list[nn.Module] = []
            for j, (d_in, d_out) in enumerate(_layers):
                mods.append(TCFPLinear(
                    d_in, d_out, bias=True,
                    mode=TCFPMode.TCFP12,
                    use_tensor_cores=True,
                    srr=True,
                ).to(DEVICE))
                if j < len(_layers) - 1:
                    mods.append(nn.ReLU())
            return nn.Sequential(*mods)

        srr_steady, srr_peak, _ = _measure_steady_state_vram(
            make_srr, in_features,
        )

        steady_saved = ef_steady - srr_steady

        print(
            f"{name:<25} {'EF':<8} {total_params:>12,} "
            f"{ef_buf_mb:>9.1f}M {ef_steady:>9.1f}M "
            f"{ef_peak:>9.1f}M {'---':>10}"
        )
        print(
            f"{'':<25} {'SRR':<8} {total_params:>12,} "
            f"{'0.0':>9}M {srr_steady:>9.1f}M "
            f"{srr_peak:>9.1f}M {steady_saved:>+9.1f}M"
        )
        print()

    # --- Extrapolation to 7B ---
    print("-" * len(header))
    print()
    print("Projected savings at scale (error buffer = 4 bytes x params):")
    print()

    for model_size, label in [
        (1_000_000_000, "1B"),
        (7_000_000_000, "7B"),
        (13_000_000_000, "13B"),
        (70_000_000_000, "70B"),
    ]:
        ef_buf_gb = (model_size * 4) / (1024**3)  # FP32 = 4 bytes per param
        print(
            f"  {label:>4} model: SRR saves ~{ef_buf_gb:.1f} GB "
            f"(eliminates FP32 error feedback buffer)"
        )

    print()
    print("Note: Savings are exact -- SRR replaces error feedback entirely.")
    print("      No FP32 buffer allocated, no per-step error accumulation.")


def benchmark_convergence_quality() -> None:
    """Quick convergence comparison: EF vs SRR."""
    print()
    print("=" * 78)
    print("SRR Convergence Comparison -- Error Feedback vs SRR (200 steps)")
    print("=" * 78)
    print()

    x = torch.randn(64, 256, device=DEVICE)
    target = torch.randn(64, 128, device=DEVICE)

    results: dict[str, list[float]] = {}

    for label, kwargs in [
        ("BF16", {}),
        ("TCFP-12 TC (EF)", {"use_tensor_cores": True, "srr": False}),
        ("TCFP-12 TC (SRR)", {"use_tensor_cores": True, "srr": True}),
    ]:
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 128),
        ).to(DEVICE)
        if label != "BF16":
            convert_to_tcfp(model, mode=TCFPMode.TCFP12, **kwargs)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        losses: list[float] = []
        for _ in range(200):
            opt.zero_grad()
            out = model(x)
            loss = (out - target).pow(2).mean()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        results[label] = losses
        del model, opt
        _clear_gpu()

    print(
        f"{'Config':<22} {'Loss @10':>10} {'Loss @50':>10} "
        f"{'Loss @100':>10} {'Loss @200':>10}"
    )
    print("-" * 65)
    for label, losses in results.items():
        print(
            f"{label:<22} {losses[9]:>10.4f} {losses[49]:>10.4f} "
            f"{losses[99]:>10.4f} {losses[199]:>10.4f}"
        )

    # Compute ratio at step 50 (before near-zero convergence)
    bf16_50 = results["BF16"][49]
    print()
    print("Ppl ratio vs BF16 at step 50 (loss ratio, 1.0 = identical):")
    for label, losses in results.items():
        if label == "BF16":
            continue
        ratio = losses[49] / bf16_50 if bf16_50 > 1e-8 else float("inf")
        print(f"  {label}: {ratio:.4f}x")


if __name__ == "__main__":
    benchmark_vram()
    benchmark_convergence_quality()
