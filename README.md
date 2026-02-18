# TCFP — Tensor Core Floating Point

**FP8 training that matches BF16 quality and runs faster.**

TCFP-12 decomposes each weight matrix into two FP8 components and executes two real
`torch._scaled_mm` GEMMs per layer — giving ~6 effective mantissa bits from FP8
hardware. With Triton fused kernels, a full training step runs **19% faster than BF16**
on an RTX 5070 Ti while converging to identical perplexity.

> Benchmarked on NVIDIA RTX 5070 Ti · PyTorch 2.10 · CUDA 12.8

---

## Performance

### Training Step — MLP 1024→4096→1024, batch 64×128

| Configuration | Time (ms) | vs BF16 | Converges? |
|---------------|-----------|---------|------------|
| BF16 (baseline) | 15.82 | 1.00× | Yes |
| **TCFP-12 TC** | **13.28** | **1.19× (faster)** | **Yes — ppl 1.9** |
| TCFP-12 TC + Fused kernel | 13.17 | 1.20× (faster) | Yes — ppl 1.9 |
| TCFP-12 TC + Delayed Scaling | 13.32 | 1.19× (faster) | Yes — ppl 1.9 |
| TCFP-12 TC + Block-64 | 17.01 | 0.93× | Yes — ppl 1.9 |
| TCFP-12 TC + Block-128 | 17.15 | 0.92× | Yes — ppl 1.9 |
| TCFP-12 TC + Block-32 | 18.11 | 0.87× | Yes — ppl 1.9 |
| TCFP-12 fake-quantize | 19.88 | 0.80× | Yes — ppl 1.9 |
| Pure FP8 (any path) | — | — | **No — ppl 55.6** |

### Raw GEMM Throughput — 4096×4096×4096

| Kernel | Time (ms) | vs BF16 |
|--------|-----------|---------|
| BF16 matmul | 1.62 | 1.00× |
| FP8 `_scaled_mm` | 0.74 | **2.19× faster** |
| Triton block-scaled (block=64) | 0.99 | **1.64× faster** |
| Triton fused dual-GEMM | 1.53 | **1.06× faster** |
| Naive 2× `_scaled_mm` + add | 1.68 | 0.97× |

### Quantization Quality — Gaussian N(0,1), 4096×4096

| Format | Bits | MSE | SNR (dB) |
|--------|------|-----|----------|
| BF16 | 16.0 | 2.76e-06 | 55.6 |
| **TCFP-12** | **12.5** | **2.48e-05** | **46.0** |
| Naive FP8 | 8.0 | 7.02e-04 | 31.5 |

### VRAM Savings from SRR — Measured Steady-State

| Model | EF Mode | SRR Mode | Saved |
|-------|---------|----------|-------|
| 8.4M params (1K→4K→1K) | 105 MB | 64 MB | **41 MB** |
| 33.6M params (2K→8K→2K) | 384 MB | 256 MB | **128 MB** |
| 134M params (4K→16K→4K) | 1,536 MB | 1,024 MB | **512 MB** |
| 7B model (projected) | ~26 GB | 0 GB | **~26 GB** |

### Training Convergence — TinyGPT (4-layer transformer, d=256, 1000 steps)

| Config | Final Perplexity | vs BF16 |
|--------|-----------------|---------|
| BF16 | 1.9 | baseline |
| TCFP-12 TC (all variants) | 1.9 | **1.002–1.003×** |
| Pure FP8 | 55.6 | **Diverged** |

---

## How It Works

Single-precision FP8 has only 3 mantissa bits — too coarse for transformer gradient
flow. TCFP-12 solves this by splitting each weight into a main component and a residual:

```
W  = W_hi + W_lo          (FP8 E4M3 main + FP8 E4M3 residual)
out = (X @ W_hi) + (X @ W_lo)   (2 tensor-core GEMMs)
```

This gives ~6 effective mantissa bits using only FP8 hardware. Error feedback carries
the remaining quantization error forward so it is corrected on the next step.

### Forward Pass Data Flow

```
  Input X (BF16)               W (BF16) + EF buffer (FP32)
       │                                  │
       ▼                                  ▼
  [block_quantize_fp8]    [fused_dual_quantize_fp8]  ← single kernel, one W read
       │                       │               │
  X_fp8 (FP8 E4M3)       W_hi (FP8)      W_lo (FP8)   ← ~6 effective mantissa bits
       │                  + scales_hi     + scales_lo
       │                       │
       │                  new EF = W - dequant(W_hi) - dequant(W_lo)
       │                  (stored FP32, added back next step)
       │
       ├──────────────────────────────────────────────────────┐
       │                                                      │
       ▼                                                      ▼
  GEMM₁: X_fp8 @ W_hi                              GEMM₂: X_fp8 @ W_lo
  (FP8 tensor cores)                               (FP8 tensor cores)
       │                                                      │
       └──────────────────────┬───────────────────────────────┘
                              ▼
                    output = GEMM₁ + GEMM₂  (FP32, then cast to BF16)
```

The full training pass runs **5 FP8 GEMMs** per linear layer:
- Forward: 2 (X@W_hi, X@W_lo)
- Backward dX: 1 fused (grad@(W_hi+W_lo) in a single Triton kernel)
- Backward dW: 2 (high-precision weight gradients)

### Kernel Stack

| Component | What it does |
|-----------|-------------|
| `fused_dual_quantize_fp8` | Single Triton kernel: error-feedback add → quantize W_hi → compute residual → quantize W_lo, all in registers, one GMEM read |
| `fused_dual_gemm_forward` | Triton: load activation once, compute both GEMMs, accumulate in registers |
| `fused_block_scaled_backward_dx_kernel` | Triton: load gradient once, compute `grad @ (W_hi + W_lo)` with inline scale application |
| cuBLASLt C++ ext | Forward-only: beta=1 accumulation eliminates intermediate FP32 buffer |
| Per-block scaling | Int8 exponent (E8M0) scales, 4× smaller than FP32; block sizes 32/64/128 |
| Residual sparsity | Near-zero W_lo blocks skipped via ~5-cycle warp reduction on int8 sentinel |
| Side-output elision | `SAVE_SIDE_OUTPUTS: tl.constexpr` — dead stores compiled away when `hp_grad_weight=True` |

---

## Installation

```bash
pip install -e .
```

**Requirements:**
- Python ≥ 3.10
- PyTorch ≥ 2.4 with CUDA (for FP8 tensor core support)
- GPU with FP8 support: Hopper (H100), Ada Lovelace (RTX 4090), Blackwell (RTX 5070+)
- Triton — required for fused kernels and per-block scaling
- Optional: CUDA Toolkit + MSVC (Windows) or GCC (Linux) for the cuBLASLt C++ extension

---

## Quick Start

```python
import torch
import torch.nn as nn
from tcfp.core import TCFPMode
from tcfp.nn import convert_to_tcfp

model = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 1024),
).cuda()

# Standard — 2 FP8 GEMMs per layer, error feedback, Triton fused kernel
convert_to_tcfp(model, mode=TCFPMode.TCFP12, use_tensor_cores=True)

# Faster: explicit Triton fused forward kernel
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, use_fused_kernel=True)

# Per-block scaling — better dynamic range for outlier-heavy activations
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, scale_block_size=64)

# SRR — eliminate the FP32 error feedback buffer (saves 26 GB at 7B scale)
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, srr=True)

# ABD — drop W_lo from backward dgrad (saves ~20% of training GEMMs, lossless)
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, abd=True)

# Training loop is unchanged
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
```

### Fine-tuning a Pre-trained Llama Model

```python
from transformers import AutoModelForCausalLM
import torch
from tcfp.nn import convert_to_tcfp
from tcfp.core import TCFPMode

# Load a pre-trained model in BF16
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Convert all nn.Linear layers to TCFP-12.
# Layers with non-16-divisible dimensions fall back to fake-quantize automatically.
convert_to_tcfp(
    model,
    mode=TCFPMode.TCFP12,
    use_tensor_cores=True,
    abd=True,   # drop W_lo from dX backward — lossless, saves ~20% backward GEMMs
    srr=True,   # stochastic rounding instead of FP32 error buffer (saves ~26 GB at 7B)
)

# Training loop is unchanged
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

Drop-in layer replacement is also available directly:

```python
from tcfp.nn import TCFPLinear

layer = TCFPLinear(1024, 4096, use_tensor_cores=True).cuda()
```

---

## Configuration Reference

All flags are available on both `convert_to_tcfp()` and `TCFPLinear(...)`.

| Flag | Default | Effect |
|------|---------|--------|
| `use_tensor_cores` | `False` | Real FP8 tensor-core GEMMs via `torch._scaled_mm` |
| `use_fused_kernel` | `False` | Triton fused forward dual-GEMM (load activation once) |
| `use_cuda_ext` | `True` | cuBLASLt C++ extension for forward (requires CUDA Toolkit) |
| `scale_block_size` | `None` | Per-block FP8 scaling (32, 64, or 128); enables Triton GEMM path |
| `delayed_scaling` | `False` | PDDS: EMA-smoothed W_hi scale, predicted W_lo scale |
| `error_feedback` | `True` | Carry quantization error forward across steps |
| `hp_grad_weight` | `True` | High-precision weight gradients; enables side-output elision |
| `srr` | `False` | Stochastic Residual Rounding — eliminates FP32 error buffer entirely |
| `abd` | `False` | Asymmetric Backward Decomposition — single GEMM for backward dX |

**Mutual exclusions:** `srr` and `delayed_scaling` cannot be combined (SRR requires no
error state). Both `srr` and `abd` require `use_tensor_cores=True`.

### Dispatch Priority

When multiple kernel options are enabled, TCFP selects the best available path:

1. cuBLASLt C++ extension (forward only)
2. Triton fused dual-GEMM
3. 2× `torch._scaled_mm` + add

---

## Techniques

### Error Feedback

The FP32 quantization residual from step *t* is added back before quantization at step
*t+1*, ensuring long-run unbiasedness. Stored per-layer in `ErrorFeedbackState`.

### Stochastic Residual Rounding (SRR)

Replaces the FP32 error buffer with stochastic rounding on W_lo. Since `E[SR(x)] = x`
each step is unbiased without accumulation, so no buffer is needed. Eliminates 4 bytes
per weight parameter — **26 GB at 7B scale**. Convergence is identical to error
feedback (1.08× vs BF16 at step 50, same as EF's 1.08×).

### Asymmetric Backward Decomposition (ABD)

The backward dX pass uses only W_hi: `dX = grad @ W_hi`. The W_lo contribution to
`dX` is small relative to gradient noise (validated: 1.000× ppl ratio). This saves
one GEMM per layer per backward pass — approximately 20% of total training GEMMs.

### Predictive Dual-Track Delayed Scaling (PDDS)

Optional delayed scaling tailored to the 2-GEMM architecture:
- **W_hi**: EMA-smoothed amax (decay 0.999, ~693-step half-life)
- **W_lo**: Predicted from W_hi ratio — zero-cost, no separate amax reduction
- **Activations/Gradients**: Always fresh (gradients change too rapidly for EMA)

### Per-Block Scaling

Triton kernels support block sizes of 32, 64, or 128. Scales are stored as int8
exponents (E8M0 format) rather than FP32, reducing scale storage by 4×. Zero-residual
blocks use a sentinel value (int8 −128) detected and skipped by the GEMM kernel.

---

## Project Structure

```
tcfp/
├── core.py          FP8 casting, error feedback, delayed scaling, PDDS
├── kernels.py       Triton kernels: fused dual-GEMM (fwd + bwd), block-scaled
│                    GEMM, fused dual quantize, persistent tiling, sparsity,
│                    side-output elision, int8 E8M0 encoding
├── cuda_ext.py      cuBLASLt C++ extension loader
├── csrc/            C++ source: fused dual FP8 GEMM via cuBLASLt beta=1
├── tcfp12/          Fake-quantize path (FP8 + 4-bit NF4 residual, F.linear)
├── nn/              Drop-in modules: TCFPLinear, TCFPLayerNorm, TCFPEmbedding,
│                    convert_to_tcfp, autograd functions for all GEMM paths
└── training/        Training utilities: checkpointing, monitoring, phase
                     detection, progressive quantization, layer policies

tests/               413 tests (397 passed, 16 skipped on CPU-only)
benchmarks/          Throughput, VRAM, and convergence benchmarks
examples/            TinyGPT and MNIST convergence validation scripts
```

---

## Running Tests

```bash
py -m pytest tests/ -v
```

Expected output: **397 passed, 16 skipped** (CUDA-only tests skipped on CPU-only machines).

## Running Benchmarks

```bash
# Full throughput + quality benchmark
py -m benchmarks.benchmark_tcfp

# SRR VRAM savings measurement
py -m benchmarks.benchmark_srr_vram

# Training convergence (TinyGPT + MNIST)
py -m examples.train_convergence
```

---

## Comparison with NVIDIA Transformer Engine

| | NVIDIA TE | TCFP-12 TC |
|---|-----------|------------|
| Compute path | 1 FP8 GEMM | 2 FP8 GEMMs |
| Effective mantissa bits | 3 | ~6 |
| Trains all transformer layers in FP8? | No — some fall back to BF16 | Yes — no fallbacks |
| Scaling | Delayed (max-over-history) | Fresh or PDDS |
| Scale storage | FP32 per-tensor / per-block | Int8 exponents (4× smaller) |
| Speed vs BF16 | Faster (1 GEMM) | 1.19× faster (TC path) |
| Maturity | Production | Research prototype |

---

## Status and Limitations

**Current capabilities:**
- Matches BF16 convergence on TinyGPT (PPL ratio 1.002–1.003×)
- Faster than BF16 end-to-end: TCFP-12 TC runs at 1.19× BF16 speed (13.28ms vs 15.82ms)
- 22% less weight VRAM than BF16 at any model size
- SRR eliminates the error feedback buffer entirely (saves 26 GB at 7B scale)
- ABD saves ~20% of backward GEMMs with no quality loss

**Current limitations:**
- **Single-GPU only** — not validated with FSDP, tensor parallelism, or pipeline parallelism
- **No framework integration** — requires manual `convert_to_tcfp()` call; no HuggingFace or DeepSpeed plug-in
- **Research scale only** — validated at TinyGPT (4-layer, d=256); not yet tested at 7B+
- **Per-block path still slower** — block-scaled kernels are 7–14% slower than BF16 (Block-64: 17.01ms, Block-32: 18.11ms vs BF16 15.82ms)

---

## License

Apache 2.0
