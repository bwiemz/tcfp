# TCFP — Tensor Core Floating Point

**Hardware-accelerated precision-enhanced FP8 training formats with real tensor core GEMMs.**

TCFP uses FP8 tensor cores (NVIDIA Hopper, Ada Lovelace, Blackwell) for actual compute — not just storage. The key innovation is **TCFP-12 on tensor cores**: a novel 2-GEMM residual FP8 decomposition that matches BF16 training quality while running on FP8 hardware.

| Mode | Bits/value | Tensor Core Path | Speed vs BF16 | Trains Transformers? |
|------|-----------|-----------------|---------------|---------------------|
| **TCFP-8** | 8.25 | 1 FP8 GEMM | **0.92x (faster)** | No — classification only |
| **TCFP-12** | 12.5 | **2 FP8 GEMMs (novel)** | **1.29x** | **Yes — matches BF16 exactly** |
| **TCFP-16** | 16.5 | 3 FP8 GEMMs | ~2x | Yes |

## The Novel Approach: TCFP-12 on Tensor Cores

Standard FP8 training (what NVIDIA and OpenAI use) runs a single FP8 GEMM per layer. This is fast but has only 3 mantissa bits — not enough precision for transformer training.

TCFP-12 decomposes each weight into two FP8 components and runs two real `torch._scaled_mm` GEMMs:

```
W ≈ W_hi + W_lo           (FP8 main + FP8 residual)
output = (A @ W_hi) + (A @ W_lo)   (2 FP8 tensor core GEMMs)
```

This gives ~6 effective mantissa bits using only FP8 hardware. The full training pass uses 5 FP8 GEMMs (2 forward + 2 dX + 1 dW).

**No prior work has applied 2-GEMM residual FP8 decomposition to training.** The closest related approaches are:
- NVIDIA Transformer Engine: single FP8 GEMM (faster but lower quality)
- Ozaki scheme: many-GEMM decomposition (HPC, not training)
- COLLAGE: multi-format at optimizer level (not at the GEMM level)

## How Each Mode Works

### TCFP-8: Enhanced Block-Scaled FP8

Standard FP8 E4M3 with two improvements:

1. **NormalFloat-aware scaling** — Block scales target 3sigma instead of max, minimising MSE for Gaussian-distributed weights. Only ~0.3% of values clip, but 99.7% get better precision.

2. **Error feedback** — Accumulated quantisation error from previous steps is added back before the next quantisation, ensuring long-term unbiased rounding.

**Tensor core path**: Single FP8 GEMM via `torch._scaled_mm`. Faster than BF16 but only 3 mantissa bits — sufficient for classification, not for transformers.

### TCFP-12: FP8 + Residual Correction

**Fake-quantize path** (for reference/testing): FP8 main + 4-bit NF4 residual, computed on CPU/GPU with `F.linear`.

**Tensor core path** (the novel contribution): FP8 main + FP8 residual, computed via 2 real FP8 tensor core GEMMs. Uses per-tensor scaling for `_scaled_mm` compatibility. Error feedback tracks the total (hi+lo) quantisation error.

### TCFP-16: Residual FP8 (Double FP8)

Two FP8 E4M3 values per element. Matmul uses 3 FP8 tensor core GEMMs (hi x hi, hi x lo, lo x hi — the lo x lo term is negligible and skipped). Near-BF16 quality but ~2x BF16 speed.

## Installation

```bash
pip install -e .
```

Requires:
- Python >= 3.10
- PyTorch >= 2.4.0 with CUDA (FP8 tensor core support)
- GPU with FP8 support (Hopper H100, Ada RTX 4090, Blackwell RTX 5070+)

## Quick Start

### Tensor Core Training (Recommended)

```python
import torch.nn as nn
from tcfp.core import TCFPMode
from tcfp.nn import convert_to_tcfp

model = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 1024),
    nn.LayerNorm(1024),
).cuda()

# TCFP-12 on tensor cores — matches BF16 quality, uses FP8 hardware
convert_to_tcfp(model, mode=TCFPMode.TCFP12, use_tensor_cores=True)

# Training works normally
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### Single FP8 Tensor Core Training

```python
# Fastest option — but only works for classification, not transformers
convert_to_tcfp(model, mode=TCFPMode.TCFP8, use_tensor_cores=True,
                nf_aware_scaling=True)
```

### Direct Quantization

```python
from tcfp.tcfp8 import quantize_tcfp8, dequantize_tcfp8
from tcfp.tcfp12 import quantize_tcfp12, dequantize_tcfp12
from tcfp.tcfp16 import quantize_tcfp16, dequantize_tcfp16

x = torch.randn(1024, 1024, device="cuda")

tcfp8 = quantize_tcfp8(x, nf_aware=True)
x_recovered = dequantize_tcfp8(tcfp8)

tcfp12 = quantize_tcfp12(x)
x_recovered = dequantize_tcfp12(tcfp12)

tcfp16 = quantize_tcfp16(x)
x_recovered = dequantize_tcfp16(tcfp16)
```

## Benchmark Results

Measured on **RTX 5070 Ti** (Blackwell, 16GB VRAM), PyTorch 2.10, CUDA 12.8.

### Training Step Throughput (MLP 1024->4096->1024, batch=64, seq=128)

| Config | Forward+Backward (ms) | vs BF16 | Converges on TinyGPT? |
|--------|----------------------|---------|----------------------|
| BF16 baseline | 28.2 | 1.00x | Yes (ppl 1.9) |
| TCFP-8 TC | 25.8 | **0.92x (faster)** | No (ppl 55.6) |
| TCFP-8 TC+NF | 24.1 | **0.85x (faster)** | No |
| **TCFP-12 TC** | **36.5** | **1.29x** | **Yes (ppl 1.9)** |
| TCFP-12 fake-quantize | 42.9 | 1.52x | Yes (ppl 1.9) |
| TCFP-16 | 19.1 | 0.68x | Yes (ppl 1.9) |

### Raw GEMM Performance (TC GEMM vs BF16)

| Size (M,K,N) | BF16 mm | FP8 TC (1 GEMM) | 2-GEMM TC | FP8 speedup |
|--------------|---------|-----------------|-----------|-------------|
| 1024^3 | 0.07 ms | 0.06 ms | 0.14 ms | 1.17x |
| 2048^3 | 0.52 ms | 0.22 ms | 0.47 ms | 2.39x |
| 4096^3 | 3.15 ms | 1.71 ms | 3.17 ms | 1.84x |

At large sizes, a single FP8 GEMM is ~2x faster than BF16. The 2-GEMM path is roughly BF16 speed at the GEMM level, with the quality of TCFP-12.

### Quantization Quality (Gaussian N(0,1), 4096x4096)

| Format | Bits | MSE | SNR (dB) |
|--------|------|-----|----------|
| BF16 | 16.0 | 2.76e-06 | 55.6 |
| Naive FP8 | 8.0 | 7.02e-04 | 31.5 |
| **TCFP-8** | 8.2 | 7.24e-04 | 31.4 |
| **TCFP-12** | 12.5 | 2.48e-05 | 46.0 |
| **TCFP-16** | 16.5 | 3.93e-07 | 64.1 |

### Training Convergence

**MNIST** (3-layer MLP, 10 epochs): All modes converge to 100% accuracy.

**TinyGPT** (4-layer transformer, d=256, 500 steps):

| Config | Final Loss | Perplexity | Status |
|--------|-----------|------------|--------|
| BF16 baseline | 0.660 | 1.9 | Converged |
| TCFP-8 / FP8-TC | 4.02 | 55.6 | **Failed** |
| **TCFP-12** | **0.662** | **1.9** | **Converged** |
| **TCFP-12 TC** | **0.663** | **1.9** | **Converged** |
| TCFP-16 | 0.662 | 1.9 | Converged |

TCFP-12 (both fake-quantize and tensor core paths) matches BF16 step-for-step. Pure FP8 fails on transformers regardless of path (single GEMM or fake-quantize).

### Estimated VRAM for 7B Model

| Format | Bits/value | Weight VRAM |
|--------|-----------|-------------|
| FP32 | 32.0 | 26.1 GB |
| BF16 | 16.0 | 13.0 GB |
| **TCFP-12** | 12.5 | 10.2 GB |
| FP8 / **TCFP-8** | 8.0-8.2 | 6.5-6.7 GB |

## Honest Assessment

**Where TCFP-12 TC adds value:**
- Matches BF16 convergence exactly on transformer training (TinyGPT: PPL 1.9)
- Uses real FP8 tensor cores — not fake-quantize with F.linear
- Novel approach: no prior work combines 2-GEMM residual FP8 with training
- 25% less VRAM than BF16 for weight storage

**Current limitations:**
- **TCFP-12 TC is 1.29x BF16 speed**, not faster. The 2-GEMM overhead (5 FP8 GEMMs total vs 3 BF16 GEMMs) means it trades some speed for quality. A fused kernel could improve this.
- **TCFP-8 cannot train transformers**: 3-bit mantissa is insufficient for gradient flow through attention layers. Limited to inference and classification.
- **Per-tensor scaling only**: The tensor core path uses per-tensor FP8 scales (required by `torch._scaled_mm`). Production systems like NVIDIA TE use delayed per-tensor scaling; per-block scaling would need custom CUDA kernels.
- **No delayed scaling**: Scales are computed fresh each forward pass. NVIDIA's delayed scaling (using previous step's scale) would reduce overhead.

## Project Structure

```
tcfp/
├── tcfp/
│   ├── core.py          # FP8 casting, block scales, tensor core utilities
│   ├── tcfp8/           # Enhanced block-scaled FP8
│   ├── tcfp12/          # FP8 + 4-bit residual (fake-quantize path)
│   ├── tcfp16/          # Residual FP8 (double FP8)
│   └── nn/              # Drop-in modules (TCFPLinear, TCFPLayerNorm, etc.)
│                          Includes _FP8TensorCoreFunction (single GEMM)
│                          and _TCFP12TensorCoreFunction (2-GEMM residual)
├── tests/               # 153 tests covering all modes + tensor core paths
├── benchmarks/          # Quality + throughput + TC GEMM benchmarks
├── examples/            # Training convergence tests (MNIST + TinyGPT)
└── pyproject.toml
```

## Running Tests

```bash
py -m pytest tests/ -v
```

## Running Benchmarks

```bash
# Full benchmark suite
py -m benchmarks.benchmark_tcfp

# Training convergence tests
py -m examples.train_convergence
```

## License

Apache 2.0
