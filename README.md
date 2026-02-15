# TCFP — Tensor Core Floating Point

**Hardware-accelerated precision-enhanced FP8 training formats.**

TCFP uses FP8 tensor cores (available on NVIDIA Hopper, Ada Lovelace, and Blackwell GPUs) while achieving higher effective precision through residual decomposition and intelligent scaling. Three modes trade off between speed and quality:

| Mode | Bits/value | Effective mantissa | Storage vs BF16 | Use case |
|------|-----------|-------------------|-----------------|----------|
| **TCFP-8** | 8.25 | ~3 bits | 50% less | Max throughput, convergence-friendly |
| **TCFP-12** | 12.5 | ~5-6 bits | 25% less | Sweet spot: good quality, less VRAM |
| **TCFP-16** | 16.5 | ~6-7 bits | Same | Near-BF16 quality via FP8 tensor cores |

## How It Works

### TCFP-8: Enhanced Block-Scaled FP8

Standard FP8 E4M3 values with two improvements over naive FP8:

1. **NormalFloat-aware scaling** — Block scales are chosen to minimise MSE for Gaussian-distributed values (which neural network weights are), scaling to 3σ instead of the block maximum. Only ~0.3% of values exceed 3σ and get clipped, but the remaining 99.7% get better precision.

2. **Error feedback** — Accumulated quantisation error from previous training steps is added back before the next quantisation, ensuring long-term unbiased rounding.

```
Storage: [8-bit block scale] + [8-bit E4M3 value] × 32  →  8.25 bits/value
Compute: Standard FP8 tensor core matmul (zero overhead)
```

### TCFP-12: FP8 + 4-Bit Residual

Each value is stored as an FP8 main component plus a 4-bit signed residual correction:

```
x ≈ S_main × (fp8_value + S_residual × int4_correction)
```

The residual captures the quantisation error from FP8, boosting effective precision from 3 to ~5-6 mantissa bits. The 4-bit correction uses 15 uniform levels ([-7, +7]).

```
Storage: [8-bit main scale][8-bit residual scale] + [8-bit FP8][4-bit int] × 32  →  12.5 bits/value
Compute: FP8 tensor core matmul + cheap scalar correction
```

### TCFP-16: Residual FP8 (Double FP8)

Two FP8 E4M3 values per element — a high-order term and a low-order residual:

```
x ≈ S_hi × (dequant(x_hi) + S_lo × dequant(x_lo))
```

Matmul uses 3 FP8 tensor core calls:
```
C ≈ T1 + T2 + T3
T1 = A_hi @ B_hi    ← FP8 tensor core
T2 = A_hi @ B_lo    ← FP8 tensor core
T3 = A_lo @ B_hi    ← FP8 tensor core
(T4 = A_lo @ B_lo is negligible, skipped)
```

```
Storage: [8-bit hi scale][8-bit lo scale] + [8-bit hi FP8][8-bit lo FP8] × 32  →  16.5 bits/value
Compute: 3× FP8 tensor core matmuls + scalar adds
```

## Installation

```bash
pip install -e .
```

Requires:
- Python ≥ 3.10
- PyTorch ≥ 2.4.0 with CUDA (FP8 tensor core support)
- GPU with FP8 support (Hopper H100, Ada RTX 4090, Blackwell RTX 5070+)

## Quick Start

### Drop-in Module Replacement

```python
import torch.nn as nn
from tcfp.core import TCFPMode
from tcfp.nn import convert_to_tcfp

# Your existing model
model = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 1024),
    nn.LayerNorm(1024),
).cuda()

# Convert to TCFP-8 (fastest, minimal quality loss)
convert_to_tcfp(model, mode=TCFPMode.TCFP8, block_size=32)

# Or TCFP-12 for better quality
convert_to_tcfp(model, mode=TCFPMode.TCFP12, block_size=32)

# Training works normally — STE handles gradients
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### Direct Quantization

```python
from tcfp.tcfp8 import quantize_tcfp8, dequantize_tcfp8
from tcfp.tcfp12 import quantize_tcfp12, dequantize_tcfp12
from tcfp.tcfp16 import quantize_tcfp16, dequantize_tcfp16

x = torch.randn(1024, 1024, device="cuda")

# Quantize and dequantize
tcfp8 = quantize_tcfp8(x, nf_aware=True)
x_recovered = dequantize_tcfp8(tcfp8)
print(f"TCFP-8 MSE: {(x - x_recovered).pow(2).mean():.2e}")

tcfp12 = quantize_tcfp12(x)
x_recovered = dequantize_tcfp12(tcfp12)
print(f"TCFP-12 MSE: {(x - x_recovered).pow(2).mean():.2e}")

tcfp16 = quantize_tcfp16(x)
x_recovered = dequantize_tcfp16(tcfp16)
print(f"TCFP-16 MSE: {(x - x_recovered).pow(2).mean():.2e}")
```

### TCFP-16 Tensor Core Matmul

```python
from tcfp.tcfp16 import quantize_tcfp16, tcfp16_matmul

a = torch.randn(512, 1024, device="cuda")
b = torch.randn(1024, 512, device="cuda")

a_q = quantize_tcfp16(a)
b_q = quantize_tcfp16(b)

# 3× FP8 tensor core matmul
result = tcfp16_matmul(a_q, b_q)  # (512, 512)
```

## Benchmark Results

Measured on **RTX 5070 Ti** (Blackwell, 16GB VRAM), PyTorch 2.10, CUDA 12.8.

### Quantization Quality (Gaussian N(0,1), 4096×4096)

| Format | Bits | MSE | SNR (dB) | Max Error |
|--------|------|-----|----------|-----------|
| BF16 | 16.0 | 2.76e-06 | 55.6 | 1.56e-02 |
| Naive FP8 | 8.0 | 7.02e-04 | 31.5 | 1.90e-01 |
| **TCFP-8** | 8.2 | 7.24e-04 | 31.4 | 1.38e+00 |
| **TCFP-12** | 12.5 | 2.48e-05 | 46.0 | 3.12e-02 |
| **TCFP-16** | 16.5 | 3.93e-07 | 64.1 | 7.81e-03 |

Key findings:
- **TCFP-12** achieves 14.5 dB better SNR than FP8 — equivalent to ~2.4 extra mantissa bits
- **TCFP-16** surpasses BF16 quality (64 dB vs 56 dB) due to block scaling advantages
- **TCFP-8** NF-aware scaling is ~equivalent to naive FP8 on Gaussian data but worse on heavy-tailed distributions (by design — clips outliers to improve average case)

### Training Step Throughput (MLP 1024→4096→1024)

| Config | Forward+Backward (ms) | Overhead vs BF16 |
|--------|----------------------|-------------------|
| BF16 baseline | 17.4 | — |
| TCFP-8 | 18.3 | +5% |
| TCFP-12 | 35.1 | +102% |
| TCFP-16 | 44.0 | +153% |

TCFP-8 has minimal overhead since it's just a fake-quantize pass (the actual matmul is standard `F.linear`). TCFP-12 and TCFP-16 are slower due to residual computation in the forward pass.

### Matmul Throughput (ms)

| Size | BF16 | FP8 Tensor Core | TCFP-16 (3× FP8) |
|------|------|-----------------|-------------------|
| 4096³ | 3.34 | 2.30 | 17.9 |

The TCFP-16 3-term matmul includes dequantize + re-quantize overhead per term. A fused Triton kernel could significantly reduce this.

### Estimated VRAM for 7B Model

| Format | Bits/value | Weight VRAM |
|--------|-----------|-------------|
| FP32 | 32.0 | 26.1 GB |
| BF16 | 16.0 | 13.0 GB |
| **TCFP-16** | 16.5 | 13.4 GB |
| **TCFP-12** | 12.5 | 10.2 GB |
| FP8 | 8.0 | 6.5 GB |
| **TCFP-8** | 8.2 | 6.7 GB |

## Honest Assessment

**Where TCFP adds value:**
- **TCFP-8**: NF-aware scaling and error feedback improve convergence stability over naive FP8, with near-zero throughput cost (+5%). Best for fitting large models in limited VRAM.
- **TCFP-12**: A genuinely novel sweet spot — 25% less VRAM than BF16 with ~5-6 bit mantissa precision. No standard format exists at this size.
- **TCFP-16**: Achieves better-than-BF16 quality (64 vs 56 dB SNR). Useful when you need maximum precision but want to use FP8 tensor cores.

**Where TCFP has limitations:**
- **TCFP-8 on outliers**: NF-aware scaling clips to 3σ, which hurts on heavy-tailed distributions. Falls back to amax-based scaling automatically for constant blocks.
- **TCFP-16 matmul speed**: The current 3-term matmul is 5-8× slower than a single BF16 matmul because of dequant/requant overhead between FP8 calls. A fused Triton kernel is needed to make this competitive.
- **Not a replacement for BF16 cuBLAS**: The fake-quantize training path uses standard `F.linear` under the hood. The FP8 tensor core path (`fp8_matmul`, `tcfp16_matmul`) is for direct quantized inference or custom training loops.

## Project Structure

```
tcfp/
├── tcfp/
│   ├── core.py          # FP8 casting, block scales, tensor core matmul
│   ├── tcfp8/           # Enhanced block-scaled FP8
│   ├── tcfp12/          # FP8 + 4-bit residual
│   ├── tcfp16/          # Residual FP8 (double FP8)
│   └── nn/              # Drop-in modules (TCFPLinear, TCFPLayerNorm, etc.)
├── tests/               # 74 tests covering all modes
├── benchmarks/          # Quality + throughput benchmarks
└── pyproject.toml
```

## Running Tests

```bash
py -m pytest tests/ -v
```

## Running Benchmarks

```bash
py -m benchmarks.benchmark_tcfp
```

## License

Apache 2.0
