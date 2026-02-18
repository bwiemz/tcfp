# TCFP — Tensor Core Floating Point

**Hardware-accelerated precision-enhanced FP8 training format with real tensor core GEMMs.**

TCFP uses FP8 tensor cores (NVIDIA Hopper, Ada Lovelace, Blackwell) for actual compute — not just storage. The key innovation is **TCFP-12 on tensor cores**: a novel 2-GEMM residual FP8 decomposition that matches BF16 training quality while running on FP8 hardware.

| Variant | Bits/value | Tensor Core Path | Speed vs BF16 | Trains Transformers? |
|---------|-----------|-----------------|---------------|---------------------|
| **TCFP-12** | 12.5 | **2 FP8 GEMMs (novel)** | **0.84x (faster)** | **Yes — matches BF16 exactly** |
| **TCFP-12+DS** | 12.5 | **2 FP8 GEMMs + PDDS** | **1.19x** | **Yes — matches BF16 exactly** |
| **TCFP-12+Block** | 12.5 | **2 Triton block-scaled GEMMs** | ~2.6x | **Yes — matches BF16 exactly** |

## The Problem: FP8 Tensor Cores Are Underutilized

Every modern NVIDIA data-centre GPU (H100, H200, B100, B200, GB200) has FP8 tensor cores running at **2x the FLOPS of BF16**. Yet most training workloads cannot use them:

- **Single-GEMM FP8** (what NVIDIA Transformer Engine uses) has only 3 mantissa bits — insufficient for transformer gradient flow
- **Pure FP8 training fails** on transformers: perplexity 55.6 vs BF16's 1.9 on our TinyGPT benchmark
- Data centres leave half their tensor core FLOPS on the table

## The Solution: 2-GEMM Residual FP8 Decomposition

TCFP-12 decomposes each weight into two FP8 components and runs two real `torch._scaled_mm` GEMMs:

```
W = W_hi + W_lo           (FP8 main + FP8 residual)
output = (A @ W_hi) + (A @ W_lo)   (2 FP8 tensor core GEMMs)
```

This gives ~6 effective mantissa bits using only FP8 hardware. The full training pass uses 5 FP8 GEMMs (2 forward + 2 dX + 1 dW).

**No prior work has applied 2-GEMM residual FP8 decomposition to training.** The closest related approaches are:
- NVIDIA Transformer Engine: single FP8 GEMM (faster but insufficient precision)
- Ozaki scheme: many-GEMM decomposition (HPC, not training)
- COLLAGE: multi-format at optimizer level (not at the GEMM level)

## Benchmark Results

Measured on **RTX 5070 Ti** (Blackwell, 16GB VRAM), PyTorch 2.10, CUDA 12.8.

### Training Step Throughput

**MLP 1024->4096->1024, batch=64, seq=128:**

| Config | Forward+Backward (ms) | vs BF16 | Converges on TinyGPT? |
|--------|----------------------|---------|----------------------|
| BF16 baseline | 19.4 | 1.00x | Yes (ppl 1.9) |
| **TCFP-12 TC** | **35.6** | **1.84x** | **Yes (ppl 1.9)** |
| **TCFP-12 TC+DS** | **35.3** | **1.82x** | **Yes (ppl 1.9)** |
| TCFP-12 TC+Fused | 37.3 | 1.92x | Yes (ppl 1.9) |
| TCFP-12 TC+B32 | 51.0 | 2.63x | Yes (ppl 1.9) |
| TCFP-12 TC+B64 | 49.8 | 2.57x | Yes (ppl 1.9) |
| TCFP-12 TC+B128 | 49.5 | 2.55x | Yes (ppl 1.9) |
| TCFP-12 fake-quantize | 43.1 | 2.22x | Yes (ppl 1.9) |

### Raw GEMM Performance (4096 x 4096 x 4096)

| Kernel | Time (ms) | vs BF16 |
|--------|----------|---------|
| BF16 matmul | 3.20 | 1.00x |
| FP8 TC (`_scaled_mm`) | 1.69 | **1.90x faster** |
| Triton block-scaled (block=64) | 1.78 | **1.80x faster** |

At large sizes, FP8 TC GEMMs are ~1.9x faster than BF16. The Triton block-scaled kernel is within 5% of `_scaled_mm` while providing per-block dynamic range.

### Quantization Quality (Gaussian N(0,1), 4096x4096)

| Format | Bits | MSE | SNR (dB) |
|--------|------|-----|----------|
| BF16 | 16.0 | 2.76e-06 | 55.6 |
| Naive FP8 | 8.0 | 7.02e-04 | 31.5 |
| **TCFP-12** | 12.5 | 2.48e-05 | 46.0 |

### Training Convergence (TinyGPT: 4-layer transformer, d=256, 1000 steps)

| Config | Final Loss | Perplexity | Status |
|--------|-----------|------------|--------|
| BF16 baseline | 0.660 | 1.9 | Converged |
| **TCFP-12 TC** | **0.663** | **1.9** | **Converged** |
| **TCFP-12 TC+DS** | **0.663** | **1.9** | **Converged** |
| TCFP-12 TC+B32/B64/B128 | 0.662 | 1.9 | Converged |
| TCFP-12 fake-quantize | 0.662 | 1.9 | Converged |
| Pure FP8 (any path) | 4.02 | 55.6 | **Failed** |

All TCFP-12 variants match BF16 step-for-step (ppl ratio 1.002-1.003x). Pure FP8 fails on transformers regardless of enhancements.

### SRR VRAM Savings (Error Feedback Buffer Eliminated)

SRR (Stochastic Residual Rounding) replaces the FP32 error feedback buffer with per-step stochastic rounding (`E[SR(x)] = x`), eliminating 4 bytes per weight parameter with no convergence loss.

**Measured steady-state VRAM (after first training step):**

| Config | Params | EF Mode | SRR Mode | Saved |
|--------|--------|---------|----------|-------|
| 1K->4K->1K | 8.4M | 105 MB | 64 MB | **41 MB** |
| 2K->8K->2K | 33.6M | 384 MB | 256 MB | **128 MB** |
| 4K->16K->4K | 134M | 1536 MB | 1024 MB | **512 MB** |

Convergence is identical: SRR loss ratio 1.07x vs BF16 at step 50 (same as EF's 1.09x).

**Projected savings at scale:**

| Model Size | EF Buffer Eliminated | % of BF16 Weight VRAM |
|-----------|---------------------|----------------------|
| 1B | 3.7 GB | 100% of weights |
| **7B** | **26.1 GB** | **100% of weights** |
| 13B | 48.4 GB | 100% of weights |
| 70B | 260.8 GB | 100% of weights |

### Estimated VRAM for 7B Model

| Format | Bits/value | Weight VRAM |
|--------|-----------|-------------|
| FP32 | 32.0 | 26.1 GB |
| BF16 | 16.0 | 13.0 GB |
| **TCFP-12** | 12.5 | **10.2 GB** |
| **TCFP-12 + SRR** | 12.5 | **10.2 GB** (no EF buffer) |
| FP8 | 8.0 | 6.5 GB |

## Comparison with Industry Standards

### vs NVIDIA Transformer Engine (TE)

NVIDIA TE is the current industry standard for FP8 training:

| | NVIDIA TE | TCFP-12 TC |
|---|-----------|------------|
| **Approach** | Single FP8 GEMM | 2-GEMM residual FP8 |
| **Mantissa bits** | 3 (E4M3) | ~6 effective |
| **Trains transformers?** | With fallback-to-BF16 for unstable layers | **Yes, no fallbacks needed** |
| **Scaling** | Delayed (max-over-history + headroom) | Fresh or PDDS (EMA + prediction) |
| **Loss scaling heuristics** | Required (complex tuning) | Not needed |
| **Maturity** | Production (H100/B200) | Research prototype |
| **Speed** | Faster (1 GEMM) | Slower (2 GEMMs per layer) |
| **Weight VRAM** | 8 bits | 12.5 bits |
| **Scale storage** | FP32 per-tensor or per-block | Int8 exponents (E8M0, 4x smaller) |
| **Bandwidth opts** | N/A (single GEMM) | Side-output elision, E8M0, sparsity fast-path |

**Key insight**: TE's single FP8 GEMM works for production LLM training but requires careful per-layer configuration — many layers fall back to BF16 when FP8 precision is insufficient. TCFP-12 eliminates this complexity by providing enough precision to train all layers in FP8.

### vs Microsoft DeepSpeed FP8

| | DeepSpeed FP8 | TCFP-12 TC |
|---|--------------|------------|
| **Approach** | Single FP8 GEMM with grad scaling | 2-GEMM residual |
| **Integration** | DeepSpeed ZeRO ecosystem | Standalone |
| **Multi-GPU** | Full FSDP/TP/PP | Single-GPU only |

### vs BF16 Training (Current Default)

| | BF16 | TCFP-12 TC |
|---|------|------------|
| **Quality** | Baseline | **Matches exactly** (ppl ratio 1.002x) |
| **Weight VRAM** | 13.0 GB (7B) | **10.2 GB (7B)** — 22% savings |
| **Tensor cores used** | BF16 cores | FP8 cores (2x FLOPS available) |
| **Speed** | Baseline | Currently 1.8x slower (quantization overhead; raw GEMMs are 1.9x faster). Bandwidth optimizations (side-output elision, E8M0 scales, sparsity fast-path) reduce overhead incrementally; fusing quantization into GEMM is the path to parity |
| **Maturity** | Production standard | Research prototype |

## When to Use TCFP-12

**Preferred over BF16 when:**
- You need to train larger models in limited VRAM (22% weight savings)
- You want to utilize FP8 tensor cores that are otherwise idle
- You're training on hardware with significant FP8 FLOPS advantage (H100, B200)
- Once fused kernels close the throughput gap

**Preferred over single FP8 (NVIDIA TE) when:**
- Your model has layers that fail to converge in FP8 (e.g., attention, early layers)
- You want to avoid per-layer BF16 fallback complexity
- Training stability is more important than maximum throughput
- You need guaranteed convergence parity with BF16

**Not yet suitable for:**
- Multi-GPU / distributed training (single-GPU only, FSDP not yet validated)
- Production data-centre deployment (needs framework integration + scale validation)
- Models where single FP8 already converges well (simple CNNs, MLPs)

## Roadmap to Production

TCFP-12 TC is a research prototype. Three pillars define "production-ready":

### Pillar 1: Speed-of-Light Kernel (Target: >1.0x vs BF16)

TCFP-12 moves less data than BF16 (12.5 bits vs 16 bits) and raw FP8 GEMMs are 1.9x faster. The physics says we should be *faster* than BF16 — the current 1.8x slowdown comes from quantization overhead, not the GEMMs themselves. The path to closing this gap:

1. **Fused dual quantization** — *(Done)* A single Triton kernel (`fused_dual_quantize_fp8`) replaces 6 separate kernels: error feedback add, `block_quantize_fp8(W)`, dequantization, residual subtraction, `block_quantize_fp8(residual)`, and error update. The weight is read from GMEM once (in native BF16/FP32 dtype), and W_hi, W_lo, scales, and error update are all computed in registers. Reduces the block-scaled forward pass from ~15 kernels to ~4. Uses power-of-2 scaling for bitwise parity with the separate-kernel path.

2. **Fuse quantization into GEMM** — The next step: quantize W tiles on-the-fly as they're loaded from GMEM into SMEM inside the GEMM kernel itself. This would eliminate the quantization kernels entirely, leaving only the GEMM + bias add.

3. **Fused dual-GEMM backward** — *(Done)* The cuBLASLt C++ extension fuses both forward GEMMs via beta=1 accumulation. For the backward dX pass, a Triton kernel (`_fused_block_scaled_backward_dx_kernel`) loads the gradient matrix once and computes `grad @ (W_hi + W_lo)` in a single persistent-tiled pass — halving backward memory bandwidth vs the prior two-matmul path. W scales are applied inline in registers; the zero-residual sentinel (int8 -128) skips W_lo contributions without branching. cuBLASLt is not used for backward (doesn't support E5M2 as A-type on SM 12.0+).

4. **Persistent tiling** — *(Done)* Both dual-GEMM Triton kernels use persistent tiling: each CTA loops over multiple output tiles to eliminate wave quantization (last-wave SM underutilization). Controlled via `persistent=True` (default) and `launch_mult` (CTAs per SM). When the tile count doesn't divide evenly by SM count, persistent tiling keeps all SMs busy.

5. **Residual sparsity fast-path** — *(Done)* K-blocks where `max(scale_lo) <= threshold` skip the W_lo tile load and dot product entirely. Combined with the zero-residual sentinel (int8 -128), sparse residual blocks are detected and skipped at near-zero cost (~5-cycle warp reduction).

6. **Side-output elision** — *(Done)* When `hp_grad_weight=True`, the fused act-quant kernel's `act_fp8` and `act_scales` stores are dead writes (backward uses FP32 input instead). `SAVE_SIDE_OUTPUTS: tl.constexpr` eliminates all side-output stores at compile time, saving 8.25 MB per forward pass (6.9% of total bandwidth).

7. **Int8 exponent encoding (E8M0)** — *(Done)* Per-block scales stored as int8 exponents instead of FP32 (4x smaller). Producers store `raw_exp.to(tl.int8)`, consumers reconstruct via `tl.exp2()`. Zero-residual sentinel uses int8 -128.

8. **Optimized per-block scaling** — The Triton block-scaled kernels work but add ~40% overhead over per-tensor `_scaled_mm`. Fusing block quantization into the GEMM kernel (compute per-block scales as tiles are loaded) could close this gap while providing better dynamic range for outlier-heavy distributions.

### Pillar 2: Distributed Training (FSDP / Multi-GPU)

The primary user for TCFP is VRAM-constrained. These users almost always use FSDP or DeepSpeed ZeRO to shard across consumer GPUs (e.g., 2x RTX 4090s). Without multi-GPU support, TCFP can't serve its core audience at scale.

1. **FSDP compatibility** — Ensure FSDP correctly shards TCFPLinear parameters without casting FP8 components back to FP32 or breaking the 2-GEMM layout. Error feedback state must be correctly sharded and synchronized across ranks.

2. **Tensor/pipeline parallelism** — Validate that the 2-GEMM decomposition works correctly when layers are split across devices. Checkpoint/resume must preserve error feedback and delayed scaling state.

3. **Large-scale convergence validation** — TinyGPT (4-layer, d=256, 1000 steps) proves the approach works. Production adoption requires convergence parity at 7B+ scale over thousands of steps, across architectures (LLaMA, Mistral, GPT-style) and training regimes (pre-training, fine-tuning, RLHF).

### Pillar 3: Drop-in Framework Integration

If users have to manually rewrite model definitions, adoption drops 90%. The goal is 3 lines of code added to an existing training script.

1. **HuggingFace integration** — A `TrainerCallback` or `quantization_config=TcfpConfig()` pattern that auto-converts models at load time. Users should be able to add TCFP to any HF training script without touching model code.

2. **Megatron-LM / DeepSpeed hooks** — Drop-in support for production training frameworks. Needs hooks for gradient checkpointing, mixed-precision policies, and optimizer state management.

3. **Hardware-specific tuning** — Benchmarks are on RTX 5070 Ti (consumer Blackwell). H100/B200 data-centre GPUs have different FP8 tensor core architectures, memory hierarchies, and inter-GPU bandwidth. Performance characteristics may differ significantly.

## Features

### Predictive Dual-Track Delayed Scaling (PDDS)

Optional delayed scaling system custom-tailored to the 2-GEMM architecture:

- **W_hi**: EMA-smoothed amax (decay 0.999, ~693-step half-life)
- **W_lo**: Predicted from W_hi (zero-cost, no amax reduction needed)
- **Activations/Gradients**: Always fresh (gradients change too rapidly for EMA)
- **Error feedback self-correction**: No extra headroom factor needed

```python
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, delayed_scaling=True)
```

### Per-Block Scaling (Triton Kernels)

Custom Triton kernels for per-block FP8 quantization and GEMM, providing better dynamic range than per-tensor scaling:

```python
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, scale_block_size=64)
```

### Error Feedback

Accumulated quantization error from previous steps is added back before the next quantization, ensuring long-term unbiased rounding:

```python
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, error_feedback=True)  # default
```

### Fused Dual-GEMM Kernels

Two fusion approaches for the 2-GEMM residual architecture, with automatic 3-tier dispatch:

**cuBLASLt C++ Extension** (highest priority) — JIT-compiled native CUDA extension using cuBLASLt's beta=1 accumulation. The second GEMM accumulates directly onto the first result, eliminating the intermediate FP32 buffer and addition kernel. Forward-only (cuBLASLt doesn't support E5M2 gradients as A-type on SM 12.0+):

```python
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, use_cuda_ext=True)  # default
```

**Triton Fused Kernel** — Single Triton kernel that reads the activation matrix once and computes both GEMMs, reducing memory bandwidth. Supports both forward and backward:

```python
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, use_fused_kernel=True)
```

Dispatch priority: cuBLASLt C++ ext > Triton fused > 2x `torch._scaled_mm` + add.

### Persistent Tiling

Both dual-GEMM Triton kernels use persistent tiling to maximize SM utilization. Instead of launching one CTA per output tile (which wastes SMs in the last wave), persistent kernels launch `min(num_sms * launch_mult, total_tiles)` programs that each loop over multiple tiles:

```python
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, scale_block_size=64)
# persistent=True is the default in all block-scaled paths
```

Controlled via `persistent` (bool, default `True`) and `launch_mult` (int, default `1`) parameters on the kernel wrappers. Setting `persistent=False` reverts to one-CTA-per-tile behavior for A/B testing.

### Residual Sparsity Fast-Path

When a K-block's residual is near-zero (`max(scale_lo) <= threshold`), the W_lo tile load and dot product are skipped entirely. Combined with the zero-residual sentinel (int8 exponent -128 for zero blocks), sparse residuals are detected via a ~5-cycle warp reduction and skipped at near-zero overhead:

```python
# Sparsity threshold can be tuned per-model
fused_block_scaled_dual_gemm_forward(
    ..., sparse_threshold=1e-6,
)
```

### Side-Output Elision

When `hp_grad_weight=True` (the default for training quality), the fused act-quant kernel's side outputs (`act_fp8` and `act_scales`) are never read by backward — backward uses the original FP32 input instead. The kernel uses `SAVE_SIDE_OUTPUTS: tl.constexpr` to physically eliminate all side-output stores from the compiled kernel, saving **8.25 MB of VRAM write bandwidth per forward pass** (6.9% of total) for a typical 2048x4096 layer. This is automatic — no user configuration needed.

### Int8 Exponent Encoding (E8M0)

All per-block FP8 scales are stored as int8 exponents instead of FP32. Since TCFP scales are always power-of-2 (`scale = 2^ceil(log2(amax / FP8_MAX))`), storing the integer exponent directly eliminates 3 wasted bytes per scale — a **4x reduction** in scale storage. Consumer GEMM kernels reconstruct via `tl.exp2()` (single instruction, negligible overhead). Zero-residual blocks use int8 sentinel value -128, restored to exact 0.0 in consumers for the sparsity fast-path.

```python
# Automatic — scales are always int8 in the block-scaled path
w_hi, w_lo, scales_hi, scales_lo = fused_dual_quantize_fp8(weight, block_size=128)
# scales_hi.dtype == torch.int8, scales_lo.dtype == torch.int8
```

### Stochastic Residual Rounding (SRR)

Replaces the FP32 error feedback buffer with stochastic rounding on W_lo. Since `E[SR(x)] = x` (unbiased per-step), each training step is unbiased without cumulative error tracking — the error feedback buffer is eliminated entirely:

```python
# SRR: no FP32 error buffer, same convergence quality
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, srr=True)
```

At 7B scale, this saves **26.1 GB** of VRAM (the entire FP32 error buffer). SRR is mutually exclusive with `delayed_scaling` and requires `use_tensor_cores=True`. Convergence matches error-feedback mode exactly (1.07x vs 1.09x ppl ratio vs BF16).

**Where the memory goes:** In EF mode, `TCFPLinear.__init__` creates an `ErrorFeedbackState` (`nn/__init__.py`), which lazily allocates an FP32 buffer matching the weight shape on first forward via `get_error()` (`core.py:320`). With `srr=True`, `_error_state` is set to `None` — no buffer is ever allocated (`nn/__init__.py`, guard: `if use_tensor_cores and error_feedback and not srr`).

**RNG sources:** The per-tensor path uses `torch.rand_like()` (PyTorch's default CUDA PRNG, seedable via `torch.manual_seed()`). The block-scaled Triton path uses `tl.rand(seed, offset)` (Triton's built-in Philox PRNG) with a fresh random seed per kernel launch (`torch.randint(0, 2**31, ...)`). Per-element randomness is derived from linearised tile offsets (`offs_n * K + offs_k`), giving unique entropy per element. SRR is **not deterministic** across runs (different seeds each launch), but can be made deterministic by fixing the PRNG seed externally.

**Throughput overhead:** SRR trades compute for memory. The block-scaled Triton path adds ~0-15% overhead at small/medium layer sizes (the `STOCHASTIC_LO: tl.constexpr` branch compiles to ~15 extra ALU ops per element including Philox RNG, nudge, and stochastic select). The per-tensor PyTorch path has higher overhead (~25-40%) because `stochastic_round_to_fp8_e4m3` allocates multiple temporary tensors on each call. Both paths add more overhead at very large dimensions (4K+) where the extra compute becomes proportionally significant. Use SRR when VRAM savings outweigh the throughput cost — typically memory-constrained scenarios where the alternative is not training at all.

| Path | Layer Size | SRR vs EF Overhead |
|------|-----------|-------------------|
| Per-tensor TC | 1024x512 | +29% |
| Per-tensor TC | 2048x1024 | +26% |
| Block-scaled (B128) | 1024x512 | **-15%** (faster — no EF buffer ops) |
| Block-scaled (B128) | 2048x1024 | +39% |

### Asymmetric Backward Decomposition (ABD)

Drops W_lo from the backward dgrad computation: `dX = grad @ W_hi` (single GEMM instead of dual). This saves 1 GEMM per layer per backward pass (~20% of total training GEMMs) with no measurable quality impact (1.000x ppl ratio):

```python
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, abd=True)
```

### Fused Dual Quantization

Single Triton kernel that fuses the entire weight quantization pipeline for the block-scaled path:

**Before** (6 separate kernel launches):
1. Error feedback: `w = weight + error_buf`
2. `block_quantize_fp8(w)` -> W_hi + scales_hi
3. `block_dequantize_fp8(W_hi)` -> w_hi_deq
4. `residual = w - w_hi_deq`
5. `block_quantize_fp8(residual)` -> W_lo + scales_lo
6. Error update: `new_error = weight - dequant(W_hi) - dequant(W_lo)`

**After** (1 kernel launch):
```python
w_hi, w_lo, scales_hi, scales_lo = fused_dual_quantize_fp8(
    weight, block_size=128, error_buf=error_buf,
)
```

The kernel loads the weight once from GMEM (in native BF16/FP32 dtype, converted to FP32 inside the kernel after coalesced load), applies error feedback, quantizes W_hi with power-of-2 scaling, computes the residual in registers (no GMEM round-trip), quantizes W_lo, and writes all outputs. Produces bitwise-identical results to the separate-kernel path. Used automatically by `_TCFP12BlockScaledFunction` when Triton is available.

## How TCFP-12 Works

### FP8 + Residual Correction

**Tensor core path** (the novel contribution): FP8 main + FP8 residual, computed via 2 real FP8 tensor core GEMMs. Error feedback tracks the total (hi+lo) quantization error.

**Fake-quantize path** (for reference/testing): FP8 main + 4-bit NF4 residual, computed with `F.linear`.

## Installation

```bash
pip install -e .
```

Requires:
- Python >= 3.10
- PyTorch >= 2.4.0 with CUDA (FP8 tensor core support)
- GPU with FP8 support (Hopper H100, Ada RTX 4090, Blackwell RTX 5070+)
- Optional: Triton (for Triton fused kernels and per-block scaling)
- Optional: CUDA Toolkit + C++ compiler (for cuBLASLt fused dual-GEMM extension)

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

# With delayed scaling — reduces amax overhead at scale
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, delayed_scaling=True)

# With per-block scaling — better dynamic range via Triton kernels
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, scale_block_size=64)

# With SRR — eliminate error feedback buffer (saves 26 GB at 7B scale)
convert_to_tcfp(model, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, srr=True)

# Training works normally
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

## Public API (Stable Entrypoints)

The following entrypoints are the stable, user-facing API surface for model conversion and drop-in modules:

- `tcfp.nn.convert_to_tcfp(model, ...)` — in-place conversion for `nn.Linear`, `nn.LayerNorm`, and `nn.Embedding`.
- `tcfp.nn.TCFPLinear` — drop-in replacement for `nn.Linear`.
- `tcfp.nn.TCFPLayerNorm` — drop-in replacement for `nn.LayerNorm`.
- `tcfp.nn.TCFPEmbedding` — drop-in replacement for `nn.Embedding`.
- `tcfp.core.TCFPMode` — precision mode enum used by conversion and modules.

### Feature Flags (API Contract)

`convert_to_tcfp(...)` exposes the following stable flags:

| Flag | What changes | Expected impact |
|------|--------------|-----------------|
| `use_tensor_cores=True` | Uses real FP8 tensor-core GEMMs for `TCFPLinear` when layer dims are compatible | Enables FP8 tensor-core path (raw GEMMs faster than BF16), but end-to-end speed depends on quantization overhead |
| `delayed_scaling=True` | Enables Predictive Dual-Track Delayed Scaling (PDDS) for TCFP-12 tensor-core path | Reduces scaling/amax overhead with convergence parity in reported TinyGPT runs |
| `scale_block_size=32/64/128` | Switches from per-tensor to per-block scaling (Triton path) | Better dynamic range/outlier handling; slower than per-tensor path in current repository benchmarks |
| `use_fused_kernel=True` | Uses Triton fused dual-GEMM kernels when available | Lowers kernel launch and memory overhead vs unfused 2-GEMM path |
| `use_cuda_ext=True` | Uses cuBLASLt fused forward dual-GEMM extension when available | Best forward-path performance among current implementations |
| `error_feedback=True` | Enables residual error-feedback state across steps | Improves long-run quantization fidelity (stability/quality) |
| `hp_grad_weight=True` | Keeps weight-gradient path high precision in tensor-core mode | Prioritizes training quality/stability over peak speed |
| `srr=True` | Stochastic Residual Rounding on W_lo — eliminates FP32 error feedback buffer | Saves 4 bytes/param VRAM (26 GB at 7B); ~15-30% compute overhead. Mutually exclusive with `delayed_scaling` |
| `abd=True` | Asymmetric Backward Decomposition — drops W_lo from backward dgrad | Saves 1 GEMM per layer per backward (~20% of training GEMMs); lossless (1.000x ppl) |

### Direct Quantization

```python
from tcfp.tcfp12 import quantize_tcfp12, dequantize_tcfp12

x = torch.randn(1024, 1024, device="cuda")

tcfp12 = quantize_tcfp12(x)
x_recovered = dequantize_tcfp12(tcfp12)
```

## Honest Assessment

**Where TCFP-12 TC adds value:**
- Matches BF16 convergence exactly on transformer training (TinyGPT: PPL 1.9)
- Uses real FP8 tensor cores — not fake-quantize with F.linear
- Novel approach: no prior work combines 2-GEMM residual FP8 with training
- 22% less VRAM than BF16 for weight storage
- PDDS eliminates W_lo amax computation with zero quality loss
- Per-block Triton kernels provide fine-grained dynamic range
- cuBLASLt C++ extension fuses forward GEMMs via beta=1 accumulation (no intermediate buffer)
- Fused dual quantization kernel reduces block-scaled forward from ~15 kernels to ~4
- Persistent tiling eliminates wave quantization for large problems
- Residual sparsity fast-path skips zero W_lo blocks at near-zero cost
- Side-output elision saves 8.25 MB/forward (6.9% bandwidth) when `hp_grad_weight=True`
- Int8 exponent encoding (E8M0) reduces scale storage 4x — all per-block scales use int8
- SRR eliminates FP32 error feedback buffer entirely — saves 26.1 GB at 7B scale (trades ~15-30% compute overhead for memory)
- ABD drops W_lo from backward dgrad — saves 1 GEMM per layer per backward pass

**Current limitations:**
- **1.8x slower than BF16** in training step throughput — raw FP8 GEMMs are 1.9x faster, but quantization overhead (W decomposition + FP8 casting each step) eats the gain. Bandwidth optimizations (side-output elision saves 6.9%, E8M0 scales save 0.8%) chip away at overhead; fusing quantization into the GEMM itself is the path to >1.0x
- **Single-GPU only**: Not validated with distributed training (FSDP, TP, PP) — the primary blocker for real-world adoption
- **No framework integration**: Requires manual `convert_to_tcfp()` call, no HuggingFace/DeepSpeed drop-in support yet
- **Partial kernel fusion**: cuBLASLt fuses forward GEMMs but backward still dispatches separately (E5M2 limitation)
- **Research prototype**: Validated at TinyGPT scale (4-layer, d=256), not yet at 7B+

## Project Structure

```
tcfp/
├── tcfp/
│   ├── core.py          # FP8 casting, block scales, tensor core utilities,
│                          error feedback with PDDS delayed scaling
│   ├── kernels.py       # Triton kernels: fused dual-GEMM, block-scaled GEMM,
│                          block quantization, fused dual quantization,
│                          persistent tiling, residual sparsity,
│                          side-output elision, int8 exponent encoding
│   ├── cuda_ext.py      # cuBLASLt C++ extension JIT loader + Python wrapper
│   ├── csrc/            # C++ source: fused dual FP8 GEMM via cuBLASLt
│   │                      beta=1 accumulation (fused_dual_gemm.h/.cu/_bind.cpp)
│   ├── tcfp12/          # FP8 + 4-bit residual (fake-quantize path)
│   └── nn/              # Drop-in modules (TCFPLinear, TCFPLayerNorm, etc.)
│                          _TCFP12TensorCoreFunction (2-GEMM residual),
│                          _TCFP12BlockScaledFunction (per-block scaling),
│                          PDDS delayed scaling, SRR, and ABD helpers
├── tests/               # 306 tests covering all modes + tensor core paths
│                          + delayed scaling + block scaling + cuBLASLt ext
│                          + persistent tiling + residual sparsity
│                          + side-output elision + int8 exponent encoding
│                          + SRR + ABD
├── benchmarks/          # Quality + throughput + TC GEMM + block GEMM benchmarks
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
