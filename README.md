# TCFP — Tensor Core Floating Point

**FP8 training that matches BF16 quality and runs faster.**

TCFP-12 decomposes each weight matrix into two FP8 components and executes two real
`torch._scaled_mm` GEMMs per layer — giving ~6 effective mantissa bits from FP8
hardware. With Triton fused kernels, a full training step runs **19% faster than BF16**
on an RTX 5070 Ti while converging to identical perplexity.

> Benchmarked on NVIDIA RTX 5070 Ti · PyTorch 2.10 · CUDA 12.8

---

## Motivation

Standard FP8 (E4M3) has only **3 mantissa bits**, which causes catastrophic precision
loss in transformer training — naive FP8 diverges to PPL 55.6 vs BF16's PPL 1.9.
TCFP solves this without falling back to BF16 by exploiting a key insight:

> **The quantization residual is small and well-structured.** A second FP8 GEMM on
> the residual recaptures most of the lost precision — doubling effective mantissa bits
> to ~6 while keeping all compute on FP8 tensor cores.

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

### The Core Idea: Residual Weight Decomposition

Standard FP8 (E4M3) has only **3 mantissa bits**, which is too coarse for stable
transformer training. TCFP-12 doubles the effective precision by decomposing each
weight matrix into a main component and a quantization residual, then running two
separate FP8 GEMMs whose results are summed in FP32:

```
W   = W_hi + W_lo + ε         (decompose into two FP8 E4M3 tensors)
out = X @ W_hi + X @ W_lo     (two tensor-core GEMMs, accumulated in FP32)
```

`W_hi = quant_fp8(W)` captures the bulk of the weight; `W_lo = quant_fp8(W − dequant(W_hi))`
is the quantization residual. Because `|W_lo| ≪ |W|`, it is well-represented in FP8
despite the limited range — yielding **~6 effective mantissa bits** with only FP8 hardware.

---

### Step 1 — FP8 Quantization

Each component is mapped to FP8 E4M3 with a per-tensor (or per-block) scale:

```
scale    = FP8_E4M3_MAX / max|W|     # 448 / amax(W)
W_scaled = W × scale                 # map to [-448, +448]
W_fp8    = round_to_fp8_e4m3(W_scaled)
inv      = 1 / scale                 # stored alongside W_fp8 for dequant
```

In Python (`tcfp/core.py`):

```python
def to_fp8_e4m3(tensor, scale=None):
    if scale is None:
        amax = tensor.abs().max().clamp(min=1e-12)
        scale = (FP8_E4M3_MAX / amax).to(torch.float32)   # FP8_E4M3_MAX = 448.0
    scaled = (tensor.float() * scale).clamp(FP8_E4M3_MIN, FP8_E4M3_MAX)
    fp8    = scaled.to(torch.float8_e4m3fn)
    inv_scale = torch.ones(1, device=tensor.device, dtype=torch.float32) / scale
    return fp8, inv_scale
```

---

### Step 2 — Error Feedback

After quantizing, the total rounding error `ε = W − dequant(W_hi) − dequant(W_lo)` is
accumulated in a per-parameter FP32 buffer. On the next step it is added back before
quantization, making the long-run quantization **unbiased**:

```
# Step t:
w_eff = W + e_{t-1}                                 # add accumulated error
W_hi  = quant_fp8(w_eff)                             # main component
W_lo  = quant_fp8(w_eff − dequant(W_hi))             # residual component
e_t   = W − (dequant(W_hi) + dequant(W_lo))          # new error → FP32 buffer
```

In Python (`tcfp/nn/__init__.py` — `_TCFP12TensorCoreFunction.forward`):

```python
# 1. Add accumulated error from step t-1
w = weight.float()
if error_state is not None:
    w = w + error_state.get_error(param_name, w)      # w_eff = W + e_{t-1}

# 2. Quantize main component
w_hi_fp8, w_hi_inv = to_fp8_e4m3(w)                  # W_hi in FP8 E4M3

# 3. Compute and quantize residual
residual   = w - w_hi_fp8.float() * w_hi_inv          # residual = w_eff − dequant(W_hi)
w_lo_fp8, w_lo_inv = to_fp8_e4m3(residual)            # W_lo in FP8 E4M3

# 4. Store new error for step t+1
if error_state is not None and update_ef:
    w_reconstructed = (w_hi_fp8.float() * w_hi_inv
                       + w_lo_fp8.float() * w_lo_inv)
    error_state.update_error(param_name, weight.float() - w_reconstructed)
```

The EF buffer costs **4 bytes × N_weights** of FP32 state. SRR (see below) eliminates
this buffer entirely.

---

### Step 3 — Forward Pass (2 FP8 GEMMs)

With both weight components ready, the forward pass quantizes the input once and runs
two FP8 GEMMs accumulated in FP32:

```
X_fp8 = quant_fp8(X)
out   = scaled_mm(X_fp8, W_hi, s_x, s_hi)    # GEMM₁ — FP8 tensor cores → FP32
      + scaled_mm(X_fp8, W_lo, s_x, s_lo)    # GEMM₂ — FP8 tensor cores → FP32
```

Data flow:

```
  Input X (BF16)               W (BF16) + EF buffer (FP32)
       │                                  │
       ▼                                  ▼
  [to_fp8_e4m3(X)]     [fused_dual_quantize_fp8]  ← single Triton kernel, one W read
       │                       │               │
  X_fp8, s_x             W_hi_fp8, s_hi    W_lo_fp8, s_lo
       │                                         │
       │             e_t = W − dequant(W_hi) − dequant(W_lo)   (→ EF buffer, FP32)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
  GEMM₁: scaled_mm(X_fp8, W_hi, s_x, s_hi)  GEMM₂: scaled_mm(X_fp8, W_lo, s_x, s_lo)
  (FP8 E4M3 tensor cores → FP32)             (FP8 E4M3 tensor cores → FP32)
       │                                         │
       └──────────────────┬──────────────────────┘
                          ▼
               output = GEMM₁ + GEMM₂   (FP32 → cast to BF16)
```

In Python (`tcfp/nn/__init__.py`):

```python
act_fp8, act_inv = to_fp8_e4m3(input_2d)    # quantize input once for both GEMMs

# 3-tier dispatch: cuBLASLt C++ ext > Triton fused > 2× _scaled_mm
if _HAS_CUDA_EXT and use_cuda_ext:
    # cuBLASLt: beta=1 accumulation, no intermediate FP32 add buffer
    output_2d = cuda_ext_dual_gemm_forward(
        act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv)

elif _HAS_FUSED_KERNELS and use_fused_kernel:
    # Triton: loads activation ONCE, accumulates both GEMMs in registers
    output_2d = fused_dual_gemm_forward(
        act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv)

else:
    # Fallback: two independent _scaled_mm calls + Python add
    out_hi = torch._scaled_mm(act_fp8, w_hi_fp8.t(),
                               scale_a=act_inv, scale_b=w_hi_inv,
                               out_dtype=torch.float32)
    out_lo = torch._scaled_mm(act_fp8, w_lo_fp8.t(),
                               scale_a=act_inv, scale_b=w_lo_inv,
                               out_dtype=torch.float32)
    output_2d = out_hi + out_lo
```

---

### Step 4 — Backward Pass (3 FP8 GEMMs)

Gradients are cast to FP8 E5M2 (wider dynamic range needed for gradient magnitudes)
and always quantized fresh — EMA-delayed scales are not used for gradients because
they change too rapidly during training:

```
grad_fp8 = quant_fp8_e5m2(∂L/∂out)

# Gradient w.r.t. input X (dual-GEMM, or single with ABD)
dX = scaled_mm(grad_fp8, W_hi.T, s_g, s_hi)
   + scaled_mm(grad_fp8, W_lo.T, s_g, s_lo)    # dropped when abd=True

# Gradient w.r.t. weights W (high-precision by default)
dW = (∂L/∂out).T @ X                           # BF16/FP32 outer product
```

**Total FP8 GEMMs per layer per training step:**

| Path | Forward | Backward dX | Backward dW | Total |
|------|---------|-------------|-------------|-------|
| Standard | 2 | 2 | 1 | **5** |
| + ABD (`abd=True`) | 2 | 1 | 1 | **4** |

In Python (`tcfp/nn/__init__.py` — `_TCFP12TensorCoreFunction.backward`):

```python
grad_fp8, grad_inv = to_fp8_e5m2(grad_2d)       # quantize gradient to E5M2

if is_abd:
    # ABD: single GEMM using W_hi only — W_lo contribution is negligible
    grad_input = torch._scaled_mm(
        grad_fp8, ensure_column_major(w_hi_fp8),
        scale_a=grad_inv, scale_b=w_hi_inv,
        out_dtype=torch.float32).reshape(input_shape)
else:
    # Full: fused Triton kernel loads grad once, computes both GEMMs in registers
    grad_input = fused_dual_gemm_backward_dx(
        grad_fp8, w_hi_fp8, w_lo_fp8,
        grad_inv, w_hi_inv, w_lo_inv).reshape(input_shape)

# dW: high-precision (BF16) for numerically stable optimizer updates
if hp_grad_weight:
    grad_weight = grad_2d.t() @ saved_act        # full-precision outer product
```

---

### Effective Precision

| Format | Mantissa bits | MSE on N(0,1) | SNR (dB) |
|--------|--------------|---------------|----------|
| BF16 | 7 | 2.76e-06 | 55.6 |
| **TCFP-12** | **~6** | **2.48e-05** | **46.0** |
| FP8 E4M3 | 3 | 7.02e-04 | 31.5 |

TCFP-12 provides 14.5 dB better SNR than naive FP8 while keeping all compute on FP8
tensor cores.

---

### Optional Enhancements

#### Stochastic Residual Rounding (SRR)

Instead of carrying a FP32 error buffer, SRR replaces the `W_lo` quantization step with
stochastic rounding. Since stochastic rounding is **unbiased per step** — `E[SR(x)] = x`
— no accumulation buffer is needed:

```
p(round_up) = (x − floor_fp8(x)) / ulp(x)     # probability proportional to proximity
```

In Python (`tcfp/core.py`):

```python
def stochastic_round_to_fp8_e4m3(tensor):
    lo   = tensor.to(torch.float8_e4m3fn)       # floor in FP8
    hi   = find_next_fp8_neighbor(lo, tensor)   # ceil in FP8
    gap  = (hi.float() - lo.float()).abs().clamp(min=1e-45)
    frac = ((tensor - lo.float()).abs() / gap).clamp(0.0, 1.0)
    rand = torch.rand_like(frac)
    return torch.where(rand < frac, hi, lo)     # probabilistic rounding
```

This eliminates **4 bytes × N_weights** of FP32 state — **~26 GB at 7B scale** —
with no convergence loss.

#### Predictive Dual-Track Delayed Scaling (PDDS)

Avoids a full `amax` reduction every step by tracking an EMA of the `W_hi` scale and
predicting the `W_lo` scale from the historical `amax_lo / amax_hi` ratio:

```
amax_hi_t = α · amax_hi_{t-1} + (1−α) · amax_hi_current    # α = 0.999
amax_lo_t ≈ amax_hi_t × ratio_ema                           # zero-cost prediction
```

In Python (`tcfp/core.py`):

```python
def get_delayed_amax(self, name, current_amax):
    # EMA: 0.999 * prev_delayed_amax + 0.001 * current  (~693-step half-life)
    self._delayed_amax[name] = (
        self._ema_decay * prev + (1 - self._ema_decay) * current_amax.detach())
    return self._delayed_amax[name]

def predict_residual_amax(self, name, main_amax):
    # Use tracked ratio; fall back to 1/8 (FP8 E4M3: 3 mantissa bits → max rounding error ≈ 2^-3)
    ratio = self._residual_ratio.get(f"{name}.ratio", 1 / 8)
    return (main_amax * ratio).clamp(min=1e-12)
```

#### Per-Block Scaling

Each block of 32/64/128 elements gets its own scale stored as an `int8` exponent
(E8M0 format) — **4× smaller than FP32 scales** — capturing per-region dynamic range:

```
scale_block = 2^ceil(log2(amax_block / FP8_E4M3_MAX))    # power-of-2, stored as int8
```

In Python (`tcfp/core.py`):

```python
def compute_block_scales(tensor, block_size=32, fp8_max=448.0):
    blocked  = tensor.reshape(*batch_dims, num_blocks, block_size)
    amax     = blocked.abs().amax(dim=-1).clamp(min=1e-12)
    raw_exp  = torch.ceil(torch.log2(amax / fp8_max))
    return torch.exp2(raw_exp)           # power-of-2 scale per block
```

---

### Kernel Stack

| Component | Role |
|-----------|------|
| `fused_dual_quantize_fp8` | Single Triton kernel: EF add → quantize `W_hi` → residual → quantize `W_lo`, one GMEM read |
| `fused_dual_gemm_forward` | Triton: load activation once, accumulate both GEMMs in registers (~25% less memory bandwidth vs two `_scaled_mm`) |
| `fused_dual_gemm_backward_dx` | Triton: load gradient once, compute `grad @ (W_hi + W_lo)` with inline scale application |
| cuBLASLt C++ ext | Forward-only: `beta=1` accumulation eliminates intermediate FP32 add buffer |
| Per-block scaling | `int8` E8M0 exponent scales; block sizes 32/64/128; 4× smaller than FP32 |
| Residual sparsity | Near-zero `W_lo` blocks skipped via `int8` sentinel (`−128`) detected in GEMM kernel |
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

### Pre-Training Diagnostic

Check model compatibility before converting:

```python
from tcfp.nn import diagnose

report = diagnose(model, scale_block_size=64)
print(f"{report.tc_eligible_layers}/{report.total_linear_layers} layers TC-eligible")
print(f"VRAM: {report.vram_fp32_bytes/1e9:.1f} GB FP32 → {report.vram_fp8_bytes/1e9:.1f} GB FP8")
print(f"EF overhead: {report.vram_ef_overhead_bytes/1e6:.0f} MB")
print(f"Savings: {report.estimated_savings_pct:.0f}%")

# Per-layer details
for layer in report.layer_details:
    if layer.fallback_reason:
        print(f"  {layer.name}: {layer.fallback_reason}")
```

### Inference Export

Strip TCFP wrappers back to vanilla PyTorch modules for deployment:

```python
from tcfp.nn import export

# After training, convert back to standard nn.Linear / nn.LayerNorm / nn.Embedding
export(model)  # in-place, shares weight tensors (no copy)
torch.save(model.state_dict(), "model.pt")
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

**Runtime methods on `TCFPLinear`:**

| Method | Effect |
|--------|--------|
| `fallback_to_bf16()` | Switch to BF16 pass-through (skip all quantisation) |
| `restore_tcfp()` | Restore TCFP quantisation after a BF16 fallback |

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

## Training Integration

### Gradient Checkpointing

TCFP is compatible with `torch.utils.checkpoint`. The error feedback buffer is
automatically protected from double-update during checkpoint recomputation:

```python
import torch.utils.checkpoint

# Works correctly — EF is updated once, not twice
out = torch.utils.checkpoint.checkpoint(tcfp_layer, x, use_reentrant=False)
```

### Gradient Accumulation

During multi-micro-batch accumulation, EF is updated only once per optimizer step.
TCFP tracks `weight._version` internally — subsequent forwards with unchanged weights
skip redundant quantisation and EF writes automatically:

```python
for micro_batch in micro_batches:
    loss = model(micro_batch).sum()
    loss.backward()                  # EF updated on first forward only
optimizer.step()                     # weight changes → next forward updates EF
optimizer.zero_grad()
```

### torch.compile

`TCFPLinear.forward` is decorated with `@torch.compiler.disable` to prevent
miscompilation from dict mutation, conditional dispatch, and step counters. The
surrounding model can still be compiled — only the TCFP forward is excluded from
graph tracing:

```python
model = torch.compile(model, backend="inductor")  # safe — TCFP layers opt out
```

### Runtime BF16 Fallback

Individual layers can be switched to BF16 pass-through at any point during training,
useful for recovering divergent layers:

```python
layer.fallback_to_bf16()   # skip all quantisation, use F.linear directly
layer.restore_tcfp()       # resume TCFP quantisation
```

The EFMA policy (`MomentumAlignmentTracker`) can escalate to BF16 fallback
automatically as Level 4 — the terminal intervention when a layer shows persistent
quantisation drag even after highway promotion.

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
│                    convert_to_tcfp, diagnose, export; autograd functions
│                    for all GEMM paths
└── training/        Training utilities: checkpointing, monitoring, phase
                     detection, progressive quantization, layer policies

tests/               436 tests (420 passed, 16 skipped on CPU-only)
benchmarks/          Throughput, VRAM, and convergence benchmarks
examples/            TinyGPT and MNIST convergence validation scripts
```

---

## Running Tests

```bash
py -m pytest tests/ -v
```

Expected output: **420 passed, 16 skipped** (CUDA-only tests skipped on CPU-only machines).

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
- Matches BF16 convergence on TinyGPT (PPL ratio 1.002-1.003x)
- Faster than BF16 end-to-end: TCFP-12 TC runs at 1.19x BF16 speed (13.28ms vs 15.82ms)
- 22% less weight VRAM than BF16 at any model size
- SRR eliminates the error feedback buffer entirely (saves 26 GB at 7B scale)
- ABD saves ~20% of backward GEMMs with no quality loss
- Compatible with `torch.utils.checkpoint`, gradient accumulation, and `torch.compile`
- Pre-training diagnostic (`diagnose()`) and inference export (`export()`)
- Runtime BF16 fallback for individual layers, with EFMA-driven auto-escalation

**Current limitations:**
- **Single-GPU only** — not validated with FSDP, tensor parallelism, or pipeline parallelism
- **No framework integration** — requires manual `convert_to_tcfp()` call; no HuggingFace or DeepSpeed plug-in
- **Research scale only** — validated at TinyGPT (4-layer, d=256); not yet tested at 7B+
- **Per-block path still slower** — block-scaled kernels are 7-14% slower than BF16 (Block-64: 17.01ms, Block-32: 18.11ms vs BF16 15.82ms)

---

## License

Apache 2.0
