"""
TCFP Fused Triton Kernels
==========================

Custom Triton kernels for TCFP FP8 training:

1. **Fused 2-GEMM** — reads shared activation once for both dot products,
   saves ~25% memory bandwidth vs two ``torch._scaled_mm`` calls.

2. **Block quantization** — fused per-block FP8 quantization (compute
   block amax, power-of-2 scale, scale, cast) in a single kernel pass.

3. **Block-scaled GEMM** — FP8 matmul with per-K-block scale correction,
   enabling finer-grained quantization than ``torch._scaled_mm``'s
   scalar-only scales.

4. **Fused block-scaled dual GEMM** — combines (1) and (3) for TCFP-12
   with per-block scaling.

Requires ``triton >= 3.0.0`` and SM89+ GPU (Ada Lovelace / Hopper /
Blackwell).
"""
from __future__ import annotations

import torch

_TRITON_AVAILABLE = False

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True

    # ── Autotune configs ──────────────────────────────────────────────
    # Smaller M/N tiles than standard matmul because we maintain two
    # FP32 accumulators (2x register pressure).  BLOCK_K can stay large
    # since FP8 data is half the size of FP16.
    _DUAL_GEMM_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
    ]

    # ── Kernel definition ─────────────────────────────────────────────

    @triton.autotune(configs=_DUAL_GEMM_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _fused_dual_gemm_kernel(
        # Pointers
        a_ptr,
        b_hi_ptr,
        b_lo_ptr,
        out_ptr,
        # Scalar inverse-scale pointers
        scale_a_ptr,
        scale_hi_ptr,
        scale_lo_ptr,
        # Dimensions
        M,
        N,
        K,
        # A strides
        stride_am,
        stride_ak,
        # B_hi strides (K, N) layout via strides
        stride_bk_hi,
        stride_bn_hi,
        # B_lo strides
        stride_bk_lo,
        stride_bn_lo,
        # Output strides
        stride_om,
        stride_on,
        # Meta-parameters
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Fused dual-GEMM with shared input and FP32 accumulation.

        Computes::

            out = (A @ B_hi) * (s_a * s_hi) + (A @ B_lo) * (s_a * s_lo)

        where A is (M, K), B_hi/B_lo are (K, N) via strides, and scales
        are scalar FP32 inverse scales.

        The key optimisation: A tiles are loaded from global memory
        **once** per K-iteration and reused for both dot products.
        """
        # ── Program-ID mapping with grouping for L2 reuse ──
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BLOCK_M)
        num_n = tl.cdiv(N, BLOCK_N)
        num_in_group = GROUP_M * num_n
        group_id = pid // num_in_group
        first_m = group_id * GROUP_M
        group_m = min(num_m - first_m, GROUP_M)
        pid_m = first_m + ((pid % num_in_group) % group_m)
        pid_n = (pid % num_in_group) // group_m

        # ── Load scalar inverse scales (once per CTA) ──
        s_a = tl.load(scale_a_ptr)
        s_hi = tl.load(scale_hi_ptr)
        s_lo = tl.load(scale_lo_ptr)
        combined_hi = s_a * s_hi  # pre-combine shared act scale
        combined_lo = s_a * s_lo

        # ── Pointer setup (% M / % N for safe wrap-around) ──
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        bh_ptrs = (
            b_hi_ptr
            + offs_k[:, None] * stride_bk_hi
            + offs_n[None, :] * stride_bn_hi
        )
        bl_ptrs = (
            b_lo_ptr
            + offs_k[:, None] * stride_bk_lo
            + offs_n[None, :] * stride_bn_lo
        )

        # ── Two FP32 accumulators ──
        acc_hi = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_lo = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # ── K-dimension loop ──
        for k_start in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k < (K - k_start * BLOCK_K)

            # Load A tile ONCE (the key bandwidth saving)
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            # Load both B tiles
            bh = tl.load(bh_ptrs, mask=k_mask[:, None], other=0.0)
            bl = tl.load(bl_ptrs, mask=k_mask[:, None], other=0.0)

            # FP8 dot products -> FP32 accumulation
            # max_num_imprecise_acc=0 matches use_fast_accum=False
            acc_hi = tl.dot(a, bh, acc_hi, max_num_imprecise_acc=0)
            acc_lo = tl.dot(a, bl, acc_lo, max_num_imprecise_acc=0)

            # Advance K pointers
            a_ptrs += BLOCK_K * stride_ak
            bh_ptrs += BLOCK_K * stride_bk_hi
            bl_ptrs += BLOCK_K * stride_bk_lo

        # ── Epilogue: scale and sum ──
        output = acc_hi * combined_hi + acc_lo * combined_lo

        # ── Store output (masked for partial tiles) ──
        offs_om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        out_ptrs = (
            out_ptr
            + offs_om[:, None] * stride_om
            + offs_on[None, :] * stride_on
        )
        out_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
        tl.store(out_ptrs, output, mask=out_mask)

    # ── Block quantization configs ──────────────────────────────────────
    # Memory-bound kernel: autotune BLOCK_M only (BLOCK_SIZE is user-set).
    _BLOCK_QUANT_CONFIGS = [
        triton.Config({"BLOCK_M": 32}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 64}, num_stages=4, num_warps=4),
        triton.Config({"BLOCK_M": 128}, num_stages=3, num_warps=8),
    ]

    @triton.autotune(configs=_BLOCK_QUANT_CONFIGS, key=["M", "K", "BLOCK_SIZE"])
    @triton.jit
    def _block_quantize_kernel(
        # Pointers
        input_ptr,
        output_ptr,
        scales_ptr,
        # Dimensions
        M,
        K,
        # Input strides
        stride_im,
        stride_ik,
        # Output strides
        stride_om,
        stride_ok,
        # Scale strides
        stride_sm,
        stride_sk,
        # Constants
        FP8_MAX: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        """Fused per-block FP8 quantization.

        For each (BLOCK_M, BLOCK_SIZE) tile: compute per-row block amax,
        power-of-2 inverse scale, scale input, clamp, and cast to FP8.
        """
        pid_m = tl.program_id(0)
        pid_kb = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_kb * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        # Load input tile
        mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        in_ptrs = (
            input_ptr
            + offs_m[:, None] * stride_im
            + offs_k[None, :] * stride_ik
        )
        x = tl.load(in_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Per-row amax within this K-block
        amax = tl.max(tl.abs(x), axis=1)  # (BLOCK_M,)
        amax = tl.maximum(amax, 1e-12)

        # Power-of-2 inverse scale: inv_scale = 2^ceil(log2(amax / FP8_MAX))
        raw_exp = tl.ceil(tl.log2(amax / FP8_MAX))
        # Clamp exponent to avoid denormals (min exp = -126)
        raw_exp = tl.maximum(raw_exp, -126.0)
        inv_scale = tl.exp2(raw_exp)  # (BLOCK_M,)

        # Scale input: x_scaled = x / inv_scale (= x * forward_scale)
        x_scaled = x / inv_scale[:, None]
        x_clamped = tl.maximum(tl.minimum(x_scaled, FP8_MAX), -FP8_MAX)

        # Cast to FP8 and store
        x_fp8 = x_clamped.to(output_ptr.dtype.element_ty)
        out_ptrs = (
            output_ptr
            + offs_m[:, None] * stride_om
            + offs_k[None, :] * stride_ok
        )
        tl.store(out_ptrs, x_fp8, mask=mask)

        # Store inverse scales
        scale_ptrs = scales_ptr + offs_m * stride_sm + pid_kb * stride_sk
        scale_mask = offs_m < M
        tl.store(scale_ptrs, inv_scale, mask=scale_mask)

    # ── Fused dual quantization configs ───────────────────────────────────
    # Memory-bound quantization kernel: BLOCK_N autotuned, BLOCK_K fixed
    # (= block_size, constexpr).  Grid is 2D (num_n_blocks, num_k_blocks).
    _FUSED_DUAL_QUANT_CONFIGS = [
        triton.Config({"BLOCK_N": 32}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_N": 64}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_N": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_N": 128}, num_stages=3, num_warps=8),
    ]

    @triton.autotune(configs=_FUSED_DUAL_QUANT_CONFIGS, key=["N", "K"])
    @triton.jit
    def _fused_dual_quantize_kernel(
        # Pointers
        w_ptr,
        error_old_ptr,
        w_hi_ptr,
        w_lo_ptr,
        scales_hi_ptr,
        scales_lo_ptr,
        # Dimensions
        N,
        K,
        # Weight strides
        stride_wn,
        stride_wk,
        # Error-old strides
        stride_eon,
        stride_eok,
        # W_hi strides
        stride_hn,
        stride_hk,
        # W_lo strides
        stride_ln,
        stride_lk,
        # Scales_hi strides
        stride_sh_n,
        stride_sh_kb,
        # Scales_lo strides
        stride_sl_n,
        stride_sl_kb,
        # Compile-time constants
        HAS_ERROR: tl.constexpr,
        FP8_MAX: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Fused dual FP8 quantisation: W → (W_hi, W_lo) in one pass.

        Each CTA handles one (BLOCK_N, BLOCK_K) tile.  Grid is 2D
        ``(cdiv(N, BLOCK_N), K // BLOCK_K)`` — no pid_m, no write-back race.

        When ``HAS_ERROR`` is True, the kernel reads old quantisation error
        from ``error_old_ptr`` and adds it to the weight before quantisation.
        The new error is computed outside the kernel using stored FP8 values.

        Uses power-of-2 scaling: ``inv_scale = exp2(ceil(log2(amax / FP8_MAX)))``
        matching :func:`_block_quantize_kernel` exactly.
        """
        pid_n = tl.program_id(0)
        pid_kb = tl.program_id(1)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
        mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        # ── 1. Load W tile (coalesced: stride-1 along K) → FP32 ──
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)

        # ── 2. Optional error feedback (compile-time branch) ──
        if HAS_ERROR:
            err_old_ptrs = (
                error_old_ptr
                + offs_n[:, None] * stride_eon
                + offs_k[None, :] * stride_eok
            )
            old_error = tl.load(err_old_ptrs, mask=mask, other=0.0)
            w_corrected = w_tile + old_error
        else:
            w_corrected = w_tile

        # ── 3. Quantize W_hi — power-of-2 scaling ──
        amax_hi = tl.max(tl.abs(w_corrected), axis=1)  # (BLOCK_N,)
        amax_hi = tl.maximum(amax_hi, 1e-12)
        raw_exp_hi = tl.ceil(tl.log2(amax_hi / FP8_MAX))
        raw_exp_hi = tl.maximum(raw_exp_hi, -126.0)
        inv_scale_hi = tl.exp2(raw_exp_hi)  # (BLOCK_N,)

        w_hi_scaled = w_corrected / inv_scale_hi[:, None]
        w_hi_clamped = tl.maximum(tl.minimum(w_hi_scaled, FP8_MAX), -FP8_MAX)
        w_hi_fp8 = w_hi_clamped.to(w_hi_ptr.dtype.element_ty)

        # ── 4. Compute residual in registers (no GMEM round-trip) ──
        residual = w_corrected - w_hi_fp8.to(tl.float32) * inv_scale_hi[:, None]

        # ── 5. Quantize W_lo — same power-of-2 scaling ──
        amax_lo = tl.max(tl.abs(residual), axis=1)  # (BLOCK_N,)

        # Zero-residual fast path: set inv_scale=1.0 BEFORE quantization
        # so FP8 values and stored scales are consistent (0/1=0 → 0*1=0).
        zero_residual = amax_lo < 1e-12
        amax_lo_safe = tl.maximum(amax_lo, 1e-12)  # safe for log2

        raw_exp_lo = tl.ceil(tl.log2(amax_lo_safe / FP8_MAX))
        raw_exp_lo = tl.maximum(raw_exp_lo, -126.0)
        inv_scale_lo = tl.exp2(raw_exp_lo)
        inv_scale_lo = tl.where(zero_residual, 1.0, inv_scale_lo)

        w_lo_scaled = residual / inv_scale_lo[:, None]
        w_lo_clamped = tl.maximum(tl.minimum(w_lo_scaled, FP8_MAX), -FP8_MAX)
        w_lo_fp8 = w_lo_clamped.to(w_lo_ptr.dtype.element_ty)

        # NOTE: Error update is computed outside the kernel using the stored
        # FP8 values.  In-kernel dequant can differ from stored-value dequant
        # due to Triton compiler optimisations folding FP8 casts.

        # ── 6. Store FP8 outputs (coalesced: stride-1 along K) ──
        hi_ptrs = (
            w_hi_ptr
            + offs_n[:, None] * stride_hn
            + offs_k[None, :] * stride_hk
        )
        tl.store(hi_ptrs, w_hi_fp8, mask=mask)

        lo_ptrs = (
            w_lo_ptr
            + offs_n[:, None] * stride_ln
            + offs_k[None, :] * stride_lk
        )
        tl.store(lo_ptrs, w_lo_fp8, mask=mask)

        # ── 7. Store per-row inverse scales ──
        n_mask = offs_n < N
        tl.store(
            scales_hi_ptr + offs_n * stride_sh_n + pid_kb * stride_sh_kb,
            inv_scale_hi,
            mask=n_mask,
        )
        tl.store(
            scales_lo_ptr + offs_n * stride_sl_n + pid_kb * stride_sl_kb,
            inv_scale_lo,
            mask=n_mask,
        )

    # ── Block-scaled GEMM configs ────────────────────────────────────────
    # BLOCK_K is passed as constexpr (= block_size), not autotuned.
    _BLOCK_GEMM_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
    ]

    @triton.autotune(configs=_BLOCK_GEMM_CONFIGS, key=["M", "N", "K", "BLOCK_K"])
    @triton.jit
    def _block_scaled_gemm_kernel(
        # Pointers
        a_ptr,
        b_ptr,
        out_ptr,
        scales_a_ptr,
        scales_b_ptr,
        # Dimensions
        M,
        N,
        K,
        num_k_blocks,
        # A strides
        stride_am,
        stride_ak,
        # B strides (K, N) via strides
        stride_bk,
        stride_bn,
        # Output strides
        stride_om,
        stride_on,
        # Scale strides: scale_a (M, num_kb), scale_b (N, num_kb)
        stride_sa_m,
        stride_sa_kb,
        stride_sb_n,
        stride_sb_kb,
        # Meta-parameters
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """FP8 GEMM with per-K-block scale correction.

        Computes::

            C[m,n] = sum_{kb} sa[m,kb] * sb[n,kb] *
                     sum_{k in block} A_q[m,k] * B_q[k,n]

        BLOCK_K must equal the scaling block_size so each K-iteration
        processes exactly one scale block.
        """
        # ── Program-ID mapping with grouping for L2 reuse ──
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BLOCK_M)
        num_n = tl.cdiv(N, BLOCK_N)
        num_in_group = GROUP_M * num_n
        group_id = pid // num_in_group
        first_m = group_id * GROUP_M
        group_m = min(num_m - first_m, GROUP_M)
        pid_m = first_m + ((pid % num_in_group) % group_m)
        pid_n = (pid % num_in_group) // group_m

        # ── Offset setup ──
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            b_ptr
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )

        # ── FP32 accumulator ──
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # ── K-dimension loop (one scale block per iteration) ──
        for kb in range(0, num_k_blocks):
            k_mask = offs_k < (K - kb * BLOCK_K)

            # Load FP8 tiles
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)

            # FP8 dot product -> FP32 partial result
            partial = tl.dot(a, b, max_num_imprecise_acc=0)

            # Load per-block scales for this K-block
            sa = tl.load(
                scales_a_ptr + offs_m * stride_sa_m + kb * stride_sa_kb
            )  # (BLOCK_M,)
            sb = tl.load(
                scales_b_ptr + offs_n * stride_sb_n + kb * stride_sb_kb
            )  # (BLOCK_N,)

            # Apply scale correction and accumulate
            acc += partial * (sa[:, None] * sb[None, :])

            # Advance K pointers
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # ── Store output (masked for partial tiles) ──
        offs_om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        out_ptrs = (
            out_ptr
            + offs_om[:, None] * stride_om
            + offs_on[None, :] * stride_on
        )
        out_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
        tl.store(out_ptrs, acc, mask=out_mask)

    # ── Fused block-scaled dual GEMM configs ─────────────────────────────
    # Smaller tiles: two accumulators + three scale arrays = high register
    # pressure.
    _BLOCK_DUAL_GEMM_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
    ]

    @triton.autotune(
        configs=_BLOCK_DUAL_GEMM_CONFIGS, key=["M", "N", "K", "BLOCK_K"]
    )
    @triton.jit
    def _fused_block_scaled_dual_gemm_kernel(
        # Pointers
        a_ptr,
        b_hi_ptr,
        b_lo_ptr,
        out_ptr,
        # Block scale pointers
        scales_a_ptr,
        scales_bhi_ptr,
        scales_blo_ptr,
        # Dimensions
        M,
        N,
        K,
        num_k_blocks,
        # A strides
        stride_am,
        stride_ak,
        # B_hi strides
        stride_bk_hi,
        stride_bn_hi,
        # B_lo strides
        stride_bk_lo,
        stride_bn_lo,
        # Output strides
        stride_om,
        stride_on,
        # Scale_a strides
        stride_sa_m,
        stride_sa_kb,
        # Scale_bhi strides
        stride_sbhi_n,
        stride_sbhi_kb,
        # Scale_blo strides
        stride_sblo_n,
        stride_sblo_kb,
        # Meta-parameters
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Fused dual-GEMM with per-block scales and shared input.

        Computes::

            out[m,n] = sum_{kb} sa[m,kb] * (
                sbhi[n,kb] * dot(A_kb, Bhi_kb)
              + sblo[n,kb] * dot(A_kb, Blo_kb)
            )

        A tiles are loaded **once** per K-iteration (bandwidth saving).
        ``sa`` is factored out of both terms (saves one multiply).
        """
        # ── Program-ID mapping with grouping for L2 reuse ──
        pid = tl.program_id(0)
        num_m = tl.cdiv(M, BLOCK_M)
        num_n = tl.cdiv(N, BLOCK_N)
        num_in_group = GROUP_M * num_n
        group_id = pid // num_in_group
        first_m = group_id * GROUP_M
        group_m = min(num_m - first_m, GROUP_M)
        pid_m = first_m + ((pid % num_in_group) % group_m)
        pid_n = (pid % num_in_group) // group_m

        # ── Offset setup ──
        offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
        offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        bh_ptrs = (
            b_hi_ptr
            + offs_k[:, None] * stride_bk_hi
            + offs_n[None, :] * stride_bn_hi
        )
        bl_ptrs = (
            b_lo_ptr
            + offs_k[:, None] * stride_bk_lo
            + offs_n[None, :] * stride_bn_lo
        )

        # ── FP32 accumulator ──
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # ── K-dimension loop (one scale block per iteration) ──
        for kb in range(0, num_k_blocks):
            k_mask = offs_k < (K - kb * BLOCK_K)

            # Load A tile ONCE (bandwidth saving)
            a = tl.load(a_ptrs, mask=k_mask[None, :], other=0.0)
            # Load both B tiles
            bh = tl.load(bh_ptrs, mask=k_mask[:, None], other=0.0)
            bl = tl.load(bl_ptrs, mask=k_mask[:, None], other=0.0)

            # Two FP8 dot products -> FP32
            partial_hi = tl.dot(a, bh, max_num_imprecise_acc=0)
            partial_lo = tl.dot(a, bl, max_num_imprecise_acc=0)

            # Load per-block scales
            sa = tl.load(
                scales_a_ptr + offs_m * stride_sa_m + kb * stride_sa_kb
            )  # (BLOCK_M,)
            sbhi = tl.load(
                scales_bhi_ptr + offs_n * stride_sbhi_n + kb * stride_sbhi_kb
            )  # (BLOCK_N,)
            sblo = tl.load(
                scales_blo_ptr + offs_n * stride_sblo_n + kb * stride_sblo_kb
            )  # (BLOCK_N,)

            # Factor sa out: acc += (hi*sbhi + lo*sblo) * sa
            combined = partial_hi * sbhi[None, :] + partial_lo * sblo[None, :]
            acc += combined * sa[:, None]

            # Advance K pointers
            a_ptrs += BLOCK_K * stride_ak
            bh_ptrs += BLOCK_K * stride_bk_hi
            bl_ptrs += BLOCK_K * stride_bk_lo

        # ── Store output (masked for partial tiles) ──
        offs_om = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_on = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        out_ptrs = (
            out_ptr
            + offs_om[:, None] * stride_om
            + offs_on[None, :] * stride_on
        )
        out_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
        tl.store(out_ptrs, acc, mask=out_mask)

except ImportError:
    pass


def is_triton_available() -> bool:
    """Return True if Triton is installed and importable."""
    return _TRITON_AVAILABLE


# ── Python wrappers: scalar-scale fused dual GEMM ────────────────────


def fused_dual_gemm_forward(
    act_fp8: torch.Tensor,
    w_hi_fp8: torch.Tensor,
    w_lo_fp8: torch.Tensor,
    act_inv: torch.Tensor,
    w_hi_inv: torch.Tensor,
    w_lo_inv: torch.Tensor,
) -> torch.Tensor:
    """Fused 2-GEMM forward: ``(act @ W_hi.T) * s + (act @ W_lo.T) * s``.

    Args:
        act_fp8:  (M, K) FP8 E4M3 activations, row-major.
        w_hi_fp8: (N, K) FP8 E4M3 main weight, row-major (nn.Linear).
        w_lo_fp8: (N, K) FP8 E4M3 residual weight, row-major.
        act_inv:  Scalar FP32 inverse scale for activations.
        w_hi_inv: Scalar FP32 inverse scale for W_hi.
        w_lo_inv: Scalar FP32 inverse scale for W_lo.

    Returns:
        (M, N) FP32 output tensor.
    """
    M, K = act_fp8.shape
    N, K2 = w_hi_fp8.shape
    assert K == K2 and w_lo_fp8.shape == (N, K), (
        f"Shape mismatch: act({M},{K}), w_hi({N},{K2}), w_lo{w_lo_fp8.shape}"
    )

    output = torch.empty((M, N), device=act_fp8.device, dtype=torch.float32)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"])  # pyright: ignore[reportPossiblyUnboundVariable]
        * triton.cdiv(N, META["BLOCK_N"]),  # pyright: ignore[reportPossiblyUnboundVariable]
    )

    # W is (N, K) row-major.  We want B = W.T = (K, N).
    # B[k, n] = W[n, k] -> stride_bk = W.stride(1), stride_bn = W.stride(0)
    _fused_dual_gemm_kernel[grid](  # pyright: ignore[reportPossiblyUnboundVariable]
        act_fp8,
        w_hi_fp8,
        w_lo_fp8,
        output,
        act_inv,
        w_hi_inv,
        w_lo_inv,
        M,
        N,
        K,
        act_fp8.stride(0),
        act_fp8.stride(1),
        w_hi_fp8.stride(1),
        w_hi_fp8.stride(0),  # transposed
        w_lo_fp8.stride(1),
        w_lo_fp8.stride(0),  # transposed
        output.stride(0),
        output.stride(1),
    )
    return output


def fused_dual_gemm_backward_dx(
    grad_fp8: torch.Tensor,
    w_hi_fp8: torch.Tensor,
    w_lo_fp8: torch.Tensor,
    grad_inv: torch.Tensor,
    w_hi_inv: torch.Tensor,
    w_lo_inv: torch.Tensor,
) -> torch.Tensor:
    """Fused 2-GEMM backward dX: ``(grad @ W_hi) * s + (grad @ W_lo) * s``.

    Reuses the same kernel as the forward -- only the strides differ.

    Args:
        grad_fp8: (M, D_out) FP8 E5M2 gradient, row-major.
        w_hi_fp8: (D_out, D_in) FP8 E4M3 main weight, row-major.
        w_lo_fp8: (D_out, D_in) FP8 E4M3 residual weight, row-major.
        grad_inv: Scalar FP32 inverse scale for gradient.
        w_hi_inv: Scalar FP32 inverse scale for W_hi.
        w_lo_inv: Scalar FP32 inverse scale for W_lo.

    Returns:
        (M, D_in) FP32 grad_input tensor.
    """
    M, D_out = grad_fp8.shape
    D_out2, D_in = w_hi_fp8.shape
    assert D_out == D_out2 and w_lo_fp8.shape == (D_out, D_in), (
        f"Shape mismatch: grad({M},{D_out}), "
        f"w_hi({D_out2},{D_in}), w_lo{w_lo_fp8.shape}"
    )

    grad_input = torch.empty(
        (M, D_in), device=grad_fp8.device, dtype=torch.float32
    )

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"])  # pyright: ignore[reportPossiblyUnboundVariable]
        * triton.cdiv(D_in, META["BLOCK_N"]),  # pyright: ignore[reportPossiblyUnboundVariable]
    )

    # grad(M, D_out) @ W(D_out, D_in) -> kernel's A(M, K) @ B(K, N)
    # where K=D_out, N=D_in.  B = W in native (D_out, D_in) layout.
    # stride_bk = W.stride(0), stride_bn = W.stride(1)  (NOT transposed)
    _fused_dual_gemm_kernel[grid](  # pyright: ignore[reportPossiblyUnboundVariable]
        grad_fp8,
        w_hi_fp8,
        w_lo_fp8,
        grad_input,
        grad_inv,
        w_hi_inv,
        w_lo_inv,
        M,
        D_in,
        D_out,  # kernel sees: M=M, N=D_in, K=D_out
        grad_fp8.stride(0),
        grad_fp8.stride(1),
        w_hi_fp8.stride(0),
        w_hi_fp8.stride(1),  # native strides
        w_lo_fp8.stride(0),
        w_lo_fp8.stride(1),  # native strides
        grad_input.stride(0),
        grad_input.stride(1),
    )
    return grad_input


# ── Python wrappers: block-scaled kernels ────────────────────────────

_FP8_E4M3_MAX = 448.0
_FP8_E5M2_MAX = 57344.0


def block_quantize_fp8(
    tensor: torch.Tensor,
    block_size: int = 128,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused per-block FP8 quantization via Triton kernel.

    Args:
        tensor:  (M, K) input tensor (float32 or bfloat16).
        block_size: Scaling block size along K (32, 64, or 128).
        fp8_dtype: Target FP8 type (``float8_e4m3fn`` or ``float8_e5m2``).

    Returns:
        ``(fp8_tensor, block_inv_scales)`` where fp8_tensor is ``(M, K)``
        and block_inv_scales is ``(M, K // block_size)`` FP32 dequant
        factors.
    """
    M, K = tensor.shape
    assert K % block_size == 0, (
        f"K ({K}) must be divisible by block_size ({block_size})"
    )
    num_kb = K // block_size
    fp8_max = _FP8_E5M2_MAX if fp8_dtype == torch.float8_e5m2 else _FP8_E4M3_MAX

    output = torch.empty((M, K), device=tensor.device, dtype=fp8_dtype)
    scales = torch.empty((M, num_kb), device=tensor.device, dtype=torch.float32)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"]),  # pyright: ignore[reportPossiblyUnboundVariable]
        num_kb,
    )

    _block_quantize_kernel[grid](  # pyright: ignore[reportPossiblyUnboundVariable]
        tensor,
        output,
        scales,
        M,
        K,
        tensor.stride(0),
        tensor.stride(1),
        output.stride(0),
        output.stride(1),
        scales.stride(0),
        scales.stride(1),
        FP8_MAX=fp8_max,
        BLOCK_SIZE=block_size,
    )
    return output, scales


def block_dequantize_fp8(
    fp8_tensor: torch.Tensor,
    block_scales: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Dequantize a block-scaled FP8 tensor to FP32.

    Args:
        fp8_tensor:   (M, K) FP8 tensor.
        block_scales: (M, K // block_size) FP32 inverse scales.
        block_size:   Scaling block size used during quantization.

    Returns:
        (M, K) FP32 dequantized tensor.
    """
    M, K = fp8_tensor.shape
    num_blocks = K // block_size
    blocked = fp8_tensor.float().reshape(M, num_blocks, block_size)
    return (blocked * block_scales.unsqueeze(-1)).reshape(M, K)


def fused_dual_quantize_fp8(
    weight: torch.Tensor,
    block_size: int = 128,
    error_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused W → W_hi + W_lo dual FP8 quantization in a single kernel.

    Reads the weight matrix once, quantizes both the main component (W_hi)
    and residual (W_lo) in-register, and writes back both FP8 tensors plus
    per-block inverse scales.  This replaces 3 separate kernel calls
    (``block_quantize_fp8`` for W_hi, dequant + subtract for residual,
    ``block_quantize_fp8`` for W_lo) with a single fused kernel.

    Weight can be BF16 or FP32 — converted to FP32 inside the Triton kernel
    after the coalesced load (avoids extra Python-side ``.float()`` cast).

    If ``error_buf`` is provided it is **modified in-place**: the kernel reads
    the old error, applies it (``w_corrected = weight + old_error``), and
    overwrites the buffer with the new quantisation error
    (``new_error = weight - dequant(W_hi) - dequant(W_lo)``).

    Args:
        weight:    (N, K) contiguous tensor in BF16 or FP32.
        block_size: Scaling block size along K (must divide K).
        error_buf: Optional (N, K) FP32 contiguous tensor for in-place
                   error-feedback I/O.

    Returns:
        ``(w_hi_fp8, w_lo_fp8, scales_hi, scales_lo)`` where the FP8
        tensors are ``(N, K)`` and the scales are ``(N, K // block_size)``
        FP32 inverse-scales compatible with :func:`block_dequantize_fp8`.
    """
    assert weight.is_contiguous(), "Weight must be contiguous (N, K) row-major"
    N, K = weight.shape
    assert K % block_size == 0, (
        f"K ({K}) must be divisible by block_size ({block_size})"
    )
    if error_buf is not None:
        assert error_buf.dtype == torch.float32, "error_buf must be float32"
        assert error_buf.shape == (N, K), (
            f"error_buf shape {error_buf.shape} != weight shape {(N, K)}"
        )
        assert error_buf.is_contiguous(), "error_buf must be contiguous"

    num_kb = K // block_size

    w_hi = torch.empty((N, K), device=weight.device, dtype=torch.float8_e4m3fn)
    w_lo = torch.empty((N, K), device=weight.device, dtype=torch.float8_e4m3fn)
    scales_hi = torch.empty(
        (N, num_kb), device=weight.device, dtype=torch.float32,
    )
    scales_lo = torch.empty(
        (N, num_kb), device=weight.device, dtype=torch.float32,
    )

    # Clone error buffer for reading — autotune reruns the kernel multiple
    # times so the read source must be stable.  The kernel only READS old
    # error; new error is computed outside the kernel using stored FP8 values
    # to avoid Triton compiler cast-folding issues.
    has_error = error_buf is not None
    if has_error:
        error_old = error_buf.clone()
    else:
        error_old = weight  # dummy, never accessed

    grid = lambda META: (  # noqa: E731
        triton.cdiv(N, META["BLOCK_N"]),  # pyright: ignore[reportPossiblyUnboundVariable]
        num_kb,
    )

    _fused_dual_quantize_kernel[grid](  # pyright: ignore[reportPossiblyUnboundVariable]
        weight,
        error_old,
        w_hi,
        w_lo,
        scales_hi,
        scales_lo,
        N,
        K,
        # Weight strides
        weight.stride(0),
        weight.stride(1),
        # Error-old strides
        error_old.stride(0) if has_error else 0,
        error_old.stride(1) if has_error else 0,
        # W_hi strides
        w_hi.stride(0),
        w_hi.stride(1),
        # W_lo strides
        w_lo.stride(0),
        w_lo.stride(1),
        # Scales_hi strides
        scales_hi.stride(0),
        scales_hi.stride(1),
        # Scales_lo strides
        scales_lo.stride(0),
        scales_lo.stride(1),
        HAS_ERROR=has_error,
        FP8_MAX=_FP8_E4M3_MAX,
        BLOCK_K=block_size,
    )

    # Compute new error outside the kernel using stored FP8 values.
    # In-kernel dequant can differ from stored-value dequant due to
    # Triton compiler optimisations that fold FP8 round-trip casts.
    if has_error:
        w_hi_deq = block_dequantize_fp8(w_hi, scales_hi, block_size)
        w_lo_deq = block_dequantize_fp8(w_lo, scales_lo, block_size)
        error_buf.copy_(weight.float() - w_hi_deq - w_lo_deq)  # type: ignore[union-attr]

    return w_hi, w_lo, scales_hi, scales_lo


def block_scaled_gemm(
    a_fp8: torch.Tensor,
    b_fp8: torch.Tensor,
    scales_a: torch.Tensor,
    scales_b: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Block-scaled FP8 GEMM with per-K-block scale correction.

    Computes ``C = sum_kb sa[m,kb] * sb[n,kb] * dot(A_kb, B_kb^T)``.

    Args:
        a_fp8:    (M, K) FP8 input, row-major.
        b_fp8:    (N, K) FP8 weight, row-major (nn.Linear layout).
        scales_a: (M, K // block_size) FP32 block scales for A.
        scales_b: (N, K // block_size) FP32 block scales for B.
        block_size: Scaling block size (must match quantization).

    Returns:
        (M, N) FP32 output tensor.
    """
    M, K = a_fp8.shape
    N, K2 = b_fp8.shape
    assert K == K2, f"K mismatch: A ({K}) vs B ({K2})"
    assert K % block_size == 0, f"K ({K}) not divisible by block_size ({block_size})"
    num_kb = K // block_size

    output = torch.empty((M, N), device=a_fp8.device, dtype=torch.float32)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"])  # pyright: ignore[reportPossiblyUnboundVariable]
        * triton.cdiv(N, META["BLOCK_N"]),  # pyright: ignore[reportPossiblyUnboundVariable]
    )

    # B is (N, K) row-major. Kernel needs B.T = (K, N).
    # stride_bk = B.stride(1), stride_bn = B.stride(0)
    _block_scaled_gemm_kernel[grid](  # pyright: ignore[reportPossiblyUnboundVariable]
        a_fp8,
        b_fp8,
        output,
        scales_a,
        scales_b,
        M,
        N,
        K,
        num_kb,
        a_fp8.stride(0),
        a_fp8.stride(1),
        b_fp8.stride(1),
        b_fp8.stride(0),  # transposed
        output.stride(0),
        output.stride(1),
        scales_a.stride(0),
        scales_a.stride(1),
        scales_b.stride(0),
        scales_b.stride(1),
        BLOCK_K=block_size,
    )
    return output


def fused_block_scaled_dual_gemm_forward(
    act_fp8: torch.Tensor,
    w_hi_fp8: torch.Tensor,
    w_lo_fp8: torch.Tensor,
    act_scales: torch.Tensor,
    w_hi_scales: torch.Tensor,
    w_lo_scales: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Fused block-scaled 2-GEMM forward for TCFP-12.

    Computes ``sum_kb sa * (sbhi * dot(A_kb, Whi_kb^T) +
    sblo * dot(A_kb, Wlo_kb^T))``.

    Args:
        act_fp8:    (M, K) FP8 E4M3 activations, row-major.
        w_hi_fp8:   (N, K) FP8 E4M3 main weight, row-major.
        w_lo_fp8:   (N, K) FP8 E4M3 residual weight, row-major.
        act_scales:  (M, K // block_size) FP32 block scales.
        w_hi_scales: (N, K // block_size) FP32 block scales.
        w_lo_scales: (N, K // block_size) FP32 block scales.
        block_size: Scaling block size.

    Returns:
        (M, N) FP32 output tensor.
    """
    M, K = act_fp8.shape
    N, K2 = w_hi_fp8.shape
    assert K == K2 and w_lo_fp8.shape == (N, K), (
        f"Shape mismatch: act({M},{K}), w_hi({N},{K2}), w_lo{w_lo_fp8.shape}"
    )
    num_kb = K // block_size

    output = torch.empty((M, N), device=act_fp8.device, dtype=torch.float32)

    grid = lambda META: (  # noqa: E731
        triton.cdiv(M, META["BLOCK_M"])  # pyright: ignore[reportPossiblyUnboundVariable]
        * triton.cdiv(N, META["BLOCK_N"]),  # pyright: ignore[reportPossiblyUnboundVariable]
    )

    # W is (N, K) row-major. Kernel sees B = W.T = (K, N).
    _fused_block_scaled_dual_gemm_kernel[grid](  # pyright: ignore[reportPossiblyUnboundVariable]
        act_fp8,
        w_hi_fp8,
        w_lo_fp8,
        output,
        act_scales,
        w_hi_scales,
        w_lo_scales,
        M,
        N,
        K,
        num_kb,
        act_fp8.stride(0),
        act_fp8.stride(1),
        w_hi_fp8.stride(1),
        w_hi_fp8.stride(0),  # transposed
        w_lo_fp8.stride(1),
        w_lo_fp8.stride(0),  # transposed
        output.stride(0),
        output.stride(1),
        act_scales.stride(0),
        act_scales.stride(1),
        w_hi_scales.stride(0),
        w_hi_scales.stride(1),
        w_lo_scales.stride(0),
        w_lo_scales.stride(1),
        BLOCK_K=block_size,
    )
    return output


def fused_block_scaled_dual_gemm_backward_dx(
    grad_fp8: torch.Tensor,
    w_hi_fp8: torch.Tensor,
    w_lo_fp8: torch.Tensor,
    grad_scales: torch.Tensor,
    w_hi_scales: torch.Tensor,
    w_lo_scales: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Backward dX for block-scaled TCFP-12: ``dX = grad @ W_hi + grad @ W_lo``.

    Weight block scales are along D_in (last dim), but the backward GEMM's
    K-dimension is D_out — the scale axes don't align.  Rather than a
    complex re-blocking kernel, we dequantize to FP32 and use ``torch.matmul``
    which still runs on tensor cores for FP32.

    Args:
        grad_fp8:    (M, D_out) FP8 E5M2 gradient.
        w_hi_fp8:    (D_out, D_in) FP8 E4M3 main weight.
        w_lo_fp8:    (D_out, D_in) FP8 E4M3 residual weight.
        grad_scales: (M, D_out // block_size) FP32 block scales.
        w_hi_scales: (D_out, D_in // block_size) FP32 block scales.
        w_lo_scales: (D_out, D_in // block_size) FP32 block scales.
        block_size:  Scaling block size.

    Returns:
        (M, D_in) FP32 grad_input tensor.
    """
    w_hi_deq = block_dequantize_fp8(w_hi_fp8, w_hi_scales, block_size)
    w_lo_deq = block_dequantize_fp8(w_lo_fp8, w_lo_scales, block_size)
    grad_deq = block_dequantize_fp8(grad_fp8, grad_scales, block_size)

    return grad_deq @ w_hi_deq + grad_deq @ w_lo_deq
