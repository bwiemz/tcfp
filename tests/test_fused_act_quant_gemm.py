"""Tests for fused activation quantization + block-scaled dual GEMM kernel."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tcfp.core import TCFPMode

# Skip entire module if no CUDA
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    ),
]

DEVICE = "cuda"

_HAS_TRITON = False
try:
    from tcfp.kernels import (
        block_dequantize_fp8,
        block_quantize_fp8,
        fused_act_quant_dual_gemm_forward,
        fused_block_scaled_dual_gemm_forward,
        is_triton_available,
    )

    _HAS_TRITON = is_triton_available()
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not _HAS_TRITON, reason="Triton not available"
)


# ── Reference: separate block_quantize_fp8 + dual GEMM path ──────────────


def _reference_forward(
    act: torch.Tensor,
    w_hi_fp8: torch.Tensor,
    w_lo_fp8: torch.Tensor,
    w_hi_scales: torch.Tensor,
    w_lo_scales: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference: separate block_quantize_fp8 + fused_block_scaled_dual_gemm_forward.

    Returns (output, act_fp8, act_scales).
    """
    act_fp8, act_scales = block_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
        act, block_size=block_size, fp8_dtype=torch.float8_e4m3fn,
    )
    output = fused_block_scaled_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
        act_fp8, w_hi_fp8, w_lo_fp8,
        act_scales, w_hi_scales, w_lo_scales,
        block_size=block_size,
    )
    return output, act_fp8, act_scales


def _make_test_data(
    M: int, N: int, K: int, block_size: int, act_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create test data: BF16/FP32 activation + FP8 weights + scales."""
    torch.manual_seed(42)
    act = torch.randn(M, K, device=DEVICE, dtype=act_dtype)

    # Create weights → quantize to FP8 + scales
    w = torch.randn(N, K, device=DEVICE)
    w_hi_fp8, w_hi_scales = block_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
        w, block_size=block_size, fp8_dtype=torch.float8_e4m3fn,
    )
    residual = w - block_dequantize_fp8(w_hi_fp8, w_hi_scales, block_size)  # type: ignore[reportPossiblyUnboundVariable]
    w_lo_fp8, w_lo_scales = block_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
        residual, block_size=block_size, fp8_dtype=torch.float8_e4m3fn,
    )
    return act, w_hi_fp8, w_lo_fp8, w_hi_scales, w_lo_scales


# ── Phase 1: Kernel correctness tests ────────────────────────────────────


class TestOutputCloseMatch:
    """Fused output matches separate block_quantize + dual GEMM path."""

    @requires_triton
    @pytest.mark.parametrize("M,N,K,bs", [
        (64, 64, 128, 128),
        (128, 256, 256, 128),
        (64, 128, 256, 64),
        (128, 128, 128, 32),
        (256, 512, 512, 128),
    ])
    def test_output_close_match(self, M: int, N: int, K: int, bs: int) -> None:
        act, w_hi, w_lo, s_hi, s_lo = _make_test_data(M, N, K, bs)

        # Reference (separate path)
        ref_out, ref_act_fp8, ref_act_scales = _reference_forward(
            act, w_hi, w_lo, s_hi, s_lo, bs,
        )

        # Fused path
        fused_out, fused_act_fp8, fused_act_scales = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi, w_lo, s_hi, s_lo, block_size=bs,
        )

        # GEMM output should be close (same quantization + dot product)
        assert torch.allclose(fused_out, ref_out, atol=1e-4, rtol=1e-3), (
            f"Max output diff: {(fused_out - ref_out).abs().max():.6e}"
        )


class TestSideOutputParity:
    """Side outputs (act_fp8, act_scales) match block_quantize_fp8."""

    @requires_triton
    def test_act_fp8_parity(self) -> None:
        act, w_hi, w_lo, s_hi, s_lo = _make_test_data(128, 128, 256, 128)
        _, ref_act_fp8, _ = _reference_forward(act, w_hi, w_lo, s_hi, s_lo, 128)
        _, fused_act_fp8, _ = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi, w_lo, s_hi, s_lo, block_size=128,
        )

        # Compare via float — FP8 bitwise parity best-effort
        ref_f = ref_act_fp8.float()
        fused_f = fused_act_fp8.float()
        assert torch.allclose(ref_f, fused_f, atol=0), (
            f"act_fp8 max diff: {(ref_f - fused_f).abs().max():.6e}"
        )

    @requires_triton
    def test_act_scales_parity(self) -> None:
        act, w_hi, w_lo, s_hi, s_lo = _make_test_data(128, 128, 256, 128)
        _, _, ref_scales = _reference_forward(act, w_hi, w_lo, s_hi, s_lo, 128)
        _, _, fused_scales = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi, w_lo, s_hi, s_lo, block_size=128,
        )

        assert torch.allclose(ref_scales, fused_scales, atol=0), (
            f"act_scales max diff: {(ref_scales - fused_scales).abs().max():.6e}"
        )


class TestOutputFormat:
    """Output shapes and dtypes are correct."""

    @requires_triton
    @pytest.mark.parametrize("M,N,K,bs", [
        (64, 128, 256, 128),
        (128, 64, 128, 64),
        (256, 256, 256, 32),
    ])
    def test_output_shapes(self, M: int, N: int, K: int, bs: int) -> None:
        act, w_hi, w_lo, s_hi, s_lo = _make_test_data(M, N, K, bs)
        out, act_fp8, act_scales = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi, w_lo, s_hi, s_lo, block_size=bs,
        )
        assert out.shape == (M, N)
        assert act_fp8.shape == (M, K)
        assert act_scales.shape == (K // bs, M)

    @requires_triton
    def test_output_dtypes(self) -> None:
        act, w_hi, w_lo, s_hi, s_lo = _make_test_data(128, 128, 256, 128)
        out, act_fp8, act_scales = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi, w_lo, s_hi, s_lo, block_size=128,
        )
        assert out.dtype == torch.float32
        assert act_fp8.dtype == torch.float8_e4m3fn
        assert act_scales.dtype == torch.float32


class TestInputDtypes:
    """Works with both BF16 and FP32 activation inputs."""

    @requires_triton
    def test_bf16_input(self) -> None:
        act, w_hi, w_lo, s_hi, s_lo = _make_test_data(
            128, 128, 256, 128, act_dtype=torch.bfloat16,
        )
        out, act_fp8, act_scales = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi, w_lo, s_hi, s_lo, block_size=128,
        )
        assert out.shape == (128, 128)
        assert torch.isfinite(out).all()

    @requires_triton
    def test_fp32_input(self) -> None:
        act, w_hi, w_lo, s_hi, s_lo = _make_test_data(
            128, 128, 256, 128, act_dtype=torch.float32,
        )
        out, act_fp8, act_scales = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi, w_lo, s_hi, s_lo, block_size=128,
        )
        assert out.shape == (128, 128)
        assert torch.isfinite(out).all()


class TestEdgeCases:
    """Edge cases: NaN/Inf, backward dequant, various block sizes."""

    @requires_triton
    def test_all_finite(self) -> None:
        act, w_hi, w_lo, s_hi, s_lo = _make_test_data(256, 256, 512, 128)
        out, act_fp8, act_scales = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi, w_lo, s_hi, s_lo, block_size=128,
        )
        assert torch.isfinite(out).all(), "NaN/Inf in output"
        assert torch.isfinite(act_fp8.float()).all(), "NaN/Inf in act_fp8"
        assert torch.isfinite(act_scales).all(), "NaN/Inf in act_scales"

    @requires_triton
    def test_backward_dequant(self) -> None:
        """block_dequantize_fp8(act_fp8, act_scales, bs) approximates input."""
        act, w_hi, w_lo, s_hi, s_lo = _make_test_data(128, 128, 256, 128)
        _, act_fp8, act_scales = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi, w_lo, s_hi, s_lo, block_size=128,
        )
        act_deq = block_dequantize_fp8(act_fp8, act_scales, 128)  # type: ignore[reportPossiblyUnboundVariable]
        # FP8 E4M3 has 3 mantissa bits → ~12.5% relative error max
        rel_err = (act_deq - act).norm() / act.norm()
        assert rel_err < 0.15, f"Dequant relative error too high: {rel_err:.4f}"

    @requires_triton
    @pytest.mark.parametrize("bs", [32, 64, 128])
    def test_block_sizes(self, bs: int) -> None:
        K = 256 if bs <= 128 else 512
        act, w_hi, w_lo, s_hi, s_lo = _make_test_data(128, 128, K, bs)
        out, act_fp8, act_scales = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi, w_lo, s_hi, s_lo, block_size=bs,
        )
        assert out.shape == (128, 128)
        assert act_scales.shape == (K // bs, 128)
        assert torch.isfinite(out).all()


# ── Phase 2: Integration + dispatch tests ────────────────────────────────


class TestIntegration:
    """End-to-end integration with TCFPLinear."""

    @requires_triton
    def test_tcfplinear_fwd_bwd(self) -> None:
        """TCFPLinear with block-scaled path: forward + backward."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            256, 128,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=128,
        ).to(DEVICE)

        x = torch.randn(4, 256, device=DEVICE, requires_grad=True)
        y = layer(x)
        assert y.shape == (4, 128)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 256)

    @requires_triton
    def test_training_convergence(self) -> None:
        """Loss decreases over training steps with fused kernel."""
        from tcfp.nn import TCFPLinear

        torch.manual_seed(123)
        model = nn.Sequential(
            TCFPLinear(
                256, 128, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, scale_block_size=128,
            ),
            nn.ReLU(),
            TCFPLinear(
                128, 64, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, scale_block_size=64,
            ),
        ).to(DEVICE)

        # Overfit a fixed batch (proves backward works correctly)
        x = torch.randn(16, 256, device=DEVICE)
        target = torch.randn(16, 64, device=DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        initial_loss = None
        for step in range(100):
            out = model(x)
            loss = nn.functional.mse_loss(out, target)
            if step == 0:
                initial_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()  # pyright: ignore[reportPossiblyUnboundVariable]
        assert final_loss < initial_loss * 0.5, (  # type: ignore[operator]
            f"Loss didn't decrease enough: {initial_loss:.4f} → {final_loss:.4f}"
        )

    @requires_triton
    def test_hp_grad_weight(self) -> None:
        """hp_grad_weight=True saves BF16 input for backward."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            256, 128,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=128,
            hp_grad_weight=True,
        ).to(DEVICE)

        x = torch.randn(4, 256, device=DEVICE, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
        assert layer.weight.grad is not None


class TestDispatchHeuristic:
    """Dispatch heuristic selects fused vs separate path based on N."""

    @requires_triton
    def test_small_n_fused_path(self) -> None:
        """Small N (num_n_tiles <= 4) uses fused path, produces valid output."""
        from tcfp.nn import TCFPLinear

        # N=256 → num_n_tiles = (256+127)//128 = 3 → fused path
        layer = TCFPLinear(
            256, 256,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=128,
        ).to(DEVICE)

        x = torch.randn(4, 256, device=DEVICE, requires_grad=True)
        y = layer(x)
        assert y.shape == (4, 256)
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert x.grad is not None

    @requires_triton
    def test_large_n_separate_path(self) -> None:
        """Large N (num_n_tiles > 4) uses separate path, produces valid output."""
        from tcfp.nn import TCFPLinear

        # N=640 → num_n_tiles = (640+127)//128 = 6 → separate path
        layer = TCFPLinear(
            256, 640,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=128,
        ).to(DEVICE)

        x = torch.randn(4, 256, device=DEVICE, requires_grad=True)
        y = layer(x)
        assert y.shape == (4, 640)
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert x.grad is not None

    @requires_triton
    def test_boundary_n_512(self) -> None:
        """N=512 → num_n_tiles = 5 → just above threshold (separate path)."""
        from tcfp.nn import TCFPLinear

        layer = TCFPLinear(
            256, 512,
            mode=TCFPMode.TCFP12,
            use_tensor_cores=True,
            scale_block_size=128,
        ).to(DEVICE)

        x = torch.randn(4, 256, device=DEVICE, requires_grad=True)
        y = layer(x)
        assert y.shape == (4, 512)
        assert torch.isfinite(y).all()
        y.sum().backward()
        assert x.grad is not None


class TestFusedActQuantSparsity:
    """Sparsity fast-path tests for fused act-quant dual GEMM."""

    @requires_triton
    def test_zero_lo_sparse_skip(self) -> None:
        """All-zero W_lo scales -> output matches W_hi-only path."""
        M, K, N = 32, 256, 64
        bs = 128

        torch.manual_seed(42)
        act = torch.randn(M, K, device=DEVICE, dtype=torch.bfloat16)

        w_hi = torch.randn(N, K, device=DEVICE)
        w_hi_fp8, w_hi_scales = block_quantize_fp8(w_hi, block_size=bs)  # type: ignore[reportPossiblyUnboundVariable]

        w_lo_fp8 = torch.zeros(N, K, device=DEVICE, dtype=torch.float8_e4m3fn)
        w_lo_scales = torch.zeros(K // bs, N, device=DEVICE)  # sentinel

        out, _, _ = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi_fp8, w_lo_fp8, w_hi_scales, w_lo_scales,
            block_size=bs, sparse_threshold=0.0,
        )

        # Reference: quantize act, single GEMM with W_hi only
        act_fp8, act_scales = block_quantize_fp8(act, block_size=bs)  # type: ignore[reportPossiblyUnboundVariable]
        num_kb = K // bs
        a_blocked = act_fp8.float().reshape(M, num_kb, bs)
        a_deq = (a_blocked * act_scales.t().unsqueeze(-1)).reshape(M, K)
        bhi_blocked = w_hi_fp8.float().reshape(N, num_kb, bs)
        bhi_deq = (bhi_blocked * w_hi_scales.t().unsqueeze(-1)).reshape(N, K)
        ref = a_deq @ bhi_deq.t()

        assert torch.allclose(out, ref, atol=1e-2)

    @requires_triton
    def test_training_convergence_with_sparsity(self) -> None:
        """Loss decreases even when sparsity threshold is active."""
        from tcfp.nn import TCFPLinear

        torch.manual_seed(99)
        model = nn.Sequential(
            TCFPLinear(
                256, 128, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, scale_block_size=128,
            ),
            nn.ReLU(),
            TCFPLinear(
                128, 64, mode=TCFPMode.TCFP12,
                use_tensor_cores=True, scale_block_size=64,
            ),
        ).to(DEVICE)

        x = torch.randn(8, 256, device=DEVICE)
        target = torch.randn(8, 64, device=DEVICE)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        criterion = nn.MSELoss()

        initial_loss = None
        for step in range(30):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, target)
            if step == 0:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()  # pyright: ignore[reportPossiblyUnboundVariable]
        assert final_loss < initial_loss * 0.8, (  # type: ignore[operator]
            f"Loss didn't decrease: {initial_loss:.4f} -> {final_loss:.4f}"
        )


# ── Test: Persistent Tiling ─────────────────────────────────────────────


class TestPersistentActQuant:
    """Verify persistent tiling for fused act-quant GEMM."""

    @requires_triton
    def test_persistent_side_outputs(self) -> None:
        """Large M: act_fp8 and act_scales match separate block_quantize_fp8."""
        M, N, K, bs = 256, 256, 256, 128
        act, w_hi_fp8, w_lo_fp8, w_hi_scales, w_lo_scales = _make_test_data(M, N, K, bs)

        output, act_fp8, act_scales = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi_fp8, w_lo_fp8, w_hi_scales, w_lo_scales,
            block_size=bs, persistent=True,
        )

        ref_fp8, ref_scales = block_quantize_fp8(  # type: ignore[reportPossiblyUnboundVariable]
            act, block_size=bs, fp8_dtype=torch.float8_e4m3fn,
        )

        assert act_fp8.shape == ref_fp8.shape
        assert act_scales.shape == ref_scales.shape
        assert torch.equal(act_fp8, ref_fp8)
        assert torch.equal(act_scales, ref_scales)

    @requires_triton
    def test_persistent_matches_non_persistent(self) -> None:
        """persistent=True matches persistent=False output and side outputs."""
        M, N, K, bs = 256, 256, 256, 128
        act, w_hi_fp8, w_lo_fp8, w_hi_scales, w_lo_scales = _make_test_data(M, N, K, bs)

        out_p, fp8_p, sc_p = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi_fp8, w_lo_fp8, w_hi_scales, w_lo_scales,
            block_size=bs, persistent=True,
        )
        out_np, fp8_np, sc_np = fused_act_quant_dual_gemm_forward(  # type: ignore[reportPossiblyUnboundVariable]
            act, w_hi_fp8, w_lo_fp8, w_hi_scales, w_lo_scales,
            block_size=bs, persistent=False,
        )

        assert torch.allclose(out_p, out_np, atol=1e-6)
        assert torch.equal(fp8_p, fp8_np)
        assert torch.equal(sc_p, sc_np)
