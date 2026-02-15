// TCFP Fused Dual FP8 GEMM — cuBLASLt Extension
//
// Fuses two FP8 scaled matmuls into one logical operation using
// cuBLASLt's beta=1 accumulation: the second GEMM accumulates
// directly onto the first result, eliminating the intermediate
// buffer and addition kernel.
//
// Forward:  output = (act * s_act) @ (W_hi^T * s_hi) + (act * s_act) @ (W_lo^T * s_lo)
// Backward: dX     = (grad * s_grad) @ (W_hi * s_hi) + (grad * s_grad) @ (W_lo * s_lo)

#pragma once
#include <torch/types.h>

namespace tcfp {

/// Fused dual FP8 GEMM forward: output(M,N) = act(M,K) @ W_hi^T + act @ W_lo^T
///
/// @param act    (M, K) FP8 E4M3, row-major contiguous
/// @param w_hi   (N, K) FP8 E4M3, row-major contiguous (nn.Linear layout)
/// @param w_lo   (N, K) FP8 E4M3, row-major contiguous
/// @param s_act  scalar FP32 inverse-scale (dequantisation factor)
/// @param s_hi   scalar FP32 inverse-scale for W_hi
/// @param s_lo   scalar FP32 inverse-scale for W_lo
/// @return       (M, N) FP32 output
torch::Tensor fused_dual_fp8_gemm_forward(
    const torch::Tensor& act,
    const torch::Tensor& w_hi,
    const torch::Tensor& w_lo,
    const torch::Tensor& s_act,
    const torch::Tensor& s_hi,
    const torch::Tensor& s_lo);

/// Fused dual FP8 GEMM backward dX: dX(M,K) = grad(M,N) @ W_hi + grad @ W_lo
///
/// @param grad   (M, N) FP8 E5M2, row-major contiguous
/// @param w_hi   (N, K) FP8 E4M3, row-major contiguous
/// @param w_lo   (N, K) FP8 E4M3, row-major contiguous
/// @param s_grad scalar FP32 inverse-scale for gradient
/// @param s_hi   scalar FP32 inverse-scale for W_hi
/// @param s_lo   scalar FP32 inverse-scale for W_lo
/// @return       (M, K) FP32 grad_input
torch::Tensor fused_dual_fp8_gemm_backward_dx(
    const torch::Tensor& grad,
    const torch::Tensor& w_hi,
    const torch::Tensor& w_lo,
    const torch::Tensor& s_grad,
    const torch::Tensor& s_hi,
    const torch::Tensor& s_lo);

}  // namespace tcfp
