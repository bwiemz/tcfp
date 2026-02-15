// TCFP Fused Dual FP8 GEMM — cuBLASLt Implementation
//
// Uses cuBLASLt's beta=1 accumulation to fuse two FP8 scaled matmuls:
//   D  = alpha * (A * scaleA) @ op(B_hi * scaleB_hi)          [beta=0]
//   D += alpha * (A * scaleA) @ op(B_lo * scaleB_lo)          [beta=1]
//
// Benefits vs two separate torch._scaled_mm calls:
//   - No intermediate FP32 buffer (saves M*N*4 bytes)
//   - No addition kernel (saves one kernel launch + bandwidth)
//   - L2 cache reuse of A between the two cuBLASLt calls
//
// Layout scheme (zero-copy — no data rearrangement):
//   A:   row-major (M, K_mm), ld=K_mm
//   B:   col-major (in_feat, out_feat), ld=in_feat
//        Interprets row-major W(out,in) data as col-major W^T(in,out)
//        Forward  transb=OP_N → op(B)=W^T  → act @ W^T
//        Backward transb=OP_T → op(B)=W    → grad @ W
//   C/D: row-major (M, N_mm), ld=N_mm

#include "fused_dual_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cublasLt.h>

#include <mutex>
#include <unordered_map>

namespace tcfp {

// ── Error checking ──────────────────────────────────────────────────

#define CUBLASLT_CHECK(expr)                                              \
    do {                                                                  \
        cublasStatus_t _status = (expr);                                  \
        TORCH_CHECK(                                                      \
            _status == CUBLAS_STATUS_SUCCESS,                              \
            "cuBLASLt error in " #expr ": status=",                       \
            static_cast<int>(_status));                                    \
    } while (0)

// ── cuBLASLt handle (process-global, thread-safe) ───────────────────

static cublasLtHandle_t get_handle() {
    static cublasLtHandle_t handle = nullptr;
    static std::once_flag flag;
    std::call_once(flag, [] {
        CUBLASLT_CHECK(cublasLtCreate(&handle));
        std::atexit([] {
            if (handle) {
                cublasLtDestroy(handle);
                handle = nullptr;
            }
        });
    });
    return handle;
}

// ── Workspace (one per CUDA device, lazily allocated) ───────────────

static constexpr size_t WORKSPACE_BYTES = 32 * 1024 * 1024;  // 32 MB

static torch::Tensor get_workspace(const torch::Device& device) {
    static std::unordered_map<int, torch::Tensor> workspaces;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);

    int idx = device.index();
    auto it = workspaces.find(idx);
    if (it == workspaces.end()) {
        workspaces[idx] = torch::empty(
            {static_cast<int64_t>(WORKSPACE_BYTES)},
            torch::TensorOptions().dtype(torch::kUInt8).device(device));
        return workspaces[idx];
    }
    return it->second;
}

// ── Dtype conversion ────────────────────────────────────────────────

static cudaDataType to_cuda_dtype(c10::ScalarType dtype) {
    switch (dtype) {
        case c10::ScalarType::Float8_e4m3fn:
            return CUDA_R_8F_E4M3;
        case c10::ScalarType::Float8_e5m2:
            return CUDA_R_8F_E5M2;
        case c10::ScalarType::Float:
            return CUDA_R_32F;
        case c10::ScalarType::Half:
            return CUDA_R_16F;
        case c10::ScalarType::BFloat16:
            return CUDA_R_16BF;
        default:
            TORCH_CHECK(false, "Unsupported dtype for cuBLASLt: ",
                        c10::toString(dtype));
            return CUDA_R_32F;  // unreachable
    }
}

// ── RAII wrappers for cuBLASLt descriptors ──────────────────────────

struct MatmulDescGuard {
    cublasLtMatmulDesc_t desc = nullptr;
    ~MatmulDescGuard() { if (desc) cublasLtMatmulDescDestroy(desc); }
};

struct MatrixLayoutGuard {
    cublasLtMatrixLayout_t layout = nullptr;
    ~MatrixLayoutGuard() { if (layout) cublasLtMatrixLayoutDestroy(layout); }
};

struct PreferenceGuard {
    cublasLtMatmulPreference_t pref = nullptr;
    ~PreferenceGuard() { if (pref) cublasLtMatmulPreferenceDestroy(pref); }
};

// ── Core: dual cuBLASLt matmul with beta=1 accumulation ────────────

static torch::Tensor dual_cublaslt_gemm(
    const torch::Tensor& a,      // (M, K_mm) FP8
    const torch::Tensor& w_hi,   // (out_feat, in_feat) FP8 row-major
    const torch::Tensor& w_lo,   // (out_feat, in_feat) FP8 row-major
    const torch::Tensor& s_a,    // scalar FP32 inv_scale
    const torch::Tensor& s_hi,   // scalar FP32 inv_scale
    const torch::Tensor& s_lo,   // scalar FP32 inv_scale
    bool is_backward              // false: A@W^T, true: A@W
) {
    // ── Validate inputs ─────────────────────────────────────────────
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(w_hi.is_contiguous(), "w_hi must be contiguous");
    TORCH_CHECK(w_lo.is_contiguous(), "w_lo must be contiguous");
    TORCH_CHECK(w_hi.sizes() == w_lo.sizes(),
                "w_hi and w_lo must have the same shape");
    TORCH_CHECK(s_a.numel() == 1 && s_hi.numel() == 1 && s_lo.numel() == 1,
                "Scale tensors must be scalars");

    const int64_t M = a.size(0);
    const int64_t K_mm = a.size(1);
    const int64_t out_feat = w_hi.size(0);
    const int64_t in_feat = w_hi.size(1);

    // Verify inner dimension
    const int64_t expected_k = is_backward ? out_feat : in_feat;
    TORCH_CHECK(K_mm == expected_k,
                "K dimension mismatch: a.size(1)=", K_mm,
                " but expected ", expected_k,
                " (is_backward=", is_backward, ")");

    const int64_t N_mm = is_backward ? in_feat : out_feat;

    // ── Allocate output ─────────────────────────────────────────────
    auto output = torch::empty(
        {M, N_mm},
        torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));

    auto handle = get_handle();
    auto stream = at::cuda::getCurrentCUDAStream(a.device().index()).stream();
    auto workspace = get_workspace(a.device());

    const cudaDataType a_type = to_cuda_dtype(a.scalar_type());
    const cudaDataType b_type = to_cuda_dtype(w_hi.scalar_type());

    // ── Matmul descriptor ───────────────────────────────────────────
    MatmulDescGuard dg;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(
        &dg.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = is_backward ? CUBLAS_OP_T : CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        dg.desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        dg.desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Scale A (device pointer to FP32 scalar)
    const void* d_scale_a = s_a.data_ptr();
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        dg.desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
        &d_scale_a, sizeof(d_scale_a)));

    // ── Matrix layouts ──────────────────────────────────────────────

    // A: row-major (M, K_mm)
    MatrixLayoutGuard la;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        &la.layout, a_type, M, K_mm, K_mm));
    cublasLtOrder_t row_order = CUBLASLT_ORDER_ROW;
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        la.layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &row_order, sizeof(row_order)));

    // B: col-major (in_feat, out_feat) — zero-copy reinterpretation of
    // row-major W(out_feat, in_feat) as col-major W^T(in_feat, out_feat).
    // Forward  (OP_N): op(B) = W^T(in_feat, out_feat)  → A(M,in) @ W^T(in,out) = (M,out)
    // Backward (OP_T): op(B) = W(out_feat, in_feat)    → A(M,out) @ W(out,in)  = (M,in)
    MatrixLayoutGuard lb;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        &lb.layout, b_type, in_feat, out_feat, in_feat));
    cublasLtOrder_t col_order = CUBLASLT_ORDER_COL;
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        lb.layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &col_order, sizeof(col_order)));

    // C/D: row-major (M, N_mm)
    MatrixLayoutGuard lc;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
        &lc.layout, CUDA_R_32F, M, N_mm, N_mm));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        lc.layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &row_order, sizeof(row_order)));

    // ── Algorithm heuristic ─────────────────────────────────────────
    PreferenceGuard pg;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pg.pref));
    size_t ws_bytes = WORKSPACE_BYTES;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pg.pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &ws_bytes, sizeof(ws_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic;
    int n_algos = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        handle, dg.desc, la.layout, lb.layout, lc.layout, lc.layout,
        pg.pref, 1, &heuristic, &n_algos));
    TORCH_CHECK(n_algos > 0,
                "cuBLASLt: no algorithm found for FP8 dual GEMM "
                "(M=", M, ", K=", K_mm, ", N=", N_mm, ", "
                "a_type=", c10::toString(a.scalar_type()), ", "
                "b_type=", c10::toString(w_hi.scalar_type()), ")");

    // ── GEMM 1: D = (A * sA) @ op(B_hi * sB_hi)  [beta=0] ─────────
    float alpha = 1.0f;
    float beta_zero = 0.0f;

    const void* d_scale_hi = s_hi.data_ptr();
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        dg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &d_scale_hi, sizeof(d_scale_hi)));

    CUBLASLT_CHECK(cublasLtMatmul(
        handle, dg.desc,
        &alpha,
        a.data_ptr(), la.layout,
        w_hi.data_ptr(), lb.layout,
        &beta_zero,
        output.data_ptr(), lc.layout,   // C (unused, beta=0)
        output.data_ptr(), lc.layout,   // D
        &heuristic.algo,
        workspace.data_ptr(), ws_bytes,
        stream));

    // ── GEMM 2: D += (A * sA) @ op(B_lo * sB_lo) [beta=1] ─────────
    float beta_one = 1.0f;

    const void* d_scale_lo = s_lo.data_ptr();
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        dg.desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
        &d_scale_lo, sizeof(d_scale_lo)));

    CUBLASLT_CHECK(cublasLtMatmul(
        handle, dg.desc,
        &alpha,
        a.data_ptr(), la.layout,
        w_lo.data_ptr(), lb.layout,
        &beta_one,                       // KEY: accumulate onto D
        output.data_ptr(), lc.layout,    // C = previous D
        output.data_ptr(), lc.layout,    // D
        &heuristic.algo,
        workspace.data_ptr(), ws_bytes,
        stream));

    return output;
}

// ── Public API ──────────────────────────────────────────────────────

torch::Tensor fused_dual_fp8_gemm_forward(
    const torch::Tensor& act,
    const torch::Tensor& w_hi,
    const torch::Tensor& w_lo,
    const torch::Tensor& s_act,
    const torch::Tensor& s_hi,
    const torch::Tensor& s_lo
) {
    return dual_cublaslt_gemm(act, w_hi, w_lo, s_act, s_hi, s_lo,
                              /*is_backward=*/false);
}

torch::Tensor fused_dual_fp8_gemm_backward_dx(
    const torch::Tensor& grad,
    const torch::Tensor& w_hi,
    const torch::Tensor& w_lo,
    const torch::Tensor& s_grad,
    const torch::Tensor& s_hi,
    const torch::Tensor& s_lo
) {
    return dual_cublaslt_gemm(grad, w_hi, w_lo, s_grad, s_hi, s_lo,
                              /*is_backward=*/true);
}

}  // namespace tcfp
