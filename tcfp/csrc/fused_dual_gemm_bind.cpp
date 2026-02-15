// TCFP Fused Dual FP8 GEMM — pybind11 bindings

#include <torch/extension.h>
#include "fused_dual_gemm.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "TCFP fused dual FP8 GEMM via cuBLASLt beta=1 accumulation";
    m.def("forward", &tcfp::fused_dual_fp8_gemm_forward,
          "Fused dual FP8 GEMM forward (act @ W_hi^T + act @ W_lo^T)",
          py::arg("act"), py::arg("w_hi"), py::arg("w_lo"),
          py::arg("s_act"), py::arg("s_hi"), py::arg("s_lo"));
    m.def("backward_dx", &tcfp::fused_dual_fp8_gemm_backward_dx,
          "Fused dual FP8 GEMM backward dX (grad @ W_hi + grad @ W_lo)",
          py::arg("grad"), py::arg("w_hi"), py::arg("w_lo"),
          py::arg("s_grad"), py::arg("s_hi"), py::arg("s_lo"));
}
