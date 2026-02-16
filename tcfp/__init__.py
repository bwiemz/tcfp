"""
TCFP — Tensor Core Floating Point
==================================

Hardware-accelerated precision-enhanced FP8 training format.

- **TCFP-12** (~12.5 bits): FP8 + FP8 residual correction (2-GEMM, matches BF16)
"""

from __future__ import annotations

from tcfp.core import (
    ErrorFeedbackState,
    FP8Config,
    TCFPMode,
    ensure_column_major,
    fp8_matmul,
    to_fp8_e4m3,
    to_fp8_e4m3_nf_aware,
    to_fp8_e5m2,
)

__all__ = [
    "ErrorFeedbackState",
    "FP8Config",
    "TCFPMode",
    "ensure_column_major",
    "fp8_matmul",
    "to_fp8_e4m3",
    "to_fp8_e4m3_nf_aware",
    "to_fp8_e5m2",
]

__version__ = "0.1.0"
