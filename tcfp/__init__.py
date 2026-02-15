"""
TCFP — Tensor Core Floating Point
==================================

Hardware-accelerated precision-enhanced FP8 training formats.
Three modes, all using FP8 tensor cores:

- **TCFP-8**  (~8.25 bits): Enhanced block-scaled FP8 with NormalFloat-aware scaling
- **TCFP-12** (~12.25 bits): FP8 + 4-bit residual correction
- **TCFP-16** (~16.5 bits): Residual FP8 (double FP8, near-BF16 quality)
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
