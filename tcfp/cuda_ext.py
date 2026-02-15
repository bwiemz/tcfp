"""
TCFP CUDA Extension -- Fused Dual FP8 GEMM via cuBLASLt
========================================================

JIT-compiles a C++ extension that calls cuBLASLt directly to fuse the
two FP8 scaled matmuls in TCFP-12's 2-GEMM residual architecture.

The extension uses beta=1 accumulation: the second GEMM accumulates
directly onto the first result, eliminating the intermediate FP32
buffer and the addition kernel.

Forward dispatch priority (highest to lowest):
  1. cuBLASLt C++ extension (this module)
  2. Triton fused kernel  (``tcfp.kernels``)
  3. Two ``torch._scaled_mm`` + add  (always available)

Backward dX uses Triton or ``_scaled_mm`` fallback because cuBLASLt
doesn't support E5M2 as A-type on SM 12.0+ (Blackwell).

Requires: CUDA Toolkit (``nvcc`` on PATH, ``CUDA_HOME`` set).
"""
from __future__ import annotations

import glob
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType

import torch

logger = logging.getLogger(__name__)

# -- Module-level state ----------------------------------------------------

_cuda_ext: ModuleType | None = None
_load_attempted: bool = False


def _get_cuda_arch_flags() -> list[str]:
    """Return nvcc gencode flags for the current GPU."""
    if not torch.cuda.is_available():
        return []
    cap = torch.cuda.get_device_capability()
    arch = cap[0] * 10 + cap[1]
    return [
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
    ]


def _find_cublas_import_lib(cuda_home: str) -> Path | None:
    """Find or generate a cublasLt import library compatible with PyTorch.

    On Windows, the CUDA Toolkit's ``cublasLt.lib`` binds to a
    version-specific DLL (e.g. ``cublasLt64_13.dll``).  If PyTorch
    bundles a different version (``cublasLt64_12.dll``), we generate
    an import library from PyTorch's DLL using ``dumpbin`` + ``lib``.
    """
    torch_lib = Path(torch.__file__).parent / "lib"
    cuda_lib = Path(cuda_home) / "lib" / "x64"

    # Find PyTorch's bundled cublasLt DLL
    torch_dlls = sorted(torch_lib.glob("cublasLt64_*.dll"))
    if not torch_dlls:
        # No PyTorch-bundled DLL; fall back to CUDA Toolkit's .lib
        lib_path = cuda_lib / "cublasLt.lib"
        return lib_path if lib_path.exists() else None

    torch_dll = torch_dlls[-1]  # highest version
    torch_ver = torch_dll.stem.split("_")[-1]  # e.g. "12"

    # Check if the CUDA Toolkit .lib matches PyTorch's DLL version
    toolkit_dlls = sorted(
        glob.glob(str(cuda_lib.parent.parent / "bin" / "cublasLt64_*.dll"))
    )
    if toolkit_dlls:
        toolkit_ver = Path(toolkit_dlls[-1]).stem.split("_")[-1]
        if toolkit_ver == torch_ver:
            # Versions match -- use toolkit's .lib directly
            lib_path = cuda_lib / "cublasLt.lib"
            return lib_path if lib_path.exists() else None

    # Version mismatch -- generate import lib from PyTorch's DLL
    logger.info(
        "CUDA toolkit cuBLAS version differs from PyTorch's "
        "(PyTorch has cublasLt64_%s.dll). Generating import library.",
        torch_ver,
    )

    cache_dir = Path(tempfile.gettempdir()) / "tcfp_cublas_cache"
    cache_dir.mkdir(exist_ok=True)
    gen_lib = cache_dir / "cublasLt.lib"
    gen_def = cache_dir / "cublasLt.def"

    if gen_lib.exists():
        return gen_lib

    try:
        # Extract exports from PyTorch's DLL
        result = subprocess.run(
            ["dumpbin", "/exports", str(torch_dll)],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning("dumpbin failed: %s", result.stderr)
            return None

        # Parse exports into a .def file
        # Format: "  ordinal  hint  RVA  name" (4+ whitespace-separated fields)
        lines = ["LIBRARY " + torch_dll.name, "EXPORTS"]
        in_exports = False
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if "ordinal" in stripped and "hint" in stripped:
                in_exports = True
                continue
            if in_exports and not stripped:
                # Empty line after summary header — skip it
                continue
            if in_exports:
                parts = stripped.split()
                if len(parts) >= 4 and parts[0].isdigit():
                    # ordinal hint RVA name [= forwarded]
                    lines.append("  " + parts[3])
                elif len(parts) == 0:
                    # End of exports section
                    break
        gen_def.write_text("\n".join(lines) + "\n")

        # Generate .lib from .def
        result = subprocess.run(
            ["lib", f"/def:{gen_def}", "/machine:x64", f"/out:{gen_lib}"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            logger.warning("lib.exe failed: %s", result.stderr)
            return None

        logger.info("Generated cublasLt import lib at %s", gen_lib)
        return gen_lib

    except FileNotFoundError:
        logger.warning(
            "dumpbin/lib.exe not found — cannot generate import library. "
            "Ensure Visual Studio Build Tools are on PATH."
        )
        return None
    except Exception as exc:
        logger.warning("Failed to generate import library: %s", exc)
        return None


def _try_load_cuda_ext() -> ModuleType | None:
    """JIT-compile the CUDA extension.  Returns the module or None."""
    global _cuda_ext, _load_attempted
    if _load_attempted:
        return _cuda_ext
    _load_attempted = True

    # Pre-flight checks
    if not torch.cuda.is_available():
        logger.debug("CUDA not available -- skipping cuBLASLt extension")
        return None

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        logger.debug("CUDA_HOME / CUDA_PATH not set -- skipping cuBLASLt extension")
        return None

    csrc = Path(__file__).parent / "csrc"
    cu_file = csrc / "fused_dual_gemm.cu"
    bind_file = csrc / "fused_dual_gemm_bind.cpp"
    if not cu_file.exists() or not bind_file.exists():
        logger.debug("C++ source files not found in %s", csrc)
        return None

    try:
        from torch.utils.cpp_extension import load

        arch_flags = _get_cuda_arch_flags()
        extra_cuda = ["-O3", *arch_flags]

        # Platform-specific link/compile flags
        cuda_lib = Path(cuda_home) / "lib" / "x64"
        extra_ldflags: list[str] = []
        extra_cflags = ["-O3"]
        if cuda_lib.exists():
            # Windows
            cublas_lib = _find_cublas_import_lib(cuda_home)
            if cublas_lib is None:
                logger.warning("Could not find/generate cublasLt import lib")
                return None
            extra_ldflags.append(f"/LIBPATH:{cublas_lib.parent}")
            extra_ldflags.append(cublas_lib.name)
            # MSVC: use standard conforming preprocessor to avoid
            # ambiguous 'std' symbol errors in PyTorch/CUDA headers
            extra_cflags = ["/O2", "/Zc:preprocessor"]
            extra_cuda.append("-Xcompiler=/Zc:preprocessor")
        else:
            # Linux
            extra_ldflags.append("-lcublasLt")

        cuda_include = Path(cuda_home) / "include"
        extra_include: list[str] = []
        if cuda_include.exists():
            extra_include.append(str(cuda_include))

        # Add PyTorch's lib dir for DLL search at load time
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.exists():
            os.add_dll_directory(str(torch_lib))

        ext: ModuleType = load(  # pyright: ignore[reportAssignmentType]
            name="tcfp_fused_dual_gemm",
            sources=[str(cu_file), str(bind_file)],
            extra_cflags=extra_cflags,
            extra_cuda_cflags=extra_cuda,
            extra_ldflags=extra_ldflags,
            extra_include_paths=[str(csrc), *extra_include],
            verbose=False,
        )
        _cuda_ext = ext
        logger.info("cuBLASLt fused dual GEMM extension loaded successfully")
        return ext

    except Exception as exc:
        logger.warning("Failed to compile cuBLASLt extension: %s", exc)
        return None


# -- Public API ------------------------------------------------------------


def is_cuda_ext_available() -> bool:
    """Return True if the cuBLASLt C++ extension is loaded."""
    return _try_load_cuda_ext() is not None


def cuda_ext_dual_gemm_forward(
    act_fp8: torch.Tensor,
    w_hi_fp8: torch.Tensor,
    w_lo_fp8: torch.Tensor,
    act_inv: torch.Tensor,
    w_hi_inv: torch.Tensor,
    w_lo_inv: torch.Tensor,
) -> torch.Tensor:
    """Fused dual FP8 GEMM forward via cuBLASLt.

    Same interface as ``tcfp.kernels.fused_dual_gemm_forward``.
    """
    ext = _try_load_cuda_ext()
    if ext is None:
        raise RuntimeError(
            "cuBLASLt extension not available. "
            "Install CUDA Toolkit and set CUDA_HOME."
        )
    return ext.forward(act_fp8, w_hi_fp8, w_lo_fp8, act_inv, w_hi_inv, w_lo_inv)


def cuda_ext_dual_gemm_backward_dx(
    grad_fp8: torch.Tensor,
    w_hi_fp8: torch.Tensor,
    w_lo_fp8: torch.Tensor,
    grad_inv: torch.Tensor,
    w_hi_inv: torch.Tensor,
    w_lo_inv: torch.Tensor,
) -> torch.Tensor:
    """Fused dual FP8 GEMM backward dX via cuBLASLt.

    Same interface as ``tcfp.kernels.fused_dual_gemm_backward_dx``.
    """
    ext = _try_load_cuda_ext()
    if ext is None:
        raise RuntimeError(
            "cuBLASLt extension not available. "
            "Install CUDA Toolkit and set CUDA_HOME."
        )
    return ext.backward_dx(
        grad_fp8, w_hi_fp8, w_lo_fp8, grad_inv, w_hi_inv, w_lo_inv,
    )
