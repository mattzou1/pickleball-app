"""Preload CUDA 12 runtime libraries for onnxruntime-gpu.

torch ships its own CUDA (potentially 13.x); onnxruntime-gpu wheels on PyPI
need CUDA 12.x. When both are installed side-by-side, onnxruntime silently
falls back to CPU unless it can find libcublasLt.so.12, libcudnn.so.9, etc.

This module ctypes-loads those libs from the pip-installed `nvidia-*-cu12`
wheels so onnxruntime's CUDAExecutionProvider binds successfully. Import
this module BEFORE `onnxruntime` or any library that imports it (rtmlib).
No-op if the wheels aren't installed.
"""

from __future__ import annotations

import ctypes
import os
import site
from pathlib import Path

# cu12 libs (for onnxruntime-gpu). Order matters: lower-level first.
_LIBS_CU12 = [
    ("cuda_runtime/lib", "libcudart.so.12"),
    ("cuda_nvrtc/lib", "libnvrtc.so.12"),
    ("nvjitlink/lib", "libnvJitLink.so.12"),
    ("cublas/lib", "libcublas.so.12"),
    ("cublas/lib", "libcublasLt.so.12"),
    ("cufft/lib", "libcufft.so.11"),
    ("curand/lib", "libcurand.so.10"),
    ("cudnn/lib", "libcudnn.so.9"),
]

# cu13 libs (for torch CUDA 13 — libnvrtc-builtins must load before torch JIT).
_LIBS_CU13 = [
    ("cu13/lib", "libnvrtc-builtins.so.13.0"),
    ("cu13/lib", "libnvrtc.so.13"),
]


def preload() -> None:
    for site_dir in site.getsitepackages():
        nvidia_root = Path(site_dir) / "nvidia"
        if nvidia_root.is_dir():
            break
    else:
        return

    for subdir, soname in _LIBS_CU13 + _LIBS_CU12:
        so = nvidia_root / subdir / soname
        if so.is_file():
            try:
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


preload()
