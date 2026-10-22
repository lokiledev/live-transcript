# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Load native libraries that need setup before their first import.

CTranslate2 wheels look for the CUDA 12 cuBLAS sonames next to the NVIDIA
package installed with this project. Those libraries are loaded first, then
CTranslate2 and faster-whisper are imported for the rest of the package.
"""

import ctypes
import glob
import io
import sys
import warnings


def install() -> None:
    """Silence library warnings, use UTF-8 stdout, and preload CUDA 12 cuBLAS."""
    warnings.filterwarnings("ignore")
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    _preload_cuda12_cublas()


def _preload_cuda12_cublas() -> None:
    """Make the pip-installed CUDA 12 cuBLAS visible to CTranslate2."""
    try:
        import nvidia.cublas  # noqa: PLC0415
    except ImportError:
        return
    for path in nvidia.cublas.__path__:
        for lib in ("libcublasLt.so.12", "libcublas.so.12"):
            for match in glob.glob(f"{path}/lib/{lib}"):
                ctypes.CDLL(match, mode=ctypes.RTLD_GLOBAL)


install()

import ctranslate2
from faster_whisper import WhisperModel

__all__ = ["WhisperModel", "ctranslate2"]
