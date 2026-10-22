# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""NVIDIA device detection for Whisper."""

import subprocess

from live_transcript.native import ctranslate2


def cuda_available() -> bool:
    """Return whether CTranslate2 can see an NVIDIA GPU."""
    try:
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


def cuda_device_name() -> str:
    """Return the first NVIDIA device name, or a generic CUDA label."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=3,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "CUDA"
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return lines[0] if lines else "CUDA"
