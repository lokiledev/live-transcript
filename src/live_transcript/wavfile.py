# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Streaming WAV files that stay readable if the process stops."""

import struct
import wave
from pathlib import Path

import numpy as np

from live_transcript.const import SAMPLE_RATE


class StreamingWavWriter:
    """Write mono 16-bit PCM and keep the WAV header in sync."""

    def __init__(self, path: Path, sample_rate: int = SAMPLE_RATE) -> None:
        """Create `path` with an empty WAV header."""
        self.path = path
        self.sample_rate = sample_rate
        self.frames = 0
        self._closed = False
        self._fh = path.open("wb")
        self._write_header(0)

    def write(self, pcm: bytes) -> None:
        """Append signed 16-bit little-endian samples."""
        if self._closed:
            raise RuntimeError(f"{self.path} is already closed.")
        if len(pcm) % 2:
            raise ValueError("PCM byte length must be even.")
        self._fh.write(pcm)
        self.frames += len(pcm) // 2

    def checkpoint(self) -> None:
        """Flush samples and rewrite the header so the file can already be read."""
        if self._closed:
            return
        position = self._fh.tell()
        self._fh.seek(0)
        self._write_header(self.frames * 2)
        self._fh.seek(position)
        self._fh.flush()

    def close(self) -> None:
        """Write the final sizes and close the file."""
        if self._closed:
            return
        self._fh.seek(0)
        self._write_header(self.frames * 2)
        self._fh.close()
        self._closed = True

    def __enter__(self) -> "StreamingWavWriter":
        """Return this writer."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the file when leaving a `with` block."""
        self.close()

    def _write_header(self, data_bytes: int) -> None:
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + data_bytes,
            b"WAVE",
            b"fmt ",
            16,
            1,
            1,
            self.sample_rate,
            self.sample_rate * 2,
            2,
            16,
            b"data",
            data_bytes,
        )
        self._fh.write(header)


def read_pcm(path: Path, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Read a mono 16-bit WAV file recorded by this project."""
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise RuntimeError(f"{path} is not mono 16-bit PCM.")
        if wav.getframerate() != sample_rate:
            raise RuntimeError(f"{path} is {wav.getframerate()} Hz, expected {sample_rate}.")
        return wav.readframes(wav.getnframes())


def read_wav(path: Path, sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Return float32 samples in the range -1 to 1, and the sample rate."""
    pcm = read_pcm(path, sample_rate)
    if not pcm:
        return np.zeros(0, dtype=np.float32), sample_rate
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return np.ascontiguousarray(audio), sample_rate
