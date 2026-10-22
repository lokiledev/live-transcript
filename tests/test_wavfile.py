# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""WAV round trip."""

from pathlib import Path

from live_transcript.wavfile import StreamingWavWriter, read_pcm


def test_checkpoint_keeps_every_sample(tmp_path: Path) -> None:
    path = tmp_path / "mic.wav"
    first = b"\x01\x00" * 100
    second = b"\x02\x00" * 50
    writer = StreamingWavWriter(path)
    writer.write(first)
    writer.checkpoint()
    writer.write(second)
    writer.close()
    assert read_pcm(path) == first + second
