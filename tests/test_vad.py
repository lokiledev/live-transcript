# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Voice activity detection on synthetic audio."""

import numpy as np

from live_transcript.const import SAMPLE_RATE
from live_transcript.vad import EnergyVad


def _pcm(seconds: float, amplitude: int) -> bytes:
    count = int(seconds * SAMPLE_RATE)
    time = np.arange(count)
    wave = amplitude * np.sin(2 * np.pi * 440 * time / SAMPLE_RATE)
    return wave.astype(np.int16).tobytes()


def _push_all(vad: EnergyVad, pcm: bytes) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    for offset in range(0, len(pcm), vad.frame_bytes):
        frame = pcm[offset : offset + vad.frame_bytes]
        if len(frame) < vad.frame_bytes:
            break
        speech = vad.push(frame)
        if speech is not None:
            found.append((speech.start_sample, speech.end_sample))
    trailing = vad.flush()
    if trailing is not None:
        found.append((trailing.start_sample, trailing.end_sample))
    return found


def test_silence_produces_no_utterance() -> None:
    vad = EnergyVad(pre_roll_ms=0, min_speech_ms=300, silence_ms=180)
    found = _push_all(vad, bytes(SAMPLE_RATE * 2))
    assert found == []


def test_a_tone_between_silence_is_one_utterance() -> None:
    vad = EnergyVad(pre_roll_ms=0, min_speech_ms=300, silence_ms=180)
    pcm = _pcm(0.5, 0) + _pcm(1.0, 8000) + _pcm(1.0, 0)
    found = _push_all(vad, pcm)
    assert len(found) == 1
    start, end = found[0]
    assert abs(start - int(0.5 * SAMPLE_RATE)) <= vad.frame_samples
    assert end - start > int(0.9 * SAMPLE_RATE)
