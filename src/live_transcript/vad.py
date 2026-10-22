# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Energy-based speech segmentation for live transcription."""

from collections import deque

import numpy as np

from live_transcript.const import SAMPLE_RATE


class SpeechChunk:
    """One utterance cut from a continuous PCM stream."""

    def __init__(self, pcm: bytes, start_sample: int, end_sample: int) -> None:
        """Store PCM and its sample range, measured from the start of the stream."""
        self.pcm = pcm
        self.start_sample = start_sample
        self.end_sample = end_sample


def rms_int16(frame: bytes) -> float:
    """Return the root-mean-square level of a signed 16-bit frame."""
    if not frame:
        return 0.0
    samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
    return float(np.sqrt(np.mean(samples * samples)))


class EnergyVad:
    """Cut utterances when the signal rises above the recent noise floor."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 30,
        silence_ms: int = 700,
        min_speech_ms: int = 400,
        max_speech_ms: int = 30_000,
        pre_roll_ms: int = 300,
        min_rms: float = 250.0,
        speech_margin: float = 4.0,
    ) -> None:
        """Configure frame size and the silence needed to close an utterance."""
        self.sample_rate = sample_rate
        self.frame_samples = sample_rate * frame_ms // 1000
        self.frame_bytes = self.frame_samples * 2
        self.silence_limit = max(1, silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_speech_frames = max(self.min_speech_frames, max_speech_ms // frame_ms)
        self.pre_frames = max(0, pre_roll_ms // frame_ms)
        self.min_rms = min_rms
        self.speech_margin = speech_margin
        self._noise = 50.0
        self._ring: deque[bytes] = deque(maxlen=self.pre_frames or None)
        self._voiced: list[bytes] = []
        self._in_speech = False
        self._silence = 0
        self._speech_start_sample = 0
        self._samples_seen = 0

    def push(self, frame: bytes) -> SpeechChunk | None:
        """Accept one frame. Return an utterance when one has just finished."""
        if len(frame) != self.frame_bytes:
            raise ValueError(f"Expected {self.frame_bytes} bytes, got {len(frame)}.")
        is_speech = self._is_speech(frame)
        self._samples_seen += self.frame_samples
        if not self._in_speech:
            if not is_speech:
                self._noise = 0.95 * self._noise + 0.05 * rms_int16(frame)
                if self.pre_frames:
                    self._ring.append(frame)
                return None
            self._in_speech = True
            self._silence = 0
            self._voiced = list(self._ring) if self.pre_frames else []
            self._voiced.append(frame)
            self._speech_start_sample = self._samples_seen - len(self._voiced) * self.frame_samples
            return None

        self._voiced.append(frame)
        if is_speech:
            self._silence = 0
            if len(self._voiced) >= self.max_speech_frames:
                return self._take()
            return None
        self._silence += 1
        if self._silence < self.silence_limit:
            return None
        return self._take()

    def flush(self) -> SpeechChunk | None:
        """Return a trailing utterance when the stream stops mid-speech."""
        if not self._in_speech:
            return None
        return self._take()

    def _is_speech(self, frame: bytes) -> bool:
        return rms_int16(frame) >= max(self.min_rms, self._noise * self.speech_margin)

    def _take(self) -> SpeechChunk | None:
        voiced = self._voiced
        start = self._speech_start_sample
        long_enough = len(voiced) >= self.min_speech_frames
        self._in_speech = False
        self._voiced = []
        self._ring.clear()
        self._silence = 0
        if not long_enough:
            return None
        pcm = b"".join(voiced)
        return SpeechChunk(pcm, start, start + len(pcm) // 2)
