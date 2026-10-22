# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Whisper transcription for one chosen language."""

import numpy as np

from live_transcript.const import SAMPLE_RATE
from live_transcript.cuda import cuda_device_name
from live_transcript.native import WhisperModel
from live_transcript.text import usable_transcript
from live_transcript.transcript import TimedWord

MODEL_ALIASES = {
    "turbo": "large-v3-turbo",
    "large": "large-v3",
}
MODEL_NAMES = ("tiny", "base", "small", "medium", "large-v3-turbo", "large-v3")


def resolve_model(name: str) -> str:
    """Expand a model alias and reject unknown names."""
    resolved = MODEL_ALIASES.get(name, name)
    if resolved not in MODEL_NAMES:
        known = "tiny, base, small, medium, turbo, large"
        raise ValueError(f"Unknown model {name!r}. Choose from: {known}.")
    return resolved


class WhisperEngine:
    """Transcribe 16 kHz audio in English or French."""

    def __init__(self, model_name: str, language: str, *, use_gpu: bool) -> None:
        """Load Whisper. `language` is `en` or `fr` for the whole session."""
        if language not in {"en", "fr"}:
            raise ValueError("Language must be en or fr.")
        self.model_name = resolve_model(model_name)
        self.language = language
        self._model = _load_model(self.model_name, use_gpu=use_gpu)

    def transcribe_pcm(self, pcm: bytes) -> str | None:
        """Transcribe one short utterance already cut by the voice detector."""
        if len(pcm) < int(SAMPLE_RATE * 0.2) * 2:
            return None
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        try:
            segments, _info = self._model.transcribe(
                np.ascontiguousarray(audio),
                language=self.language,
                task="transcribe",
                beam_size=1,
                temperature=0.0,
                condition_on_previous_text=False,
                without_timestamps=True,
                word_timestamps=False,
                vad_filter=False,
            )
        except Exception as exc:
            print(f"Transcription failed: {exc}", flush=True)
            return None
        parts: list[str] = []
        for segment in segments:
            if segment.no_speech_prob > 0.6 or segment.compression_ratio > 2.4:
                continue
            parts.append(segment.text)
        return usable_transcript(" ".join(parts))

    def transcribe_timeline(self, audio: np.ndarray) -> list[TimedWord]:
        """Transcribe a whole recording and return word times."""
        if audio.size < int(SAMPLE_RATE * 0.2):
            return []
        try:
            segments, _info = self._model.transcribe(
                np.ascontiguousarray(audio, dtype=np.float32),
                language=self.language,
                task="transcribe",
                beam_size=5,
                temperature=0.0,
                condition_on_previous_text=True,
                without_timestamps=False,
                word_timestamps=True,
                vad_filter=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Transcription failed: {exc}") from exc
        words: list[TimedWord] = []
        for segment in segments:
            if segment.no_speech_prob > 0.6 or segment.compression_ratio > 2.4:
                continue
            segment_words = segment.words or []
            if segment_words:
                for word in segment_words:
                    words.append(TimedWord(float(word.start), float(word.end), word.word))
                continue
            if segment.text.strip():
                words.append(TimedWord(float(segment.start), float(segment.end), segment.text))
        return words


def _load_model(model_name: str, *, use_gpu: bool) -> WhisperModel:
    if use_gpu:
        print(f"Loading {model_name} on {cuda_device_name()}...", flush=True)
        try:
            model = WhisperModel(model_name, device="cuda", compute_type="float16")
        except Exception as exc:
            print(f"GPU load failed ({exc}). Using the CPU.", flush=True)
            return WhisperModel(model_name, device="cpu", compute_type="int8")
        print("Model ready.", flush=True)
        return model
    print(f"Loading {model_name} on CPU...", flush=True)
    model = WhisperModel(model_name, device="cpu", compute_type="int8")
    print("Model ready.", flush=True)
    return model
