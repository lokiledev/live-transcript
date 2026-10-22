# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Offline and live speaker diarization with sherpa-onnx.

The segmentation model is the ONNX build of pyannote segmentation 3.0.
The embedding model is WeSpeaker's VoxCeleb ResNet34. Both are downloaded on
first use. The release tag for the embedding models is spelled
"speaker-recongition-models" upstream; that spelling is the real URL.
"""

import os
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx

from live_transcript.const import SAMPLE_RATE
from live_transcript.wavfile import read_wav

_SEGMENTATION_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
)
_EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx"
)
_SEGMENTATION_DIR = "sherpa-onnx-pyannote-segmentation-3-0"
_EMBEDDING_NAME = "wespeaker_en_voxceleb_resnet34_LM.onnx"


@dataclass
class _Track:
    label: str
    centroid: np.ndarray
    count: int


class SpeakerBank:
    """Assign live utterances to speakers by cosine similarity of embeddings."""

    def __init__(self, prefix: str, threshold: float = 0.65, max_speakers: int = 8) -> None:
        """Create an empty bank. Labels look like "Remote 1"."""
        self.prefix = prefix
        self.threshold = threshold
        self.max_speakers = max_speakers
        self._tracks: list[_Track] = []
        self._last_label: str | None = None
        self._last_end = float("-inf")

    def assign(
        self,
        embedding: np.ndarray | None,
        *,
        allow_new: bool,
        start: float,
        end: float,
    ) -> str:
        """Return the speaker label for this embedding.

        Short or missing embeddings reuse the previous speaker when the gap is
        under two seconds. A new speaker is opened only when `allow_new` is set
        and the voice is far from the ones already heard.
        """
        if embedding is None:
            if self._last_label is not None and start - self._last_end <= 2.0:
                self._last_end = end
                return self._last_label
            return self.prefix

        vector = _l2_normalize(embedding)
        if float(np.linalg.norm(vector)) == 0.0:
            return self.prefix
        if not self._tracks:
            if not allow_new:
                return self.prefix
            return self._add(vector, end)

        scores = [float(np.dot(vector, track.centroid)) for track in self._tracks]
        best_index = int(np.argmax(scores))
        best = scores[best_index]
        at_capacity = len(self._tracks) >= self.max_speakers
        if best >= self.threshold or at_capacity or not allow_new:
            if not allow_new and best < 0.3 and not at_capacity:
                return self.prefix
            self._update(best_index, vector)
            label = self._tracks[best_index].label
            self._remember(label, end)
            return label
        return self._add(vector, end)

    def _add(self, vector: np.ndarray, end: float) -> str:
        label = f"{self.prefix} {len(self._tracks) + 1}"
        self._tracks.append(_Track(label, vector, 1))
        self._remember(label, end)
        return label

    def _update(self, index: int, vector: np.ndarray) -> None:
        track = self._tracks[index]
        total = track.count + 1
        merged = track.centroid * (track.count / total) + vector / total
        track.centroid = _l2_normalize(merged)
        track.count = total

    def _remember(self, label: str, end: float) -> None:
        self._last_label = label
        self._last_end = end


def model_cache_dir() -> Path:
    """Return the directory where diarization models are stored."""
    override = os.environ.get("LIVE_TRANSCRIPT_MODELS")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "live-transcript" / "models"


def ensure_diarization_models() -> tuple[Path, Path]:
    """Download the segmentation and embedding models when they are missing."""
    cache = model_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    segmentation = cache / _SEGMENTATION_DIR / "model.onnx"
    embedding = cache / _EMBEDDING_NAME
    if not _large_enough(segmentation, 1_000_000):
        archive = cache / "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
        _download(_SEGMENTATION_URL, archive, minimum_bytes=1_000_000)
        print("Extracting speaker segmentation model...", flush=True)
        with tarfile.open(archive, "r:bz2") as bundle:
            bundle.extractall(cache, filter="data")
        archive.unlink(missing_ok=True)
    if not _large_enough(segmentation, 1_000_000):
        raise RuntimeError("Speaker segmentation model did not extract.")
    if not _large_enough(embedding, 5_000_000):
        _download(_EMBEDDING_URL, embedding, minimum_bytes=5_000_000)
    return segmentation, embedding


class EmbeddingExtractor:
    """Speaker embeddings for short microphone or meeting chunks."""

    def __init__(self, model_path: Path) -> None:
        """Load an ONNX speaker-embedding model."""
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model_path),
            num_threads=2,
            debug=False,
            provider="cpu",
        )
        if not config.validate():
            raise RuntimeError(f"Speaker embedding model is invalid: {model_path}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embed(self, pcm: bytes) -> np.ndarray | None:
        """Return an embedding, or None when the chunk is too short."""
        if len(pcm) < int(SAMPLE_RATE * 0.5) * 2:
            return None
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        waveform = np.ascontiguousarray(samples)
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=waveform)
        stream.input_finished()
        if not self._extractor.is_ready(stream):
            return None
        return np.asarray(self._extractor.compute(stream), dtype=np.float32)


def load_embedding_extractor() -> EmbeddingExtractor:
    """Download models if needed and return an embedding extractor."""
    _segmentation, embedding = ensure_diarization_models()
    return EmbeddingExtractor(embedding)


def diarize_channel(
    path: Path,
    *,
    num_speakers: int,
    threshold: float,
    prefix: str,
    single_name: str | None,
) -> list[tuple[float, float, str]]:
    """Diarize one WAV file. Times are seconds from the start of that file."""
    audio, rate = read_wav(path)
    if rate != SAMPLE_RATE:
        raise RuntimeError(f"{path} is {rate} Hz, expected {SAMPLE_RATE}.")
    raw = _diarize_samples(audio, num_speakers=num_speakers, threshold=threshold)
    return name_turns(raw, prefix, single_name)


def name_turns(
    raw: list[tuple[float, float, int]],
    prefix: str,
    single_name: str | None,
) -> list[tuple[float, float, str]]:
    """Map cluster indexes to stable labels in order of first appearance."""
    order: list[int] = []
    for _start, _end, index in raw:
        if index not in order:
            order.append(index)
    if len(order) == 1 and single_name:
        names = {order[0]: single_name}
    else:
        stem = single_name or prefix
        names = {index: f"{stem} {number}" for number, index in enumerate(order, start=1)}
    return [(start, end, names[index]) for start, end, index in raw]


def _diarize_samples(
    audio: np.ndarray,
    *,
    num_speakers: int,
    threshold: float,
) -> list[tuple[float, float, int]]:
    if audio.size < SAMPLE_RATE:
        return []
    segmentation, embedding = ensure_diarization_models()
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(segmentation),
            ),
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(embedding),
            num_threads=4,
            debug=False,
            provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers,
            threshold=threshold,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise RuntimeError("Speaker diarization configuration is invalid.")
    diarizer = sherpa_onnx.OfflineSpeakerDiarization(config)
    if diarizer.sample_rate != SAMPLE_RATE:
        raise RuntimeError(
            f"Diarization model expects {diarizer.sample_rate} Hz, not {SAMPLE_RATE}."
        )
    waveform = np.ascontiguousarray(audio, dtype=np.float32)
    last_bucket = -1

    def _progress(done: int, total: int) -> int:
        nonlocal last_bucket
        if total <= 0:
            return 0
        percent = int(done * 100 / total)
        bucket = percent // 10
        if bucket != last_bucket:
            last_bucket = bucket
            print(f"  diarization {percent}%", flush=True)
        return 0

    result = diarizer.process(waveform, callback=_progress).sort_by_start_time()
    return [(float(item.start), float(item.end), int(item.speaker)) for item in result]


def _download(url: str, destination: Path, *, minimum_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name}...", flush=True)
    request = urllib.request.Request(url, headers={"User-Agent": "live-transcript"})
    total = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            total = int(response.headers.get("Content-Length", "0") or "0")
            written = 0
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                written += len(block)
                if total:
                    print(f"\r  {written / total:.0%}", end="", flush=True)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download {url}: {exc}") from exc
    if total:
        print()
    if not _large_enough(temporary, minimum_bytes):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Download of {destination.name} was incomplete.")
    temporary.replace(destination)


def _large_enough(path: Path, minimum_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= minimum_bytes


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    flat = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(flat))
    if norm == 0.0:
        return flat
    return flat / norm
