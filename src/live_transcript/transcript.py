# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Transcript records: JSONL is canonical, Markdown and SRT are views."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from live_transcript.session import SessionMeta
from live_transcript.text import usable_transcript


@dataclass(frozen=True)
class Utterance:
    """One spoken turn in a session."""

    id: int
    channel: str
    speaker: str
    speaker_live: str
    start: float
    end: float
    text: str
    language: str

    def to_json(self) -> dict[str, object]:
        """Return a JSON object with times rounded to milliseconds."""
        return {
            "id": self.id,
            "channel": self.channel,
            "speaker": self.speaker,
            "speaker_live": self.speaker_live,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "language": self.language,
        }


@dataclass
class _WordGroup:
    speaker: str
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TimedWord:
    """A word with times relative to one recording."""

    start: float
    end: float
    text: str


def utterance_from_json(payload: dict[str, object]) -> Utterance:
    """Parse one JSONL object."""
    return Utterance(
        id=_int(payload, "id"),
        channel=_str(payload, "channel"),
        speaker=_str(payload, "speaker"),
        speaker_live=_str(payload, "speaker_live"),
        start=_float(payload, "start"),
        end=_float(payload, "end"),
        text=_str(payload, "text"),
        language=_str(payload, "language"),
    )


def read_utterances(path: Path) -> list[Utterance]:
    """Read transcript.jsonl. A missing file is an empty transcript."""
    if not path.is_file():
        return []
    utterances: list[Utterance] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parsed: object = json.loads(line)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{path}:{line_number} is not an object.")
        utterances.append(utterance_from_json(cast("dict[str, object]", parsed)))
    return utterances


def speaker_for_interval(
    turns: list[tuple[float, float, str]] | tuple[tuple[float, float, str], ...],
    start: float,
    end: float,
) -> str | None:
    """Return the speaker whose turn overlaps this interval the most."""
    if end < start:
        start, end = end, start
    best_name: str | None = None
    best_overlap = 0.0
    for turn_start, turn_end, name in turns:
        overlap = min(end, turn_end) - max(start, turn_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
    return best_name


def group_words(
    words: list[TimedWord],
    turns: list[tuple[float, float, str]],
    *,
    default_speaker: str,
    gap: float = 0.8,
) -> list[tuple[str, float, float, str]]:
    """Merge consecutive words from the same speaker into utterances."""
    groups: list[_WordGroup] = []
    for word in words:
        if not word.text.strip():
            continue
        speaker = speaker_for_interval(turns, word.start, word.end) or default_speaker
        if groups and groups[-1].speaker == speaker and word.start - groups[-1].end <= gap:
            groups[-1].end = max(groups[-1].end, word.end)
            groups[-1].text += word.text
            continue
        groups.append(_WordGroup(speaker, word.start, word.end, word.text))

    merged: list[tuple[str, float, float, str]] = []
    for group in groups:
        usable = usable_transcript(group.text)
        if usable is None:
            continue
        merged.append((group.speaker, group.start, group.end, usable))
    return merged


def render_markdown(meta: SessionMeta, utterances: list[Utterance]) -> str:
    """Render the readable transcript."""
    lines = [f"# {meta['title'] or 'Live transcript'}", ""]
    lines.append(f"- Mode: {meta['mode']}")
    lines.append(f"- Started: {meta['started']}")
    lines.append(f"- Ended: {meta['ended'] or 'in progress'}")
    lines.append(f"- Language: {meta['language']}")
    lines.append(f"- Model: {meta['model']}")
    for channel in meta["channels"]:
        role = "Microphone" if channel["role"] == "local" else "Meeting"
        lines.append(f"- {role}: {channel['description']} (`{channel['file']}`)")
    lines.extend(["", "## Transcript", ""])
    ordered = sorted(utterances, key=lambda item: (item.start, item.id))
    if not ordered:
        lines.append("No speech detected.")
        lines.append("")
    for utterance in ordered:
        clock = f"{format_clock(utterance.start)}-{format_clock(utterance.end)}"
        lines.append(f"**[{clock}] {utterance.speaker}**")
        lines.append("")
        lines.append(utterance.text)
        lines.append("")
    stats = meta["stats"]
    if meta["ended"]:
        lines.extend(
            [
                "---",
                "",
                f"- Utterances: {stats['utterances']}",
                f"- Words: {stats['words']}",
                f"- Dropped chunks: {stats['dropped']}",
                "",
            ]
        )
    return "\n".join(lines)


def render_srt(utterances: list[Utterance]) -> str:
    """Render SubRip cues. The speaker name is part of the cue text."""
    blocks: list[str] = []
    ordered = sorted(utterances, key=lambda item: (item.start, item.id))
    for index, utterance in enumerate(ordered, start=1):
        end = utterance.end if utterance.end > utterance.start else utterance.start + 0.2
        text = utterance.text.replace("\n", " ")
        blocks.append(
            f"{index}\n{format_srt_time(utterance.start)} --> {format_srt_time(end)}\n"
            f"[{utterance.speaker}] {text}\n"
        )
    return "\n".join(blocks) + ("\n" if blocks else "")


def format_clock(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_srt_time(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""
    millis = round(max(0.0, seconds) * 1000)
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def word_count(utterances: list[Utterance]) -> int:
    """Count words in utterance text."""
    return sum(len(utterance.text.split()) for utterance in utterances)


class TranscriptStore:
    """Append-only JSONL plus Markdown and SRT views."""

    def __init__(self, directory: Path, meta: SessionMeta) -> None:
        """Create or continue transcript.jsonl in `directory`."""
        self.directory = directory
        self.meta = meta
        self.utterances: list[Utterance] = []
        self._next_id = 1
        self._path = directory / "transcript.jsonl"
        self._fh = self._path.open("a", encoding="utf-8")

    def add(
        self,
        *,
        channel: str,
        speaker: str,
        start: float,
        end: float,
        text: str,
        language: str,
    ) -> Utterance:
        """Append one utterance and refresh the readable views."""
        utterance = Utterance(
            id=self._next_id,
            channel=channel,
            speaker=speaker,
            speaker_live=speaker,
            start=start,
            end=end,
            text=text,
            language=language,
        )
        self._next_id += 1
        self.utterances.append(utterance)
        self._fh.write(json.dumps(utterance.to_json(), ensure_ascii=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self.write_views()
        return utterance

    def rewrite(self, utterances: list[Utterance]) -> None:
        """Replace the transcript, keeping the first live copy beside it."""
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        self._preserve_live_copy()
        ordered = sorted(utterances, key=lambda item: (item.start, item.id))
        payload = "".join(json.dumps(item.to_json(), ensure_ascii=False) + "\n" for item in ordered)
        _atomic_write(self._path, payload)
        self.utterances = ordered
        self._next_id = ordered[-1].id + 1 if ordered else 1
        self._fh = self._path.open("a", encoding="utf-8")
        self.write_views()

    def write_views(self) -> None:
        """Rewrite transcript.md and transcript.srt from memory."""
        _atomic_write(self.directory / "transcript.md", render_markdown(self.meta, self.utterances))
        _atomic_write(self.directory / "transcript.srt", render_srt(self.utterances))

    def close(self) -> None:
        """Close the JSONL file and write the views once more."""
        if self._fh.closed:
            self.write_views()
            return
        self._fh.flush()
        self._fh.close()
        self.write_views()

    def _preserve_live_copy(self) -> None:
        backup = self.directory / "transcript.live.jsonl"
        if backup.exists() or not self._path.is_file() or self._path.stat().st_size == 0:
            return
        backup.write_bytes(self._path.read_bytes())


def _atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"Transcript field {key} is missing.")
    return value


def _int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"Transcript field {key} is missing.")
    return value


def _float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"Transcript field {key} is missing.")
    return float(value)
