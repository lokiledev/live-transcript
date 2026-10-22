# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Capture directories and session metadata."""

import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import TypedDict, cast


class ChannelMeta(TypedDict):
    """One recorded audio stream."""

    name: str
    role: str
    device: str
    description: str
    file: str
    start_offset: float


class SessionMeta(TypedDict):
    """Metadata stored in session.json."""

    version: int
    id: str
    title: str
    mode: str
    started: str
    ended: str | None
    language: str
    model: str
    sample_rate: int
    local_name: str
    channels: list[ChannelMeta]
    stats: dict[str, int]


def default_captures_dir() -> Path:
    """Return the default folder for session recordings."""
    override = os.environ.get("LIVE_TRANSCRIPT_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Captures" / "live-transcript"


def slugify(text: str) -> str:
    """Turn a title into a short folder suffix."""
    normalized = unicodedata.normalize("NFKD", text.casefold())
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    cleaned = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    return cleaned[:40].strip("-")


def allocate_session_dir(captures: Path, mode: str, title: str | None, now: datetime) -> Path:
    """Return a new session directory path. The directory is not created."""
    stamp = now.strftime("%Y-%m-%dT%H%M%S")
    slug = slugify(title) if title else ""
    base = f"{stamp}-{mode}" + (f"-{slug}" if slug else "")
    candidate = captures / base
    suffix = 2
    while candidate.exists():
        candidate = captures / f"{base}-{suffix}"
        suffix += 1
    return candidate


def save_session(directory: Path, meta: SessionMeta) -> None:
    """Write session.json atomically."""
    path = directory / "session.json"
    payload = json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def load_session(directory: Path) -> SessionMeta:
    """Read and check session.json."""
    path = directory / "session.json"
    if not path.is_file():
        raise RuntimeError(f"No session.json in {directory}.")
    parsed: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict) or parsed.get("version") != 1:
        raise RuntimeError(f"Unsupported session file: {path}.")
    payload = cast("dict[str, object]", parsed)
    channels_raw = payload.get("channels")
    if not isinstance(channels_raw, list):
        raise RuntimeError(f"{path} has no channels.")
    channels: list[ChannelMeta] = []
    for item in channels_raw:
        if not isinstance(item, dict):
            raise RuntimeError(f"{path} has a channel that is not an object.")
        channel = cast("dict[str, object]", item)
        channels.append(
            {
                "name": _string(channel, "name"),
                "role": _string(channel, "role"),
                "device": _string(channel, "device"),
                "description": _string(channel, "description"),
                "file": _string(channel, "file"),
                "start_offset": _number(channel, "start_offset"),
            }
        )
    language = _string(payload, "language")
    if language not in {"en", "fr"}:
        raise RuntimeError("Session language must be en or fr.")
    return {
        "version": 1,
        "id": _string(payload, "id"),
        "title": _string(payload, "title"),
        "mode": _string(payload, "mode"),
        "started": _string(payload, "started"),
        "ended": _optional_string(payload, "ended"),
        "language": language,
        "model": _string(payload, "model"),
        "sample_rate": int(_number(payload, "sample_rate")),
        "local_name": _string(payload, "local_name"),
        "channels": channels,
        "stats": _stats(payload.get("stats")),
    }


def latest_session(captures: Path) -> Path:
    """Return the newest session directory under `captures`."""
    if not captures.is_dir():
        raise RuntimeError(f"No captures in {captures}.")
    found = [path for path in captures.iterdir() if (path / "session.json").is_file()]
    if not found:
        raise RuntimeError(f"No captures in {captures}.")
    return max(found, key=lambda path: path.name)


def _stats(value: object) -> dict[str, int]:
    stats = {"utterances": 0, "words": 0, "dropped": 0}
    if not isinstance(value, dict):
        return stats
    raw = cast("dict[str, object]", value)
    for key in stats:
        item = raw.get(key, 0)
        if isinstance(item, int):
            stats[key] = item
    return stats


def _string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise RuntimeError(f"Session field {key} is missing.")
    return value


def _optional_string(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise RuntimeError(f"Session field {key} is not text.")


def _number(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError(f"Session field {key} is missing.")
    return float(value)
