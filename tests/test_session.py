# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Session folders."""

from datetime import datetime
from pathlib import Path

import pytest

from live_transcript.session import (
    SessionMeta,
    allocate_session_dir,
    default_captures_dir,
    load_session,
    save_session,
    slugify,
)
from live_transcript.transcript import Utterance


def test_slugify_strips_accents() -> None:
    assert slugify("Point hebdo") == "point-hebdo"
    assert slugify("Réunion") == "reunion"


def test_allocate_session_dir_adds_a_suffix_when_the_name_exists(tmp_path: Path) -> None:
    now = datetime(2026, 10, 22, 10, 36, 0)
    first = allocate_session_dir(tmp_path, "visio", "Weekly", now)
    first.mkdir()
    second = allocate_session_dir(tmp_path, "visio", "Weekly", now)
    assert first.name == "2026-10-22T103600-visio-weekly"
    assert second.name == "2026-10-22T103600-visio-weekly-2"


def test_captures_dir_follows_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LIVE_TRANSCRIPT_DIR", str(tmp_path))
    assert default_captures_dir() == tmp_path


def test_session_round_trip(tmp_path: Path) -> None:
    meta: SessionMeta = {
        "version": 1,
        "id": "session",
        "title": "",
        "mode": "live",
        "started": "2026-10-22T10:36:00+02:00",
        "ended": None,
        "language": "fr",
        "model": "large-v3-turbo",
        "sample_rate": 16000,
        "local_name": "Me",
        "channels": [
            {
                "name": "mic",
                "role": "local",
                "device": "mic",
                "description": "Microphone",
                "file": "mic.wav",
                "start_offset": 0.25,
            }
        ],
        "stats": {"utterances": 1, "words": 2, "dropped": 0},
    }
    save_session(tmp_path, meta)
    loaded = load_session(tmp_path)
    assert loaded["language"] == "fr"
    assert loaded["channels"][0]["start_offset"] == 0.25
    assert Utterance(1, "mic", "Me", "Me", 0.0, 1.0, "Bonjour", "fr").language == "fr"
