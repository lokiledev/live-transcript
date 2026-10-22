# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Transcript views and speaker assignment."""

from pathlib import Path

from live_transcript.session import SessionMeta
from live_transcript.text import usable_transcript
from live_transcript.transcript import (
    TimedWord,
    TranscriptStore,
    Utterance,
    format_srt_time,
    group_words,
    render_markdown,
    render_srt,
    speaker_for_interval,
)


def _meta() -> SessionMeta:
    return {
        "version": 1,
        "id": "2026-10-22T103600-visio",
        "title": "Weekly",
        "mode": "visio",
        "started": "2026-10-22T10:36:00+02:00",
        "ended": None,
        "language": "en",
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
                "start_offset": 0.0,
            },
            {
                "name": "meeting",
                "role": "remote",
                "device": "speakers",
                "description": "Speakers",
                "file": "meeting.wav",
                "start_offset": 0.2,
            },
        ],
        "stats": {"utterances": 0, "words": 0, "dropped": 0},
    }


def test_speaker_for_interval_picks_the_larger_overlap() -> None:
    turns = [(0.0, 5.0, "A"), (4.0, 10.0, "B")]
    assert speaker_for_interval(turns, 0.0, 1.0) == "A"
    assert speaker_for_interval(turns, 4.5, 9.0) == "B"
    assert speaker_for_interval(turns, 20.0, 21.0) is None


def test_group_words_merges_a_short_gap_and_splits_speakers() -> None:
    words = [
        TimedWord(0.0, 0.4, " Hello"),
        TimedWord(0.5, 1.0, " there"),
        TimedWord(5.0, 5.4, " Next"),
    ]
    groups = group_words(words, [(0.0, 2.0, "Me"), (4.0, 6.0, "Remote 1")], default_speaker="Me")
    assert groups == [
        ("Me", 0.0, 1.0, "Hello there"),
        ("Remote 1", 5.0, 5.4, "Next"),
    ]


def test_markdown_is_chronological_and_srt_names_the_speaker() -> None:
    meta = _meta()
    utterances = [
        Utterance(2, "meeting", "Remote 1", "Remote", 5.0, 6.25, "Later", "en"),
        Utterance(1, "mic", "Me", "Me", 1.0, 2.0, "Earlier", "en"),
    ]
    markdown = render_markdown(meta, utterances)
    assert markdown.index("Earlier") < markdown.index("Later")
    assert "# Weekly" in markdown
    assert "[00:00:01-00:00:02] Me" in markdown
    srt = render_srt(utterances)
    assert srt.startswith("1\n00:00:01,000 --> 00:00:02,000\n[Me] Earlier")
    assert format_srt_time(6.25) == "00:00:06,250"


def test_store_keeps_the_live_copy_when_labels_change(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path, _meta())
    store.add(channel="mic", speaker="Me", start=1.0, end=2.0, text="Hello", language="en")
    updated = [
        Utterance(1, "mic", "Host", "Me", 1.0, 2.0, "Hello", "en"),
    ]
    store.rewrite(updated)
    store.close()
    live = (tmp_path / "transcript.live.jsonl").read_text(encoding="utf-8")
    current = (tmp_path / "transcript.jsonl").read_text(encoding="utf-8")
    assert '"speaker": "Me"' in live
    assert '"speaker": "Host"' in current
    assert "Hello" in (tmp_path / "transcript.md").read_text(encoding="utf-8")


def test_usable_transcript_drops_known_hallucinations() -> None:
    assert usable_transcript("Thanks for watching.") is None
    assert usable_transcript("Sous-titres réalisés par la communauté d'Amara.org") is None
    assert usable_transcript("Bonjour tout le monde") == "Bonjour tout le monde"
    assert usable_transcript("yes yes yes yes") == "yes"
