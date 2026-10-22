# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Live speaker bank and offline label updates."""

import numpy as np

from live_transcript.diarize import SpeakerBank, name_turns
from live_transcript.session import SessionMeta
from live_transcript.speakers import channel_is_diarized, label_recording
from live_transcript.transcript import Utterance


def test_speaker_bank_keeps_a_voice_and_opens_a_second() -> None:
    bank = SpeakerBank("Speaker", threshold=0.65)
    first = np.array([1.0, 0.0], dtype=np.float32)
    same = np.array([0.9, 0.1], dtype=np.float32)
    other = np.array([0.0, 1.0], dtype=np.float32)
    assert bank.assign(first, allow_new=True, start=0.0, end=1.0) == "Speaker 1"
    assert bank.assign(same, allow_new=True, start=2.0, end=3.0) == "Speaker 1"
    assert bank.assign(other, allow_new=True, start=4.0, end=5.5) == "Speaker 2"
    assert bank.assign(None, allow_new=False, start=5.6, end=6.0) == "Speaker 2"


def test_short_unknown_voice_does_not_create_a_speaker() -> None:
    bank = SpeakerBank("Remote", threshold=0.65)
    bank.assign(np.array([1.0, 0.0], dtype=np.float32), allow_new=True, start=0.0, end=1.0)
    label = bank.assign(
        np.array([0.0, 1.0], dtype=np.float32),
        allow_new=False,
        start=3.0,
        end=3.4,
    )
    assert label == "Remote"


def test_name_turns_uses_a_plain_name_for_one_local_speaker() -> None:
    assert name_turns([(0.0, 1.0, 3)], "Me", "Me") == [(0.0, 1.0, "Me")]
    assert name_turns([(0.0, 1.0, 4), (2.0, 3.0, 1)], "Remote", None) == [
        (0.0, 1.0, "Remote 1"),
        (2.0, 3.0, "Remote 2"),
    ]


def _meta() -> SessionMeta:
    return {
        "version": 1,
        "id": "session",
        "title": "",
        "mode": "visio",
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
                "start_offset": 0.0,
            },
            {
                "name": "meeting",
                "role": "remote",
                "device": "speakers",
                "description": "Speakers",
                "file": "meeting.wav",
                "start_offset": 1.0,
            },
        ],
        "stats": {"utterances": 0, "words": 0, "dropped": 0},
    }


def test_visio_microphone_stays_local_and_meeting_uses_turns() -> None:
    meta = _meta()
    calls: list[str] = []

    def diarize(name: str) -> list[tuple[float, float, str]]:
        calls.append(name)
        return [(0.0, 5.0, "Remote 1"), (5.0, 10.0, "Remote 2")]

    utterances = [
        Utterance(1, "mic", "Me", "Me", 0.2, 1.0, "Bonjour", "fr"),
        Utterance(2, "meeting", "Remote", "Remote", 5.2, 5.8, "Oui", "fr"),
    ]
    updated = label_recording(meta, utterances, diarize, diarize_mic=False)
    assert calls == ["meeting"]
    assert updated[0].speaker == "Me"
    assert updated[0].speaker_live == "Me"
    assert updated[1].speaker == "Remote 1"
    assert channel_is_diarized(meta, meta["channels"][0], diarize_mic=False) is False
    assert channel_is_diarized(meta, meta["channels"][1], diarize_mic=False) is True
