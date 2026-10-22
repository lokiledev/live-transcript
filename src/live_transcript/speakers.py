# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Speaker labels for a two-channel meeting or a shared microphone.

Visio mode already knows the local microphone is one person. Diarization is
used to split the meeting recording, and the room microphone in live mode.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace

from live_transcript.session import ChannelMeta, SessionMeta
from live_transcript.transcript import Utterance, speaker_for_interval

Diarize = Callable[[str], Sequence[tuple[float, float, str]]]


def channel_is_diarized(meta: SessionMeta, channel: ChannelMeta, *, diarize_mic: bool) -> bool:
    """Return whether this recording should be split into speakers."""
    if meta["mode"] == "live":
        return True
    if channel["role"] == "remote":
        return True
    return diarize_mic


def channel_naming(meta: SessionMeta, channel: ChannelMeta) -> tuple[str, str | None]:
    """Return the label prefix and the name to use when only one speaker is found.

    The second value is set for the local visio microphone so a single speaker
    stays "Me" instead of "Me 1".
    """
    if meta["mode"] == "visio" and channel["role"] == "local":
        name = meta["local_name"]
        return name, name
    if channel["role"] == "remote":
        return "Remote", None
    return "Speaker", None


def fallback_speaker(meta: SessionMeta, channel: ChannelMeta) -> str:
    """Label used when diarization finds nobody on this channel."""
    if channel["role"] == "local" and meta["mode"] == "visio":
        return meta["local_name"]
    if meta["mode"] == "live":
        return "Speaker"
    return "Remote"


def label_recording(
    meta: SessionMeta,
    utterances: Sequence[Utterance],
    diarize: Diarize,
    *,
    diarize_mic: bool,
) -> list[Utterance]:
    """Replace speaker names using diarization turns in each channel's clock.

    `diarize` is called with a channel name and returns turns whose times start
    at the beginning of that channel's WAV file.
    """
    cache: dict[str, Sequence[tuple[float, float, str]]] = {}

    def turns_for(name: str) -> Sequence[tuple[float, float, str]]:
        if name not in cache:
            cache[name] = list(diarize(name))
        return cache[name]

    updated: list[Utterance] = []
    for utterance in utterances:
        channel = _channel(meta, utterance.channel)
        if channel is not None and not channel_is_diarized(meta, channel, diarize_mic=diarize_mic):
            updated.append(replace(utterance, speaker=meta["local_name"]))
            continue
        assigned = speaker_for_interval(
            list(turns_for(utterance.channel)),
            utterance.start - _offset(meta, utterance.channel),
            utterance.end - _offset(meta, utterance.channel),
        )
        if assigned is None:
            updated.append(utterance)
        else:
            updated.append(replace(utterance, speaker=assigned))
    return updated


def _channel(meta: SessionMeta, name: str) -> ChannelMeta | None:
    return next((channel for channel in meta["channels"] if channel["name"] == name), None)


def _offset(meta: SessionMeta, name: str) -> float:
    channel = _channel(meta, name)
    if channel is None:
        return 0.0
    return float(channel["start_offset"])
