# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""PipeWire graph parsing and device selection."""

from live_transcript.audio import (
    DeviceMatchError,
    graph_from_dump,
    match_device,
    record_command,
    select_meeting,
    select_microphone,
)
from live_transcript.const import SAMPLE_RATE

DUMP = [
    {
        "id": 10,
        "type": "PipeWire:Interface:Metadata",
        "metadata": [
            {"key": "default.audio.sink", "value": {"name": "sink.speakers"}},
            {"key": "default.audio.source", "value": {"name": "source.mic"}},
        ],
    },
    {
        "id": 30,
        "type": "PipeWire:Interface:Node",
        "info": {
            "state": "running",
            "props": {
                "node.name": "sink.speakers",
                "node.description": "Speakers",
                "media.class": "Audio/Sink",
            },
        },
    },
    {
        "id": 31,
        "type": "PipeWire:Interface:Node",
        "info": {
            "state": "suspended",
            "props": {
                "node.name": "sink.hdmi",
                "node.description": "HDMI",
                "media.class": "Audio/Sink",
            },
        },
    },
    {
        "id": 32,
        "type": "PipeWire:Interface:Node",
        "info": {
            "state": "running",
            "props": {
                "node.name": "source.mic",
                "node.description": "Microphone",
                "media.class": "Audio/Source",
            },
        },
    },
    {
        "id": 33,
        "type": "PipeWire:Interface:Node",
        "info": {
            "state": "suspended",
            "props": {
                "node.name": "source.webcam",
                "node.description": "Webcam",
                "media.class": "Audio/Source",
            },
        },
    },
    {
        "id": 34,
        "type": "PipeWire:Interface:Node",
        "info": {
            "state": "running",
            "props": {
                "node.name": "source.monitor",
                "node.description": "Monitor",
                "media.class": "Audio/Source",
            },
        },
    },
    {
        "id": 40,
        "type": "PipeWire:Interface:Node",
        "info": {
            "state": "running",
            "props": {
                "node.name": "chrome",
                "application.name": "Google Chrome",
                "media.class": "Stream/Output/Audio",
            },
        },
    },
    {
        "id": 50,
        "type": "PipeWire:Interface:Link",
        "info": {"state": "active", "output-node-id": 40, "input-node-id": 30},
    },
]


def test_graph_skips_monitors_and_marks_playing_apps() -> None:
    graph = graph_from_dump(DUMP)
    names = [device.name for device in graph.microphones]
    assert names == ["source.mic", "source.webcam"]
    assert graph.default_microphone == "source.mic"
    speakers = next(device for device in graph.sinks if device.name == "sink.speakers")
    assert speakers.apps == ("Google Chrome",)
    assert "source.monitor" not in names


def test_default_microphone_is_used_without_a_prompt() -> None:
    graph = graph_from_dump(DUMP)

    def fail(_prompt: str) -> str:
        raise AssertionError("prompted")

    device = select_microphone(graph, None, choose=False, interactive=True, input_func=fail)
    assert device.name == "source.mic"


def test_choose_prompts_for_a_microphone() -> None:
    graph = graph_from_dump(DUMP)
    device = select_microphone(
        graph,
        None,
        choose=True,
        interactive=True,
        input_func=lambda _prompt: "2",
    )
    assert device.description == "Webcam"


def test_playing_output_is_the_meeting() -> None:
    graph = graph_from_dump(DUMP)

    def fail(_prompt: str) -> str:
        raise AssertionError("prompted")

    device = select_meeting(graph, None, choose=False, interactive=False, input_func=fail)
    assert device.name == "sink.speakers"


def test_match_device_accepts_a_unique_substring() -> None:
    graph = graph_from_dump(DUMP)
    assert match_device(graph.microphones, "web").description == "Webcam"


def test_match_device_rejects_an_ambiguous_query() -> None:
    graph = graph_from_dump(DUMP)
    try:
        match_device(graph.microphones, "source")
    except DeviceMatchError as exc:
        assert "more than one" in str(exc)
    else:
        raise AssertionError("expected an error")


def test_record_command_monitors_a_sink_and_not_a_microphone() -> None:
    mic = record_command("source.mic", monitor_sink=False)
    meeting = record_command("sink.speakers", monitor_sink=True)
    assert "stream.capture.sink=true" not in mic
    assert mic[mic.index("--rate") + 1] == str(SAMPLE_RATE)
    assert meeting[1:5] == ["--target", "sink.speakers", "--properties", "stream.capture.sink=true"]
    assert meeting[-3:] == ["s16", "--raw", "-"]
