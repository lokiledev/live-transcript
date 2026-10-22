# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""PipeWire microphones and playback monitors."""

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from live_transcript.const import SAMPLE_RATE


class DeviceMatchError(ValueError):
    """The requested device name did not match one device."""


@dataclass(frozen=True)
class Device:
    """A microphone or a playback device."""

    name: str
    description: str
    apps: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Human-readable name, including applications currently playing."""
        base = self.description or self.name
        if not self.apps:
            return base
        return f"{base} ({', '.join(self.apps)})"


@dataclass(frozen=True)
class AudioGraph:
    """Microphones and playback devices reported by PipeWire."""

    microphones: tuple[Device, ...]
    sinks: tuple[Device, ...]
    default_microphone: str | None
    default_sink: str | None


def require_pipewire() -> None:
    """Fail with a clear message when the PipeWire tools are missing."""
    missing = [tool for tool in ("pw-dump", "pw-record") if shutil.which(tool) is None]
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(f"Missing PipeWire tools: {names}. Install pipewire.")


def record_command(target: str, *, monitor_sink: bool) -> list[str]:
    """Build a pw-record command that writes 16 kHz mono PCM to stdout.

    A microphone is recorded directly. Meeting audio is the monitor of a
    playback device, which is what the computer is playing.
    """
    command = ["pw-record", "--target", target]
    if monitor_sink:
        command.extend(["--properties", "stream.capture.sink=true"])
    command.extend(
        [
            "--rate",
            str(SAMPLE_RATE),
            "--channels",
            "1",
            "--format",
            "s16",
            "--raw",
            "-",
        ]
    )
    return command


def load_graph() -> AudioGraph:
    """Read the current PipeWire graph."""
    require_pipewire()
    try:
        result = subprocess.run(
            ["pw-dump"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("pw-dump failed.") from exc
    try:
        parsed: object = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("pw-dump did not return JSON.") from exc
    if not isinstance(parsed, list):
        raise RuntimeError("pw-dump did not return a list.")
    return graph_from_dump(parsed)


def graph_from_dump(objects: Sequence[object]) -> AudioGraph:
    """Build an audio graph from a pw-dump JSON list."""
    nodes: dict[int, _Node] = {}
    default_microphone: str | None = None
    default_sink: str | None = None
    for item in objects:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "PipeWire:Interface:Metadata":
            mic, sink = _defaults(item.get("metadata"))
            default_microphone = mic or default_microphone
            default_sink = sink or default_sink
            continue
        if kind != "PipeWire:Interface:Node":
            continue
        node_id = item.get("id")
        if not isinstance(node_id, int):
            continue
        node = _node(item.get("info"))
        if node is not None:
            nodes[node_id] = node

    apps_by_sink: dict[int, set[str]] = {}
    for item in objects:
        if not isinstance(item, dict) or item.get("type") != "PipeWire:Interface:Link":
            continue
        info = item.get("info")
        if not isinstance(info, dict) or info.get("state") != "active":
            continue
        source_id = info.get("output-node-id")
        sink_id = info.get("input-node-id")
        if not isinstance(source_id, int) or not isinstance(sink_id, int):
            continue
        source = nodes.get(source_id)
        sink = nodes.get(sink_id)
        if source is None or sink is None:
            continue
        if source.media != "Stream/Output/Audio" or sink.media != "Audio/Sink":
            continue
        if source.state not in {"running", "idle"}:
            continue
        apps_by_sink.setdefault(sink_id, set()).add(source.app or source.name)

    microphones = [
        Device(node.name, node.description)
        for node in nodes.values()
        if node.media == "Audio/Source" and _usable(node.name)
    ]
    sinks = [
        Device(node.name, node.description, tuple(sorted(apps_by_sink.get(node_id, set()))))
        for node_id, node in nodes.items()
        if node.media == "Audio/Sink" and _usable(node.name)
    ]
    microphones.sort(key=lambda device: device.label.casefold())
    sinks.sort(key=lambda device: device.label.casefold())
    return AudioGraph(tuple(microphones), tuple(sinks), default_microphone, default_sink)


def select_microphone(
    graph: AudioGraph,
    requested: str | None,
    *,
    choose: bool,
    interactive: bool,
    input_func: Callable[[str], str] = input,
) -> Device:
    """Pick the microphone, prompting only when the choice is ambiguous."""
    devices = graph.microphones
    if not devices:
        raise RuntimeError("No microphone found.")
    if requested:
        return match_device(devices, requested)
    if len(devices) == 1 and not (choose and interactive):
        return devices[0]
    default = _by_name(devices, graph.default_microphone)
    if choose and interactive:
        return _prompt(devices, "microphone", default.name if default else None, input_func)
    if default is not None and not choose:
        return default
    if interactive:
        return _prompt(devices, "microphone", default.name if default else None, input_func)
    if default is not None:
        return default
    raise RuntimeError("Several microphones are connected. Re-run with --mic or --choose.")


def select_meeting(
    graph: AudioGraph,
    requested: str | None,
    *,
    choose: bool,
    interactive: bool,
    input_func: Callable[[str], str] = input,
) -> Device:
    """Pick the playback device whose monitor is the meeting audio."""
    if not graph.sinks:
        raise RuntimeError("No playback device found.")
    if requested:
        return match_device(graph.sinks, requested)

    playing = tuple(sink for sink in graph.sinks if sink.apps)
    if len(playing) > 1:
        if interactive:
            return _prompt(playing, "meeting output", graph.default_sink, input_func)
        chosen = _by_name(playing, graph.default_sink) or playing[0]
        print(f"Several outputs are playing; using {chosen.label}.")
        return chosen
    if len(playing) == 1 and not (choose and interactive):
        return playing[0]

    candidates = playing or graph.sinks
    if len(candidates) == 1 and not (choose and interactive):
        return candidates[0]
    default = _by_name(candidates, graph.default_sink)
    if choose and interactive:
        return _prompt(candidates, "meeting output", default.name if default else None, input_func)
    if default is not None:
        return default
    if interactive:
        return _prompt(candidates, "meeting output", None, input_func)
    raise RuntimeError("Several playback devices found. Pass --meeting to choose one.")


def match_device(devices: Sequence[Device], query: str) -> Device:
    """Match a node name or description. A unique substring is enough."""
    folded = query.casefold()
    exact = [
        device
        for device in devices
        if device.name == query
        or device.description.casefold() == folded
        or device.label.casefold() == folded
    ]
    if len(exact) == 1:
        return exact[0]
    pool = (
        list(exact)
        if len(exact) > 1
        else [
            device
            for device in devices
            if folded in device.name.casefold()
            or folded in device.description.casefold()
            or folded in device.label.casefold()
        ]
    )
    if len(pool) == 1:
        return pool[0]
    rendered = "\n".join(f"  {device.label}  ({device.name})" for device in (pool or devices))
    if not pool:
        raise DeviceMatchError(f"No device matches {query!r}.\n{rendered}")
    raise DeviceMatchError(f"{query!r} matches more than one device.\n{rendered}")


def _prompt(
    devices: Sequence[Device],
    kind: str,
    default_name: str | None,
    input_func: Callable[[str], str],
) -> Device:
    default_index = 0
    print(f"Select {kind}:")
    for index, device in enumerate(devices):
        mark = " (default)" if device.name == default_name else ""
        print(f"  {index + 1}) {device.label}{mark}")
        if device.name == default_name:
            default_index = index
    raw = input_func(f"{kind} [{default_index + 1}]: ").strip()
    if not raw:
        return devices[default_index]
    if raw.isdigit():
        choice = int(raw)
        if 1 <= choice <= len(devices):
            return devices[choice - 1]
    raise DeviceMatchError(f"Choose a number from 1 to {len(devices)}.")


def _by_name(devices: Sequence[Device], name: str | None) -> Device | None:
    if name is None:
        return None
    return next((device for device in devices if device.name == name), None)


def _usable(name: str) -> bool:
    lowered = name.casefold()
    return "monitor" not in lowered and "loopback" not in lowered and bool(name)


@dataclass(frozen=True)
class _Node:
    name: str
    description: str
    media: str
    state: str
    app: str


def _node(info: object) -> _Node | None:
    if not isinstance(info, dict):
        return None
    props = info.get("props")
    if not isinstance(props, dict):
        return None
    name = props.get("node.name")
    if not isinstance(name, str) or not name:
        return None
    description = props.get("node.description")
    media = props.get("media.class")
    app = props.get("application.name")
    state = info.get("state")
    return _Node(
        name,
        description if isinstance(description, str) and description else name,
        media if isinstance(media, str) else "",
        state if isinstance(state, str) else "",
        app if isinstance(app, str) else "",
    )


def _defaults(metadata: object) -> tuple[str | None, str | None]:
    microphone: str | None = None
    sink: str | None = None
    if not isinstance(metadata, list):
        return None, None
    for entry in metadata:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        name = _meta_name(entry.get("value"))
        if key == "default.audio.source":
            microphone = name
        elif key == "default.audio.sink":
            sink = name
    return microphone, sink


def _meta_name(value: object) -> str | None:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(parsed, dict):
        name = parsed.get("name")
        if isinstance(name, str):
            return name
    return None
