# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Command-line entry point."""

import argparse
import sys
from pathlib import Path

from live_transcript.audio import (
    DeviceMatchError,
    load_graph,
    require_pipewire,
)
from live_transcript.capture import RunConfig, reprocess_session, resolve_session_path, run_session
from live_transcript.session import default_captures_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="lt",
        description="Capture a live transcript in English or French.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  lt devices
  lt visio --language en
  lt visio --language fr --title "Point hebdo"
  lt live --language en
  lt reprocess latest

Each session is one language. Use --language en or --language fr.
Visio records the microphone and the meeting audio separately.
Live records the microphone only.
        """.strip(),
    )
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--captures",
        type=Path,
        help="Capture folder (default: ~/Captures/live-transcript)",
    )
    shared.add_argument("--title", help="Session title, used in the folder name and transcript")
    shared.add_argument("--me", default="Me", help="Name for the local microphone (default: Me)")
    shared.add_argument("--cpu", action="store_true", help="Force CPU transcription")
    shared.add_argument(
        "--speakers",
        type=int,
        default=0,
        help="Expected number of diarized speakers, 0 to detect (default: 0)",
    )
    shared.add_argument(
        "--same-speaker",
        type=float,
        default=0.65,
        help="Cosine similarity to treat a live voice as the same speaker (default: 0.65)",
    )
    shared.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.5,
        help="Offline clustering threshold; lower values split more speakers (default: 0.5)",
    )
    shared.add_argument(
        "--diarize-mic",
        action="store_true",
        help="Also split the visio microphone into speakers",
    )
    sub = parser.add_subparsers(dest="command")

    devices = sub.add_parser("devices", help="List microphones and playback devices")
    devices.add_argument("--captures", type=Path, help=argparse.SUPPRESS)

    capture_parent = argparse.ArgumentParser(add_help=False, parents=[shared])
    _add_language_and_model(capture_parent, language="en", model="turbo")
    capture_parent.add_argument("--mic", help="Microphone node name or description")
    capture_parent.add_argument("--choose", action="store_true", help="Ask which devices to use")
    capture_parent.add_argument("--no-keyboard", action="store_true", help="Ignore q, p, and h")
    capture_parent.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip the speaker-label pass when the session ends",
    )

    visio = sub.add_parser(
        "visio",
        parents=[capture_parent],
        help="Record the microphone and the meeting audio",
    )
    visio.add_argument("--meeting", help="Playback device to record as the meeting")

    sub.add_parser(
        "live",
        parents=[capture_parent],
        help="Record the microphone only, for a meeting in the room",
    )

    reprocess = sub.add_parser(
        "reprocess",
        parents=[shared],
        help="Transcribe and diarize a saved session again",
    )
    _add_language_and_model(reprocess, language=None, model=None)
    reprocess.add_argument("session", help="Session directory, its folder name, or 'latest'")
    reprocess.add_argument(
        "--labels-only",
        action="store_true",
        help="Keep the existing text and only recompute speaker labels",
    )
    return parser


def _add_language_and_model(
    parser: argparse.ArgumentParser,
    *,
    language: str | None,
    model: str | None,
) -> None:
    """Add language and model flags. Reprocess defaults to the saved session."""
    if language is None:
        language_help = "Replace the saved language: en or fr"
    else:
        language_help = "Language of this session: en or fr (default: en)"
    if model is None:
        model_help = "Replace the saved Whisper model"
    else:
        model_help = "Whisper model (default: turbo)"
    parser.add_argument("--language", choices=("en", "fr"), default=language, help=language_help)
    parser.add_argument(
        "--model",
        choices=("tiny", "base", "small", "medium", "turbo", "large"),
        default=model,
        help=model_help,
    )


def main(argv: list[str] | None = None) -> None:
    """Run a capture or list devices."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        raise SystemExit(2)
    try:
        if args.command == "devices":
            _show_devices()
            return
        if args.command == "reprocess":
            _check_speakers(args.speakers)
            reprocess_session(
                resolve_session_path(args.session, args.captures),
                language=args.language,
                model=args.model,
                cpu=args.cpu,
                speakers=None if args.speakers == 0 else args.speakers,
                cluster_threshold=args.cluster_threshold,
                labels_only=args.labels_only,
                diarize_mic=args.diarize_mic,
            )
            return
        _check_speakers(args.speakers)
        captures = args.captures.expanduser() if args.captures else default_captures_dir()
        run_session(
            RunConfig(
                mode=args.command,
                captures_dir=captures,
                title=args.title,
                language=args.language,
                model=args.model,
                local_name=args.me,
                cpu=args.cpu,
                keyboard=not args.no_keyboard,
                refine=not args.no_refine,
                choose=args.choose,
                speakers=None if args.speakers == 0 else args.speakers,
                same_speaker=args.same_speaker,
                cluster_threshold=args.cluster_threshold,
                diarize_mic=args.diarize_mic,
                mic=args.mic,
                meeting=getattr(args, "meeting", None),
            )
        )
    except DeviceMatchError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2) from exc
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc


def _show_devices() -> None:
    require_pipewire()
    graph = load_graph()
    print("Microphones")
    if not graph.microphones:
        print("  (none)")
    for device in graph.microphones:
        mark = " [default]" if device.name == graph.default_microphone else ""
        print(f"  {device.label}{mark}")
        print(f"    {device.name}")
    print()
    print("Outputs (meeting audio)")
    if not graph.sinks:
        print("  (none)")
    for device in graph.sinks:
        marks = []
        if device.name == graph.default_sink:
            marks.append("default")
        if device.apps:
            marks.append("playing")
        suffix = f" [{', '.join(marks)}]" if marks else ""
        print(f"  {device.label}{suffix}")
        print(f"    {device.name}")


def _check_speakers(value: int) -> None:
    if value < 0:
        raise RuntimeError("--speakers must be 0 or a positive number.")
