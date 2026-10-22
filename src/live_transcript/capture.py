# Copyright (C) 2026 lokiledev
# SPDX-License-Identifier: MIT
"""Record a meeting and write the transcript as speech arrives."""

import queue
import signal
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import IO, BinaryIO

from live_transcript.audio import (
    load_graph,
    record_command,
    require_pipewire,
    select_meeting,
    select_microphone,
)
from live_transcript.const import SAMPLE_RATE
from live_transcript.cuda import cuda_available
from live_transcript.diarize import (
    EmbeddingExtractor,
    SpeakerBank,
    diarize_channel,
    load_embedding_extractor,
)
from live_transcript.keyboard import Keyboard
from live_transcript.session import (
    ChannelMeta,
    SessionMeta,
    allocate_session_dir,
    default_captures_dir,
    latest_session,
    load_session,
    save_session,
)
from live_transcript.speakers import (
    channel_is_diarized,
    channel_naming,
    fallback_speaker,
    label_recording,
)
from live_transcript.transcribe import WhisperEngine, resolve_model
from live_transcript.transcript import (
    TranscriptStore,
    Utterance,
    group_words,
    read_utterances,
    word_count,
)
from live_transcript.vad import EnergyVad
from live_transcript.wavfile import StreamingWavWriter, read_wav


@dataclass
class RunConfig:
    """Options for one live capture."""

    mode: str
    captures_dir: Path
    title: str | None
    language: str
    model: str
    local_name: str
    cpu: bool
    keyboard: bool
    refine: bool
    choose: bool
    speakers: int | None
    same_speaker: float
    cluster_threshold: float
    diarize_mic: bool
    mic: str | None
    meeting: str | None


@dataclass
class _Speech:
    channel: str
    role: str
    pcm: bytes
    start: float
    end: float


@dataclass
class _Runtime:
    stop: threading.Event
    pause: threading.Event
    t0: float
    lock: threading.Lock
    procs: list[subprocess.Popen[bytes]]
    dropped: int
    interrupts: int
    meta: SessionMeta
    directory: Path
    error: str | None = None


def run_session(config: RunConfig) -> None:
    """Capture audio, transcribe it, and diarize speakers."""
    require_pipewire()
    if config.language not in {"en", "fr"}:
        raise RuntimeError("Language must be en or fr.")
    graph = load_graph()
    interactive = sys.stdin.isatty()
    microphone = select_microphone(
        graph,
        config.mic,
        choose=config.choose,
        interactive=interactive,
    )
    meeting = None
    if config.mode == "visio":
        meeting = select_meeting(
            graph,
            config.meeting,
            choose=config.choose,
            interactive=interactive,
        )
    now = datetime.now().astimezone()
    directory = allocate_session_dir(config.captures_dir, config.mode, config.title, now)
    directory.mkdir(parents=True, exist_ok=True)
    channels: list[ChannelMeta] = [
        {
            "name": "mic",
            "role": "local",
            "device": microphone.name,
            "description": microphone.label,
            "file": "mic.wav",
            "start_offset": 0.0,
        }
    ]
    if meeting is not None:
        channels.append(
            {
                "name": "meeting",
                "role": "remote",
                "device": meeting.name,
                "description": meeting.label,
                "file": "meeting.wav",
                "start_offset": 0.0,
            }
        )
    meta: SessionMeta = {
        "version": 1,
        "id": directory.name,
        "title": config.title or "",
        "mode": config.mode,
        "started": now.isoformat(timespec="seconds"),
        "ended": None,
        "language": config.language,
        "model": resolve_model(config.model),
        "sample_rate": SAMPLE_RATE,
        "local_name": config.local_name,
        "channels": channels,
        "stats": {"utterances": 0, "words": 0, "dropped": 0},
    }
    save_session(directory, meta)
    runtime = _Runtime(
        stop=threading.Event(),
        pause=threading.Event(),
        t0=time.monotonic(),
        lock=threading.Lock(),
        procs=[],
        dropped=0,
        interrupts=0,
        meta=meta,
        directory=directory,
    )
    speeches: queue.Queue[_Speech] = queue.Queue(maxsize=500)
    threads = [
        threading.Thread(
            target=_capture_channel,
            args=(runtime, channel, speeches),
            name=f"capture-{channel['name']}",
        )
        for channel in channels
    ]
    _print_banner(config, meta, directory)
    for thread in threads:
        thread.start()
    print("Recording. Loading the transcription model...", flush=True)

    previous_sigint = signal.getsignal(signal.SIGINT)

    def _on_sigint(signum: int, frame: types.FrameType | None) -> None:
        runtime.interrupts += 1
        runtime.stop.set()
        if runtime.interrupts >= 2:
            raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _on_sigint)
    store = TranscriptStore(directory, meta)
    keyboard = Keyboard()
    try:
        engine = WhisperEngine(
            meta["model"],
            config.language,
            use_gpu=not config.cpu and cuda_available(),
        )
        embedder = _try_embedder()
        banks = _speaker_banks(config)
        if config.keyboard and keyboard.start():
            print("Listening. q quits, p pauses, h shows help.", flush=True)
        else:
            print("Listening. Ctrl+C stops and saves.", flush=True)
        while not runtime.stop.is_set():
            _handle_key(keyboard, runtime)
            try:
                speech = speeches.get(timeout=0.1)
            except queue.Empty:
                continue
            text = engine.transcribe_pcm(speech.pcm)
            if not text:
                continue
            speaker = _live_speaker(speech, config, banks, embedder)
            utterance = store.add(
                channel=speech.channel,
                speaker=speaker,
                start=speech.start,
                end=speech.end,
                text=text,
                language=config.language,
            )
            print(
                f"[{_clock(utterance.start)}] {utterance.speaker:<12} {utterance.text}", flush=True
            )
    except KeyboardInterrupt:
        runtime.stop.set()
        runtime.interrupts += 1
    finally:
        runtime.stop.set()
        _terminate(runtime)
        for thread in threads:
            thread.join(timeout=3)
        keyboard.stop()
        signal.signal(signal.SIGINT, previous_sigint)
        if config.refine and runtime.interrupts < 2 and store.utterances:
            _refine(store, config)
        _finish(runtime, store)


def reprocess_session(
    directory: Path,
    *,
    language: str | None,
    model: str | None,
    cpu: bool,
    speakers: int | None,
    cluster_threshold: float,
    labels_only: bool,
    diarize_mic: bool,
) -> None:
    """Rebuild a transcript from the saved recordings."""
    meta = load_session(directory)
    if language is not None:
        if language not in {"en", "fr"}:
            raise RuntimeError("Language must be en or fr.")
        meta["language"] = language
    if model is not None:
        meta["model"] = resolve_model(model)
    if labels_only:
        updated = label_recording(
            meta,
            read_utterances(directory / "transcript.jsonl"),
            _diarize_getter(directory, meta, speakers, cluster_threshold, diarize_mic),
            diarize_mic=diarize_mic,
        )
    else:
        engine = WhisperEngine(
            meta["model"],
            meta["language"],
            use_gpu=not cpu and cuda_available(),
        )
        updated = _transcribe_recordings(
            directory,
            meta,
            engine,
            speakers=speakers,
            cluster_threshold=cluster_threshold,
            diarize_mic=diarize_mic,
        )
    store = TranscriptStore(directory, meta)
    try:
        store.rewrite(updated)
        meta["ended"] = meta["ended"] or datetime.now().astimezone().isoformat(timespec="seconds")
        meta["stats"] = {
            "utterances": len(store.utterances),
            "words": word_count(store.utterances),
            "dropped": meta["stats"]["dropped"],
        }
        save_session(directory, meta)
    finally:
        store.close()
    print(f"Wrote {directory / 'transcript.md'}", flush=True)


def resolve_session_path(argument: str, captures: Path | None) -> Path:
    """Find a session directory from a path, a folder name, or `latest`."""
    root = captures or default_captures_dir()
    if argument == "latest":
        return latest_session(root)
    path = Path(argument).expanduser()
    if (path / "session.json").is_file():
        return path
    named = root / argument
    if (named / "session.json").is_file():
        return named
    raise RuntimeError(f"No session at {argument}.")


def _capture_channel(
    runtime: _Runtime,
    channel: ChannelMeta,
    speeches: queue.Queue[_Speech],
) -> None:
    vad = EnergyVad()
    wav_path = runtime.directory / channel["file"]
    log_path = runtime.directory / f"{channel['name']}.log"
    log_handle = log_path.open("wb")
    wav = StreamingWavWriter(wav_path)
    proc: subprocess.Popen[bytes] | None = None
    offset: float | None = None
    last_frame_at = runtime.t0
    failures = 0
    next_checkpoint = time.monotonic() + 2

    def write_frame(frame: bytes) -> None:
        wav.write(frame)
        speech = vad.push(frame)
        if speech is None or runtime.pause.is_set() or offset is None:
            return
        start = offset + speech.start_sample / SAMPLE_RATE
        end = offset + speech.end_sample / SAMPLE_RATE
        item = _Speech(channel["name"], channel["role"], speech.pcm, start, end)
        try:
            speeches.put(item, timeout=1)
        except queue.Full:
            with runtime.lock:
                runtime.dropped += 1

    try:
        while not runtime.stop.is_set():
            if proc is None or proc.poll() is not None:
                if proc is not None:
                    _stop_proc(proc)
                    _unregister(runtime, proc)
                    proc = None
                    if offset is not None:
                        missed = time.monotonic() - last_frame_at
                        silence = bytes(vad.frame_bytes)
                        for _ in range(max(0, int(missed * SAMPLE_RATE) // vad.frame_samples)):
                            write_frame(silence)
                if runtime.stop.is_set():
                    break
                if failures > 5:
                    runtime.error = f"Lost the {channel['name']} recording."
                    runtime.stop.set()
                    break
                proc = _start_proc(channel, log_handle)
                _register(runtime, proc)
            stdout = proc.stdout
            if stdout is None:
                runtime.error = "pw-record produced no audio stream."
                runtime.stop.set()
                break
            frame = _read_exact(stdout, vad.frame_bytes)
            if len(frame) < vad.frame_bytes:
                failures += 1
                if failures == 1:
                    print(f"\n{channel['description']} stopped. Reconnecting...", flush=True)
                _stop_proc(proc)
                _unregister(runtime, proc)
                proc = None
                time.sleep(0.5)
                continue
            failures = 0
            now = time.monotonic()
            if offset is None:
                offset = now - runtime.t0
                _set_offset(runtime, channel["name"], offset)
            last_frame_at = now
            write_frame(frame)
            if now >= next_checkpoint:
                wav.checkpoint()
                next_checkpoint = now + 2
        trailing = vad.flush()
        if trailing is not None and offset is not None and not runtime.pause.is_set():
            start = offset + trailing.start_sample / SAMPLE_RATE
            end = offset + trailing.end_sample / SAMPLE_RATE
            try:
                speeches.put(
                    _Speech(channel["name"], channel["role"], trailing.pcm, start, end),
                    timeout=1,
                )
            except queue.Full:
                with runtime.lock:
                    runtime.dropped += 1
    except Exception as exc:
        runtime.error = f"{channel['name']} capture failed: {exc}"
        runtime.stop.set()
    finally:
        if proc is not None:
            _stop_proc(proc)
            _unregister(runtime, proc)
        wav.close()
        log_handle.close()
        if log_path.is_file() and log_path.stat().st_size == 0:
            log_path.unlink()


def _start_proc(channel: ChannelMeta, log_handle: BinaryIO) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        record_command(channel["device"], monitor_sink=channel["role"] == "remote"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=log_handle,
    )


def _read_exact(stream: IO[bytes], size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        piece = stream.read(remaining)
        if not piece:
            break
        chunks.append(piece)
        remaining -= len(piece)
    return b"".join(chunks)


def _set_offset(runtime: _Runtime, name: str, offset: float) -> None:
    with runtime.lock:
        for channel in runtime.meta["channels"]:
            if channel["name"] == name:
                channel["start_offset"] = round(offset, 3)
        save_session(runtime.directory, runtime.meta)


def _live_speaker(
    speech: _Speech,
    config: RunConfig,
    banks: dict[str, SpeakerBank],
    embedder: EmbeddingExtractor | None,
) -> str:
    if config.mode == "visio" and speech.role == "local":
        return config.local_name
    bank = banks.get(speech.channel)
    generic = "Remote" if speech.role == "remote" else "Speaker"
    if bank is None or embedder is None:
        return generic
    try:
        embedding = embedder.embed(speech.pcm)
    except Exception as exc:
        print(f"\nSpeaker embedding failed: {exc}", flush=True)
        return generic
    return bank.assign(
        embedding,
        allow_new=speech.end - speech.start >= 1.0,
        start=speech.start,
        end=speech.end,
    )


def _speaker_banks(config: RunConfig) -> dict[str, SpeakerBank]:
    if config.mode == "live":
        return {"mic": SpeakerBank("Speaker", threshold=config.same_speaker)}
    return {"meeting": SpeakerBank("Remote", threshold=config.same_speaker)}


def _try_embedder() -> EmbeddingExtractor | None:
    try:
        return load_embedding_extractor()
    except Exception as exc:
        print(
            f"Speaker diarization models are unavailable ({exc}). "
            "Microphone and meeting audio will still be labeled separately.",
            flush=True,
        )
        return None


def _refine(store: TranscriptStore, config: RunConfig) -> None:
    print("Refining speaker labels from the recording...", flush=True)
    try:
        updated = label_recording(
            store.meta,
            store.utterances,
            _diarize_getter(
                store.directory,
                store.meta,
                config.speakers,
                config.cluster_threshold,
                config.diarize_mic,
            ),
            diarize_mic=config.diarize_mic,
        )
    except Exception as exc:
        print(f"Could not refine speakers ({exc}). The live transcript was kept.", flush=True)
        return
    store.rewrite(updated)


def _transcribe_recordings(
    directory: Path,
    meta: SessionMeta,
    engine: WhisperEngine,
    *,
    speakers: int | None,
    cluster_threshold: float,
    diarize_mic: bool,
) -> list[Utterance]:
    utterances: list[Utterance] = []
    next_id = 1
    for channel in meta["channels"]:
        path = directory / channel["file"]
        if not path.is_file():
            raise RuntimeError(f"Missing recording {path}.")
        print(f"Transcribing {channel['file']}...", flush=True)
        audio, rate = read_wav(path)
        if rate != SAMPLE_RATE:
            raise RuntimeError(f"{path} is {rate} Hz, expected {SAMPLE_RATE}.")
        words = engine.transcribe_timeline(audio)
        if channel_is_diarized(meta, channel, diarize_mic=diarize_mic):
            print(f"Diarizing {channel['file']}...", flush=True)
            prefix, single_name = channel_naming(meta, channel)
            turns = diarize_channel(
                path,
                num_speakers=-1 if speakers is None else speakers,
                threshold=cluster_threshold,
                prefix=prefix,
                single_name=single_name,
            )
        else:
            turns = []
        offset = float(channel["start_offset"])
        for speaker, start, end, text in group_words(
            words,
            turns,
            default_speaker=fallback_speaker(meta, channel),
        ):
            utterances.append(
                Utterance(
                    id=next_id,
                    channel=channel["name"],
                    speaker=speaker,
                    speaker_live=speaker,
                    start=start + offset,
                    end=end + offset,
                    text=text,
                    language=meta["language"],
                )
            )
            next_id += 1
    ordered = sorted(utterances, key=lambda item: (item.start, item.channel, item.id))
    return [replace(item, id=index) for index, item in enumerate(ordered, start=1)]


def _diarize_getter(
    directory: Path,
    meta: SessionMeta,
    speakers: int | None,
    threshold: float,
    diarize_mic: bool,
) -> Callable[[str], list[tuple[float, float, str]]]:
    cache: dict[str, list[tuple[float, float, str]]] = {}

    def get(name: str) -> list[tuple[float, float, str]]:
        if name in cache:
            return cache[name]
        channel = next(item for item in meta["channels"] if item["name"] == name)
        if not channel_is_diarized(meta, channel, diarize_mic=diarize_mic):
            cache[name] = []
            return []
        print(f"Diarizing {channel['file']}...", flush=True)
        prefix, single_name = channel_naming(meta, channel)
        cache[name] = diarize_channel(
            directory / channel["file"],
            num_speakers=-1 if speakers is None else speakers,
            threshold=threshold,
            prefix=prefix,
            single_name=single_name,
        )
        return cache[name]

    return get


def _finish(runtime: _Runtime, store: TranscriptStore) -> None:
    meta = runtime.meta
    meta["ended"] = datetime.now().astimezone().isoformat(timespec="seconds")
    meta["stats"] = {
        "utterances": len(store.utterances),
        "words": word_count(store.utterances),
        "dropped": runtime.dropped,
    }
    save_session(runtime.directory, meta)
    store.close()
    if runtime.error:
        print(runtime.error, file=sys.stderr)
    print()
    print(f"Saved {runtime.directory}")
    print(f"  transcript  {runtime.directory / 'transcript.md'}")
    print(f"  record      {runtime.directory / 'transcript.jsonl'}")
    for channel in meta["channels"]:
        print(f"  audio       {runtime.directory / channel['file']}")
    if runtime.dropped:
        print(
            f"{runtime.dropped} speech chunks were not transcribed in time. "
            "The audio was kept; reprocess this session to fill them in."
        )


def _print_banner(config: RunConfig, meta: SessionMeta, directory: Path) -> None:
    language = "English" if config.language == "en" else "French"
    print()
    print(f"Recording to {directory}")
    print(f"  Mode       {config.mode}")
    for channel in meta["channels"]:
        label = "Microphone" if channel["role"] == "local" else "Meeting"
        print(f"  {label:<10} {channel['description']}")
    print(f"  Language   {language}")
    print(f"  Model      {meta['model']}")
    print()
    if config.mode == "visio":
        print("Use headphones so the microphone does not hear the meeting.")
    print()


def _handle_key(keyboard: Keyboard, runtime: _Runtime) -> None:
    key = keyboard.poll(0.01) if keyboard.active else None
    if key == "q":
        print("\nStopping...", flush=True)
        runtime.stop.set()
    elif key == "p":
        if runtime.pause.is_set():
            runtime.pause.clear()
            print("\nResumed.", flush=True)
        else:
            runtime.pause.set()
            print("\nPaused. Audio is still being recorded.", flush=True)
    elif key == "h":
        print("\nq quit    p pause or resume    h help\n", flush=True)


def _register(runtime: _Runtime, proc: subprocess.Popen[bytes]) -> None:
    with runtime.lock:
        runtime.procs.append(proc)


def _unregister(runtime: _Runtime, proc: subprocess.Popen[bytes]) -> None:
    with runtime.lock:
        if proc in runtime.procs:
            runtime.procs.remove(proc)


def _terminate(runtime: _Runtime) -> None:
    with runtime.lock:
        procs = list(runtime.procs)
    for proc in procs:
        _stop_proc(proc)


def _stop_proc(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)


def _clock(seconds: float) -> str:
    total = int(max(0.0, seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
