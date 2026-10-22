# live-transcript

Record a meeting on your own machine and keep a transcript, the audio, and speaker labels.

A session is one language: English or French. Choose it when you start. The microphone and, for a video call, the meeting audio are saved as separate recordings so the people on your side of the room are easier to tell apart from the people on the call.

## What gets written

Sessions go to `~/Captures/live-transcript/` unless you pass `--captures` or set `LIVE_TRANSCRIPT_DIR`.

```text
~/Captures/live-transcript/2026-10-22T103600-visio-weekly/
  session.json          metadata: mode, language, devices, model
  mic.wav               your microphone
  meeting.wav           what the computer played (visio only)
  transcript.jsonl      canonical transcript, one utterance per line
  transcript.md         the same transcript, for reading
  transcript.srt        subtitles; the speaker name is in the cue
  transcript.live.jsonl the labels from the live pass, kept after a refine
```

`transcript.jsonl` is the file to keep if you want to reprocess. Each line has the channel (`mic` or `meeting`), the speaker, start and end in seconds, the text, and the session language:

```json
{"id": 1, "channel": "mic", "speaker": "Me", "speaker_live": "Me", "start": 1.2, "end": 3.4, "text": "Can you hear me?", "language": "en"}
```

Markdown is the copy you read. SRT is there for players. SRT has no speaker field, so the name is written at the start of the cue. JSONL is the better record: it keeps the channel, both speaker labels, and exact times, and a line can be appended while the meeting is still going.

## Modes

**Visio** records two streams:

- the microphone, labeled with your name (`Me` unless you pass `--me`)
- the meeting, which is the monitor of the playback device the call is using

Those two streams are the first speaker split. Diarization then separates the people on the meeting recording into Remote 1, Remote 2, and so on. Use headphones so your microphone does not also hear the call.

**Live** records the microphone only. Everyone is in the room, so diarization splits that one recording into Speaker 1, Speaker 2, and so on.

While the meeting runs, each phrase is transcribed and given a live speaker label. When you quit, the saved audio is diarized again and the transcript speakers are updated. The first labels stay in `transcript.live.jsonl`. Pass `--no-refine` to skip that pass. Press Ctrl+C a second time to stop without waiting for it.

`lt reprocess` runs Whisper over the saved WAV files again and diarizes them. Use that when a chunk was dropped, or when you want a cleaner transcript than the live one. `--labels-only` keeps the text and only recomputes speakers.

## Install

Linux with PipeWire (`pw-dump` and `pw-record`). Python 3.12 or newer. [uv](https://docs.astral.sh/uv/) installs the project.

```bash
cd live-transcript
uv sync
```

The first capture downloads a Whisper model and two small diarization models (pyannote segmentation and a WeSpeaker embedding). After that, transcription and diarization stay on this machine. An NVIDIA GPU is used for Whisper when CUDA is available. `--cpu` forces the CPU. Diarization runs on the CPU.

## Usage

```bash
uv run lt devices
uv run lt visio --language en
uv run lt visio --language fr --title "Point hebdo"
uv run lt live --language en --me Host
uv run lt reprocess latest
uv run lt reprocess latest --labels-only
```

`--language` is `en` or `fr`. A new capture defaults to English. Reprocess keeps the language and model stored with the session unless you pass them again.

Other useful options:

| Option | Meaning |
| --- | --- |
| `--model tiny\|base\|small\|medium\|turbo\|large` | Whisper model. `turbo` is `large-v3-turbo` and is the default. |
| `--mic` / `--meeting` | PipeWire node name or a unique part of its description. |
| `--choose` | Ask which devices to use. |
| `--speakers N` | How many speakers to find in the diarized recording. `0` detects them. |
| `--no-keyboard` | Do not read `q`, `p`, and `h`. Ctrl+C still stops. |
| `--cpu` | Transcribe on the CPU. |

During a capture: `q` quits, `p` pauses transcription (the audio is still recorded), `h` shows the keys.

`lt devices` lists microphones and playback devices. The default microphone is used when several are connected. If one playback device is already playing, that one is the meeting audio. If several are playing, you get a menu. If nothing is playing yet, the default output is used, so you can start the capture before joining the call.

## How speakers are assigned

1. Visio already knows your microphone from the meeting audio.
2. During the meeting, short voice embeddings are compared with people already heard on that recording. A close match reuses the label. A new voice becomes the next Remote or Speaker number.
3. At the end, and again if you reprocess, an offline diarization pass (speech segmentation, embeddings, clustering) assigns speakers from the whole recording. That pass is the one to trust when two people sound similar live.

Overlapping speech is still hard. A phrase is given to the speaker who occupies most of its time.

## Privacy

Audio and transcripts are written only in the capture folder. Whisper and the diarization models run locally. The first run needs a network connection to download those models. There is no account and no telemetry.

## Development

```bash
uv run ruff check
uv run ruff format
uv run ty check
uv run pytest
uv run pre-commit install
```

## License

MIT. See [LICENSE](LICENSE).

Copyright (C) 2026 lokiledev.
