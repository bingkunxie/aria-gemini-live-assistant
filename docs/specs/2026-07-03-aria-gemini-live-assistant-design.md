# Aria + Gemini Live Cooking Assistant — Design

**Date:** 2026-07-03
**Status:** Approved by user
**Context:** Prototype for a research project on human-AI reliance in physical
tasks.
Users wearing Project Aria glasses perform sequential decision-making tasks with
AI assistance; the study will later infer reliance (over/appropriate/under)
from physiological signals, mainly eye gaze. This prototype establishes the
live AI-assistance loop and produces a recorded demo of a lemonade-making task
(expected duration 5–6 minutes).

## Goal

A single Python app that:
1. Streams the Aria Gen1 RGB camera to Gemini so it sees what the wearer sees.
2. Holds a real-time voice conversation with the wearer (AirPods in/out).
3. Shows a live transcript of the conversation in a browser window.
4. In record mode, saves a shareable demo (video + audio + transcript) and a
   timestamped event log suitable for later accept/reject reliance coding.

## Decisions (from brainstorming)

- **Approach:** single Python process (approved over OBS/AI-Studio no-code and
  browser-app alternatives) — needed for synced recording and research logging.
- **Gemini access:** Gemini API with a free Google AI Studio API key (user will
  create one, guided). Not Vertex.
- **Model:** `gemini-3.1-flash-live-preview`; fallback
  `gemini-2.5-flash-native-audio-preview-12-2025`.
- **Streaming interface:** WiFi preferred for mobility, USB fallback
  (eduroam-style networks may block device-to-device traffic; use home
  WiFi/hotspot).
- **AI behavior:** proactive hands-on coach issuing discrete, rejectable
  suggestions; adapts when the user deviates; short utterances; English.
- **Demo output:** one MP4 with RGB view + rendered transcript panel and both
  voices on the audio track, plus separate raw artifacts.

## Architecture

```
Aria glasses ──WiFi/USB──▶ Aria SDK observer ──▶ latest RGB frame ──┐
                                                                    │ JPEG @1fps
AirPods mic ──▶ sounddevice ──▶ 16kHz PCM16 ────────────────────────┤
                                                                    ▼
                                                     Gemini Live API session
                                                                    │
              ◀── 24kHz PCM16 voice ◀───────────────────────────────┤
AirPods out ──┘                                                     │
                                                                    │ transcription events
Browser transcript window ◀── local WebSocket ◀── transcript hub ◀──┘
                                                        │
Record mode: composite MP4 writer ◀─────────────────────┴──▶ events.json, WAVs
```

Verified API facts (docs, 2026-07-03): audio in 16 kHz PCM16 mono, audio out
24 kHz PCM16; video as JPEG ≤ 1 fps via `send_realtime_input(video=Blob(...))`;
transcription enabled with `input_audio_transcription` /
`output_audio_transcription`; Python SDK `google-genai`
(`client.aio.live.connect(model=..., config=...)`).

## Components

Project root: `~/ProjectAria/gemini_assistant/` (own git repo). Runs in the
existing venv `~/projectaria_tools_python_env` (Python 3.11; already has
`projectaria_client_sdk` 2.4.0, `opencv-python`, `numpy`). New deps:
`google-genai`, `sounddevice`, `aiohttp` (transcript server), `python-dotenv`.

| Unit | Responsibility | Interface |
|---|---|---|
| `aria_source.py` | Connect to glasses over USB or WiFi, subscribe to RGB, expose latest frame | `AriaSource(interface, device_ip).start() / .latest_frame() -> np.ndarray \| None / .stop()`; reports stream failures via callback |
| `audio_io.py` | Full-duplex AirPods audio via `sounddevice`: capture 16 kHz mono PCM16 chunks; play 24 kHz PCM16; flush playback on interruption | `AudioIO.read_chunk()`, `.play(bytes)`, `.flush_playback()`, device selection by name substring with default-device fallback |
| `gemini_session.py` | Live API session: config (system prompt, transcription, compression, resumption), async send loops (audio, frames), receive loop (audio, transcripts, GoAway), reconnect with resumption handle | `GeminiSession(config).run(audio_in, frame_source, on_audio, on_event)` |
| `transcript_hub.py` | Fan out transcript/status events to: browser page (local WebSocket, `localhost` only), `events.json` (append, timestamped), in-memory state for the recorder's panel | `hub.emit(event)`, serves `http://localhost:8899` |
| `recorder.py` | Record mode only: write composite frames (RGB + transcript panel, ~1080p, 10 fps) to MP4; write `mic.wav` and `gemini.wav`; on stop, mix + mux via `ffmpeg` (checked at startup in record mode; installed via Homebrew during setup if missing) | `Recorder(session_dir).add_frame(...) / .add_mic(...) / .add_model_audio(...) / .finalize()` |
| `app.py` | CLI entry: `--interface {usb,wifi}`, `--device-ip`, `--record`, `--audio-test` (voice-only bring-up stage, no glasses) | `python -m assistant ...` |

Event log schema (`events.json`, one JSON object per line):
`{t_wall, t_session, type: user_transcript | model_transcript | frame_sent |
session_event | error, text?, detail?}` — model transcripts are the
"suggestions"; user transcripts + the video are the accept/reject ground truth.

## AI behavior (system prompt outline)

- Role: hands-on cooking coach for a lemonade task; sees through the wearer's
  camera; speaks English, short conversational utterances.
- One discrete suggestion at a time, phrased explicitly ("I suggest …") so
  suggestions are identifiable in the transcript for coding.
- Never insists: if the user does something else or declines, acknowledge and
  adapt; no repeated nagging.
- Loose task plan in prompt (wash/cut lemon, juice, water, sugar, stir, taste),
  not a rigid script; react to what is actually visible.

## Session robustness / error handling

- **2-min audio+video session cap** → enable context window compression
  (`ContextWindowCompressionConfig(sliding_window=SlidingWindow())`).
- **~10-min connection lifetime** → enable session resumption; store the latest
  resumption handle; on `GoAway` (server sends `timeLeft`) or socket drop,
  reconnect with the handle and continue seamlessly. Resumption tokens valid
  2 h. (A 5–6 min demo may never trigger this, but it is implemented anyway.)
- **Aria stream failure** → status event to transcript window; audio continues;
  frames resume when the SDK reconnects. In record mode the video shows a
  "stream lost" panel rather than freezing silently.
- **Interruptions (barge-in)** → on Live API `interrupted` signal, flush queued
  playback so Gemini stops talking promptly.
- **AirPods disconnect** → clean shutdown with an explicit error message; in
  record mode, artifacts written so far are finalized, not lost.
- **Recording finalization** → WAVs and MP4 are written incrementally;
  `finalize()` runs even on Ctrl-C (try/finally), so a crash loses at most the
  final mux, which can be re-run.

## Bring-up plan (each stage gated on the previous)

1. `--audio-test`: AirPods ↔ Gemini voice + transcript window, no glasses.
2. USB vision: `--interface usb`, verify "what am I holding?" works.
3. WiFi vision: same test unplugged; if the network blocks it, fall back to USB.
4. Rehearsal: live mode, short mock task.
5. `--record`: real lemonade demo.

Component-level checks precede integration: audio loopback test, Aria frame
grab test, transcript page render test. No formal unit-test suite — this is a
research prototype; verification is behavioral per stage.

## Record-mode outputs

`~/ProjectAria/gemini_assistant/sessions/<YYYYMMDD-HHMMSS>/`:
- `demo.mp4` — composite video + mixed conversation audio
- `mic.wav`, `gemini.wav` — separate raw tracks
- `transcript.txt`, `events.json`

## Known limitations

- Aria Gen1 cannot record VRS on-device while streaming → no MPS eye gaze
  during live-AI sessions. Flagged as a study-design question for the mentor
  (options later: Gen2 hardware, alternating record/stream phases, or gaze from
  streamed EyeTrack images).
- Free-tier Live API rate/usage limits: fine for bring-up + one demo; heavy
  iteration may hit daily caps.
- Preview-model churn: model IDs may change; config keeps the model name in one
  place (`.env`).
- API key stored in `.env` (gitignored), never committed.

## Out of scope (this prototype)

- Gaze-based reliance inference, multi-task/user support, any study
  instrumentation beyond the event log, Gen2 support, non-English.
