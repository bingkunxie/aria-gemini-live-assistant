# Aria × Gemini Live Assistant

Real-time AI assistant that sees through [Project Aria](https://www.projectaria.com/) Gen1
glasses and talks with the wearer. The Aria RGB camera streams to the
[Gemini Live API](https://ai.google.dev/gemini-api/docs/live) at 1 fps while audio flows
both ways through AirPods; a local web page shows the live camera view and conversation
transcript.

Built as a prototype for a research project on **human-AI reliance in physical tasks**:
the wearer performs a sequential decision-making task (e.g. making
lemonade) while the AI coach offers discrete, rejectable suggestions. The event log
captures every suggestion and response with timestamps for later accept/reject coding.

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
Browser (camera + transcript) ◀── localhost:8899 ◀── transcript hub ┘
```

| Module | Responsibility |
|---|---|
| `assistant/aria_source.py` | Glasses connection (USB/WiFi), latest RGB frame / JPEG |
| `assistant/audio_io.py` | AirPods duplex audio; flushable playback for barge-in |
| `assistant/gemini_session.py` | Live API session, transcription, compression, resumption, reconnect |
| `assistant/transcript_hub.py` | Event log + stitched transcript + web page (`localhost:8899`) |
| `assistant/prompts.py` | Coach persona (tasks) — `chat` mode runs promptless |
| `assistant/app.py` | CLI and wiring |

Long sessions survive the Live API's 2-minute audio+video cap (sliding-window context
compression) and ~10-minute connection lifetime (session resumption across reconnects).

## Setup (fresh machine)

### What you need

| Requirement | Notes |
|---|---|
| **Project Aria Gen1 glasses** with Client SDK access | SDK access is granted per-account by Meta ([Aria research program](https://www.projectaria.com/)); glasses must be set up in the **Aria mobile app** |
| **macOS** (tested on macOS 14; Apple Silicon and Intel both fine) | The Aria Client SDK also ships Linux wheels, but this project's audio/network notes are macOS-specific |
| **Python 3.11** | The `projectaria-client-sdk` wheel targets specific Python versions; 3.11 is what this project is tested on |
| **A Gemini API key** | Free at [aistudio.google.com](https://aistudio.google.com) → *Get API key* → *Create API key* |
| **Headphones with a mic** (e.g. AirPods) | Any input/output pair works; the app prefers devices whose name matches `--audio-device` (default `AirPods`) and falls back to the system default |

### 1. Install

```bash
git clone <this repo> && cd gemini_assistant
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # includes the Aria Client SDK
```

### 2. Pair the glasses (one-time)

With the glasses set up in the Aria mobile app and connected to your Mac via USB:

```bash
aria auth pair          # then approve the prompt in the Aria mobile app
```

### 3. Fix macOS networking (one-time, important)

When Aria streams over USB, macOS ranks the glasses' USB network link **above Wi-Fi**,
hijacking the default route and DNS — your Mac loses internet, so the Gemini
connection dies with DNS errors. Fix permanently:

```bash
aria_doctor             # answer "y" to lower the Aria interface priority (asks for sudo)
```

### 4. Configure

```bash
cp .env.example .env    # then paste your Gemini API key into .env
```

`.env` knobs: `GEMINI_API_KEY` (required), `GEMINI_MODEL` (default
`gemini-3.1-flash-live-preview`; a fallback model is listed in `.env.example`),
`GEMINI_VOICE` (default `Kore` — pinned so the voice never changes mid-session).

### 5. Verify incrementally (recommended order)

```bash
python -m pytest tests/ -q                    # unit tests, no hardware
python -m assistant --audio-test              # mic + speaker + Gemini, no glasses
python -m assistant --interface usb --task chat    # + glasses over USB
python -m assistant --interface wifi --device-ip <GLASSES_IP> --task chat  # cable-free
```

The first mic use triggers a macOS microphone-permission prompt for your terminal —
allow it. Find the glasses' IP in the Aria app (Device settings → Wi-Fi); Mac and
glasses must be on the same network, and university networks (eduroam etc.) usually
block device-to-device traffic — use home Wi-Fi or a phone hotspot.

## Usage

```bash
# glasses over USB
python -m assistant --interface usb --task chat

# glasses over WiFi (find the IP in the Aria app)
python -m assistant --interface wifi --device-ip <GLASSES_IP> --task chat
```

Transcript + live camera: **http://localhost:8899** (opens automatically).
Stop with Ctrl-C (WiFi teardown can take ~20 s).

## Tests

```bash
python -m pytest tests/ -v
```

Covers the pure logic: transcript stitching, panel word-wrap, wall-clock frame/audio
padding math. Hardware paths (glasses, AirPods, Live API) are verified behaviorally.

## Known limitations

- **No MPS eye gaze during live AI**: Aria Gen1 cannot record VRS on-device while
  streaming, so gaze capture and live assistance are mutually exclusive — an open
  study-design question (Gen2 hardware, alternating phases, or gaze from the streamed
  EyeTrack camera).
- The glasses drop WiFi when idle/asleep — wear them (or keep unfolded), keep the Aria
  app connected, keep them charged. Once streaming, the link is stable.
- Preview model IDs churn; if the model errors at connect, try the fallback listed in
  `.env.example`.
