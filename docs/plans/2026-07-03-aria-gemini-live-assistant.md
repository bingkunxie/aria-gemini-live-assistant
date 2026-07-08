# Aria + Gemini Live Cooking Assistant — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Python app that streams Aria Gen1 RGB video to the Gemini Live API, holds a real-time voice conversation via AirPods, shows a live browser transcript, and (in record mode) saves a composite demo MP4 + audio + transcript + research event log.

**Architecture:** Single asyncio process. Aria SDK and PortAudio callbacks (their own threads) hand data into the loop via thread-safe queues. One `GeminiSession` owns the Live API WebSocket (send audio + 1 fps JPEG frames; receive audio + transcriptions), with sliding-window compression and session resumption for reconnects. A `TranscriptHub` fans events out to a local web page, `events.jsonl`, and the recorder's side panel.

**Tech Stack:** Python 3.11 (existing venv `~/projectaria_tools_python_env`), `projectaria_client_sdk` 2.4.0, `google-genai`, `sounddevice`, `aiohttp`, `opencv-python`, `python-dotenv`, `pytest`, `ffmpeg`.

**Spec:** `docs/specs/2026-07-03-aria-gemini-live-assistant-design.md`

**Conventions used below:**
- `PY` = `~/projectaria_tools_python_env/bin/python`. All commands run from `~/ProjectAria/gemini_assistant/`.
- Steps marked **[USER ACTION]** require the human (API key creation, wearing devices, speaking). The executor must stop and ask, not fake these.
- Verified API facts (docs 2026-07-03): audio in = PCM16 16 kHz mono, out = PCM16 24 kHz; JPEG ≤1 fps via `send_realtime_input(video=Blob)`; transcription via `input_audio_transcription`/`output_audio_transcription`; compression via `ContextWindowCompressionConfig(sliding_window=SlidingWindow())`; resumption via `SessionResumptionConfig(handle=...)`. If a `google-genai` attribute in this plan doesn't exist at runtime, check https://ai.google.dev/gemini-api/docs/live-guide — field names, not concepts, may have drifted.

## File structure

```
gemini_assistant/
├── assistant/
│   ├── __init__.py
│   ├── __main__.py          # python -m assistant → app.main()
│   ├── app.py               # CLI, wiring, lifecycle
│   ├── aria_source.py       # glasses → latest RGB frame / JPEG
│   ├── audio_io.py          # AirPods duplex audio
│   ├── gemini_session.py    # Live API session + reconnect
│   ├── transcript_hub.py    # events → browser WS + jsonl + stitched lines
│   ├── panel.py             # wrap_text + render_panel (pure/CV, testable)
│   ├── prompts.py           # system prompt builder
│   └── recorder.py          # MP4 + WAVs + ffmpeg mux
├── tests/
│   ├── test_panel.py
│   ├── test_hub.py
│   └── test_recorder.py
├── requirements.txt
├── .env.example             # .env itself is gitignored
└── docs/{specs,plans}/
```

---

### Task 1: Environment + skeleton

**Files:** Create `requirements.txt`, `.env.example`, `assistant/__init__.py`, `assistant/__main__.py`, empty test dir.

- [ ] **Step 1.1: Write requirements.txt**

```
google-genai>=1.0
sounddevice>=0.4.6
aiohttp>=3.9
python-dotenv>=1.0
pytest>=8.0
```

- [ ] **Step 1.2: Install into the Aria venv**

Run: `~/projectaria_tools_python_env/bin/pip install -r requirements.txt`
Expected: all install without breaking `projectaria_client_sdk` (pip prints no red dependency-conflict error mentioning aria packages).

- [ ] **Step 1.3: Sanity imports**

Run: `PY -c "from google import genai; from google.genai import types; import sounddevice, aiohttp, aria.sdk; print('ok', genai.__version__)"`
Expected: `ok <version>`. If `aria.sdk` import breaks after installs, `pip check` and report — do not proceed.

- [ ] **Step 1.4: ffmpeg present?**

Run: `ffmpeg -version | head -1 || brew install ffmpeg`
Expected: a version line. (Homebrew install is several minutes; fine.)

- [ ] **Step 1.5: Create `.env.example` and package skeleton**

`.env.example`:
```
GEMINI_API_KEY=paste-your-key-here
GEMINI_MODEL=gemini-3.1-flash-live-preview
# Fallback model if the above misbehaves:
# GEMINI_MODEL=gemini-2.5-flash-native-audio-preview-12-2025
```

`assistant/__init__.py`: empty.
`assistant/__main__.py`:
```python
from assistant.app import main

main()
```

- [ ] **Step 1.6: Commit**

```bash
git add -A && git commit -m "chore: project skeleton and dependencies"
```

---

### Task 2: `panel.py` — text wrapping + transcript panel (TDD)

**Files:** Create `assistant/panel.py`, `tests/test_panel.py`.

- [ ] **Step 2.1: Write failing tests**

`tests/test_panel.py`:
```python
import numpy as np

from assistant.panel import render_panel, wrap_text


def test_wrap_short_line_unchanged():
    assert wrap_text("hello world", 20) == ["hello world"]


def test_wrap_breaks_on_words():
    assert wrap_text("cut the lemon in half now", 12) == [
        "cut the",
        "lemon in",
        "half now",
    ]


def test_wrap_hard_breaks_long_word():
    assert wrap_text("aaaaaaaaaa", 4) == ["aaaa", "aaaa", "aa"]


def test_wrap_empty():
    assert wrap_text("", 10) == []


def test_render_panel_shape_and_dtype():
    panel = render_panel([("user", "hi"), ("model", "I suggest washing the lemon")])
    assert panel.shape == (900, 400, 3)
    assert panel.dtype == np.uint8


def test_render_panel_empty_lines():
    assert render_panel([]).shape == (900, 400, 3)
```

- [ ] **Step 2.2: Run to verify failure**

Run: `PY -m pytest tests/test_panel.py -v`
Expected: FAIL — `ModuleNotFoundError: assistant.panel`.

- [ ] **Step 2.3: Implement `assistant/panel.py`**

```python
import cv2
import numpy as np

PANEL_W, PANEL_H = 400, 900
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def wrap_text(text: str, max_chars: int) -> list[str]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        while len(word) > max_chars:
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:max_chars])
            word = word[max_chars:]
        cand = f"{cur} {word}".strip()
        if len(cand) <= max_chars:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def render_panel(lines: list[tuple[str, str]], width: int = PANEL_W, height: int = PANEL_H) -> np.ndarray:
    """Draw stitched transcript lines bottom-up so the newest text is always visible.

    lines: [(role, text)] where role is "user" or "model".
    """
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width, 36), (40, 40, 40), -1)
    cv2.putText(panel, "Transcript", (12, 25), _FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    y = height - 14
    for role, text in reversed(lines):
        color = (90, 220, 90) if role == "user" else (60, 200, 255)
        prefix = "You: " if role == "user" else "AI: "
        for seg in reversed(wrap_text(prefix + text, 34)):
            cv2.putText(panel, seg, (10, y), _FONT, 0.5, color, 1, cv2.LINE_AA)
            y -= 22
            if y < 55:
                return panel
        y -= 8
    return panel
```

- [ ] **Step 2.4: Run tests**

Run: `PY -m pytest tests/test_panel.py -v`
Expected: 6 passed.

- [ ] **Step 2.5: Commit**

```bash
git add assistant/panel.py tests/test_panel.py && git commit -m "feat: transcript panel rendering with word wrap"
```

---

### Task 3: `transcript_hub.py` — events, stitching, browser window (TDD for stitching)

**Files:** Create `assistant/transcript_hub.py`, `tests/test_hub.py`.

- [ ] **Step 3.1: Write failing tests for stitching + event records**

`tests/test_hub.py`:
```python
from assistant.transcript_hub import TranscriptHub


def make_hub(tmp_path):
    return TranscriptHub(session_dir=tmp_path)


def test_same_role_fragments_merge(tmp_path):
    hub = make_hub(tmp_path)
    hub.emit("model_transcript", "I suggest ")
    hub.emit("model_transcript", "cutting the lemon.")
    assert hub.lines == [("model", "I suggest cutting the lemon.")]


def test_role_change_starts_new_line(tmp_path):
    hub = make_hub(tmp_path)
    hub.emit("model_transcript", "Wash the lemon.")
    hub.emit("user_transcript", "no thanks")
    hub.emit("user_transcript", " I'll cut first")
    assert hub.lines == [("model", "Wash the lemon."), ("user", "no thanks I'll cut first")]


def test_turn_complete_splits_model_utterances(tmp_path):
    hub = make_hub(tmp_path)
    hub.emit("model_transcript", "First suggestion.")
    hub.emit("turn_complete")
    hub.emit("model_transcript", "Second suggestion.")
    assert hub.lines == [("model", "First suggestion."), ("model", "Second suggestion.")]


def test_events_jsonl_written(tmp_path):
    hub = make_hub(tmp_path)
    hub.emit("user_transcript", "hello")
    hub.emit("frame_sent")
    hub.close()
    rows = [l for l in (tmp_path / "events.jsonl").read_text().splitlines() if l]
    assert len(rows) == 2
    import json

    ev = json.loads(rows[0])
    assert ev["type"] == "user_transcript" and ev["text"] == "hello"
    assert "t_wall" in ev and "t_session" in ev


def test_transcript_txt(tmp_path):
    hub = make_hub(tmp_path)
    hub.emit("user_transcript", "hi")
    hub.emit("model_transcript", "Hello!")
    hub.save_txt(tmp_path / "transcript.txt")
    assert (tmp_path / "transcript.txt").read_text() == "You: hi\nAI: Hello!\n"
```

- [ ] **Step 3.2: Run to verify failure**

Run: `PY -m pytest tests/test_hub.py -v`
Expected: FAIL — module not found.

- [ ] **Step 3.3: Implement `assistant/transcript_hub.py`**

```python
import asyncio
import json
import time
from pathlib import Path

from aiohttp import WSMsgType, web

_ROLE = {"user_transcript": "user", "model_transcript": "model"}

_PAGE = """<!doctype html>
<meta charset="utf-8"><title>Aria x Gemini transcript</title>
<style>
 body{background:#111;color:#eee;font-family:-apple-system,sans-serif;margin:0}
 #log{max-width:720px;margin:0 auto;padding:16px 16px 60px}
 .b{padding:8px 12px;border-radius:10px;margin:6px 0;white-space:pre-wrap;line-height:1.4}
 .user{background:#1e4620;margin-left:15%}
 .model{background:#123a5c;margin-right:15%}
 .status{color:#888;font-size:12px;text-align:center}
 h3{text-align:center;color:#aaa;font-weight:normal}
</style>
<h3>Aria &times; Gemini — live transcript</h3><div id="log"></div>
<script>
const log=document.getElementById('log');let last=null,lastRole=null;
function add(role,text,replace){
  if(replace&&lastRole===role&&last){last.textContent=text;}
  else{last=document.createElement('div');last.className='b '+role;last.textContent=text;log.appendChild(last);lastRole=role;}
  window.scrollTo(0,document.body.scrollHeight);
}
const ws=new WebSocket(`ws://${location.host}/ws`);
ws.onmessage=e=>{const m=JSON.parse(e.data);
  if(m.kind==='line')add(m.role,(m.role==='user'?'You: ':'AI: ')+m.text,m.merged);
  else if(m.kind==='status')add('status',m.text,false);};
ws.onclose=()=>add('status','[disconnected]',false);
</script>"""


class TranscriptHub:
    """Collects events; stitches transcript lines; fans out to jsonl + websocket clients."""

    def __init__(self, session_dir: Path | None = None):
        self.t0 = time.monotonic()
        self.lines: list[tuple[str, str]] = []
        self._open_role: str | None = None
        self._clients: set[web.WebSocketResponse] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._events_f = None
        if session_dir is not None:
            Path(session_dir).mkdir(parents=True, exist_ok=True)
            self._events_f = open(Path(session_dir) / "events.jsonl", "a")

    # -- events ------------------------------------------------------------
    def emit(self, type: str, text: str = "", detail: dict | None = None) -> None:
        ev = {
            "t_wall": time.time(),
            "t_session": round(time.monotonic() - self.t0, 3),
            "type": type,
            "text": text,
        }
        if detail:
            ev["detail"] = detail
        if self._events_f:
            self._events_f.write(json.dumps(ev) + "\n")
            self._events_f.flush()

        role = _ROLE.get(type)
        if role:
            merged = self._open_role == role and bool(self.lines)
            if merged:
                prev_role, prev_text = self.lines[-1]
                self.lines[-1] = (prev_role, prev_text + text)
            else:
                self.lines.append((role, text))
            self._open_role = role
            self._broadcast({"kind": "line", "role": role, "text": self.lines[-1][1], "merged": merged})
        elif type == "turn_complete":
            self._open_role = None
        elif type in ("session_event", "error"):
            self._open_role = None
            self._broadcast({"kind": "status", "text": text})
        # frame_sent etc: jsonl only

    def save_txt(self, path: Path) -> None:
        with open(path, "w") as f:
            for role, text in self.lines:
                f.write(("You: " if role == "user" else "AI: ") + text + "\n")

    def close(self) -> None:
        if self._events_f:
            self._events_f.close()
            self._events_f = None

    # -- web ---------------------------------------------------------------
    def _broadcast(self, msg: dict) -> None:
        if not self._clients or self._loop is None:
            return
        data = json.dumps(msg)
        for ws in list(self._clients):
            self._loop.create_task(ws.send_str(data))

    async def _ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        # replay existing lines to late joiners
        for role, text in self.lines:
            await ws.send_str(json.dumps({"kind": "line", "role": role, "text": text, "merged": False}))
        self._clients.add(ws)
        try:
            async for msg in ws:
                if msg.type == WSMsgType.ERROR:
                    break
        finally:
            self._clients.discard(ws)
        return ws

    async def start_server(self, port: int = 8899) -> None:
        self._loop = asyncio.get_running_loop()
        app = web.Application()
        app.router.add_get("/", lambda r: web.Response(text=_PAGE, content_type="text/html"))
        app.router.add_get("/ws", self._ws_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
```

- [ ] **Step 3.4: Run tests**

Run: `PY -m pytest tests/test_hub.py tests/test_panel.py -v`
Expected: all pass.

- [ ] **Step 3.5: Visual check of the browser page**

Run this throwaway command, then open http://localhost:8899 in a browser:
```bash
PY - <<'EOF'
import asyncio
from assistant.transcript_hub import TranscriptHub

async def main():
    hub = TranscriptHub()
    await hub.start_server()
    print("open http://localhost:8899 — fake conversation starts in 5s, Ctrl-C to stop")
    await asyncio.sleep(5)
    hub.emit("session_event", text="connected")
    for i in range(5):
        hub.emit("model_transcript", f"Suggestion {i}: I suggest doing step {i}. ")
        await asyncio.sleep(1)
        hub.emit("turn_complete")
        hub.emit("user_transcript", f"okay doing step {i}")
        await asyncio.sleep(1)
    await asyncio.sleep(600)

asyncio.run(main())
EOF
```
Expected: green user bubbles on the right, blue AI bubbles on the left, streaming in live, auto-scrolling.

- [ ] **Step 3.6: Commit**

```bash
git add assistant/transcript_hub.py tests/test_hub.py && git commit -m "feat: transcript hub with jsonl log and live browser window"
```

---

### Task 4: `audio_io.py` — AirPods duplex audio

**Files:** Create `assistant/audio_io.py`.

- [ ] **Step 4.1: Implement `assistant/audio_io.py`**

```python
import asyncio
import threading

import numpy as np
import sounddevice as sd

IN_RATE = 16000   # Gemini Live input requirement (PCM16 mono)
OUT_RATE = 24000  # Gemini Live output format
IN_BLOCK = 640    # 40 ms
OUT_BLOCK = 1200  # 50 ms


class AudioIO:
    """Full-duplex audio. Mic chunks land on an asyncio queue; playback is a
    flushable byte buffer so barge-in interruptions cut Gemini off promptly."""

    def __init__(self, loop: asyncio.AbstractEventLoop, prefer: str = "AirPods", on_mic_chunk=None):
        self.loop = loop
        self.prefer = prefer
        self.on_mic_chunk = on_mic_chunk  # called on PortAudio thread (recorder tap)
        self.mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._pending = bytearray()
        self._lock = threading.Lock()
        self._in = None
        self._out = None

    def _find_device(self, kind: str):
        for i, dev in enumerate(sd.query_devices()):
            if self.prefer.lower() in dev["name"].lower() and dev[f"max_{kind}_channels"] > 0:
                return i
        return None  # PortAudio default

    def start(self) -> None:
        in_dev = self._find_device("input")
        out_dev = self._find_device("output")
        in_name = sd.query_devices(in_dev)["name"] if in_dev is not None else "(system default)"
        out_name = sd.query_devices(out_dev)["name"] if out_dev is not None else "(system default)"
        print(f"[audio] mic: {in_name}   speaker: {out_name}")
        self._in = sd.InputStream(
            device=in_dev, samplerate=IN_RATE, channels=1, dtype="int16",
            blocksize=IN_BLOCK, callback=self._in_cb,
        )
        self._out = sd.OutputStream(
            device=out_dev, samplerate=OUT_RATE, channels=1, dtype="int16",
            blocksize=OUT_BLOCK, callback=self._out_cb,
        )
        self._in.start()
        self._out.start()

    def stop(self) -> None:
        for s in (self._in, self._out):
            if s is not None:
                s.stop()
                s.close()

    # -- capture (PortAudio thread) -----------------------------------------
    def _in_cb(self, indata, frames, t, status) -> None:
        chunk = bytes(indata)
        if self.on_mic_chunk:
            self.on_mic_chunk(chunk)
        self.loop.call_soon_threadsafe(self._enqueue, chunk)

    def _enqueue(self, chunk: bytes) -> None:
        try:
            self.mic_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass  # drop oldest-style behavior isn't needed; dropping newest is fine under stall

    # -- playback ------------------------------------------------------------
    def play(self, pcm: bytes) -> None:
        with self._lock:
            self._pending.extend(pcm)

    def flush_playback(self) -> None:
        with self._lock:
            self._pending.clear()

    def _out_cb(self, outdata, frames, t, status) -> None:
        need = frames * 2
        with self._lock:
            chunk = bytes(self._pending[:need])
            del self._pending[: len(chunk)]
        chunk = chunk.ljust(need, b"\x00")
        outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)
```

- [ ] **Step 4.2: [USER ACTION] Loopback test with AirPods**

Ask the user to connect AirPods (System Settings → Bluetooth), then run:
```bash
PY - <<'EOF'
import asyncio
from assistant.audio_io import AudioIO, IN_RATE

async def main():
    audio = AudioIO(asyncio.get_running_loop())
    audio.start()
    print("Say something for 3 seconds...")
    chunks = []
    end = asyncio.get_running_loop().time() + 3
    while asyncio.get_running_loop().time() < end:
        chunks.append(await audio.mic_queue.get())
    print("Playing it back (pitch will sound ~1.5x high — 16k data on a 24k stream; that is expected here).")
    audio.play(b"".join(chunks))
    await asyncio.sleep(3)
    audio.stop()

asyncio.run(main())
EOF
```
Expected: user hears their own voice through AirPods (chipmunk-pitched — fine, this only proves capture+playback paths; Gemini audio arrives natively at 24 kHz). First mic use triggers a macOS microphone-permission prompt for the terminal — user must click Allow.

- [ ] **Step 4.3: Commit**

```bash
git add assistant/audio_io.py && git commit -m "feat: AirPods duplex audio io"
```

---

### Task 5: `prompts.py` + `gemini_session.py` + `--audio-test` mode (bring-up stage 1)

**Files:** Create `assistant/prompts.py`, `assistant/gemini_session.py`, `assistant/app.py`.

- [ ] **Step 5.1: [USER ACTION] Create the Gemini API key**

Guide the user:
1. Open https://aistudio.google.com and sign in with a Google account (a personal one is fine).
2. Click **Get API key** (key icon, left sidebar) → **Create API key**.
3. Copy the key.
4. `cp .env.example .env` and paste the key into `GEMINI_API_KEY=`.

Verify (never print the key): `PY -c "from dotenv import dotenv_values; v=dotenv_values('.env'); print('key set:', bool(v.get('GEMINI_API_KEY') and 'paste' not in v['GEMINI_API_KEY']))"`
Expected: `key set: True`

- [ ] **Step 5.2: Implement `assistant/prompts.py`**

```python
_BASE = """You are a hands-on cooking coach speaking with someone wearing
camera glasses; you see what they see through periodic snapshots. You are part
of a research prototype studying how people rely on AI suggestions.

Rules:
- Speak English, short conversational utterances (1-3 sentences). This is a
  live voice conversation; never lecture.
- Guide the task ONE step at a time. Phrase each recommendation explicitly as
  a suggestion, e.g. "I suggest cutting the lemon in half next."
- Watch what the person actually does. If they follow your suggestion,
  acknowledge briefly and move on. If they do something different or decline,
  accept it gracefully, adapt your plan to their path, and never nag or repeat
  a rejected suggestion.
- Ground your remarks in what you can actually see. If you can't see clearly,
  say so or ask.
- Wait for the person to act or respond; don't chain multiple suggestions.
"""

_TASKS = {
    "lemonade": """
Current task: help them make a cup of lemonade. Loose plan (adapt freely to
what you observe; steps may be reordered or skipped):
wash the lemon -> cut it -> squeeze juice into the cup -> add water ->
add sugar -> stir -> taste and adjust.
Start, when you first see the scene, by greeting them in one sentence and
offering your first suggestion.""",
    "chat": """
Current task: none yet. Converse naturally and describe what you see when
asked. This is a systems test.""",
}


def build_prompt(task: str) -> str:
    return _BASE + _TASKS[task]
```

- [ ] **Step 5.3: Implement `assistant/gemini_session.py`**

```python
import asyncio
import traceback

from google import genai
from google.genai import types


class GeminiSession:
    """Owns the Live API connection: sends mic audio and camera frames,
    receives model audio + transcriptions, survives reconnects via session
    resumption + sliding-window compression."""

    def __init__(self, api_key, model, system_prompt, hub, audio_io,
                 get_frame_jpeg=None, on_model_audio=None):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.hub = hub
        self.audio_io = audio_io
        self.get_frame_jpeg = get_frame_jpeg
        self.on_model_audio = on_model_audio
        self.resume_handle = None
        self.running = True

    def _config(self):
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=self.system_prompt,
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            session_resumption=types.SessionResumptionConfig(handle=self.resume_handle),
        )

    async def run(self):
        while self.running:
            try:
                async with self.client.aio.live.connect(model=self.model, config=self._config()) as session:
                    self.hub.emit("session_event", text="Gemini connected")
                    send_audio = asyncio.create_task(self._send_audio(session))
                    send_frames = (
                        asyncio.create_task(self._send_frames(session))
                        if self.get_frame_jpeg else None
                    )
                    try:
                        await self._receive(session)  # returns when server ends the connection
                    finally:
                        send_audio.cancel()
                        if send_frames:
                            send_frames.cancel()
                if self.running:
                    self.hub.emit("session_event", text="connection ended; resuming session")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if not self.running:
                    break
                traceback.print_exc()
                self.hub.emit("error", text=f"session error: {e!r}; reconnecting in 1s")
                await asyncio.sleep(1)

    def stop(self):
        self.running = False

    async def _send_audio(self, session):
        while True:
            chunk = await self.audio_io.mic_queue.get()
            await session.send_realtime_input(
                audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
            )

    async def _send_frames(self, session):
        while True:
            jpeg = self.get_frame_jpeg()
            if jpeg is not None:
                await session.send_realtime_input(
                    video=types.Blob(data=jpeg, mime_type="image/jpeg")
                )
                self.hub.emit("frame_sent")
            await asyncio.sleep(1.0)

    async def _receive(self, session):
        async for msg in session.receive():
            sru = getattr(msg, "session_resumption_update", None)
            if sru is not None and getattr(sru, "resumable", False) and getattr(sru, "new_handle", None):
                self.resume_handle = sru.new_handle
            if getattr(msg, "go_away", None) is not None:
                self.hub.emit("session_event", text="server GoAway; will reconnect")

            sc = getattr(msg, "server_content", None)
            if sc is not None:
                if getattr(sc, "interrupted", None):
                    self.audio_io.flush_playback()
                    self.hub.emit("session_event", text="[interrupted]")
                it = getattr(sc, "input_transcription", None)
                if it is not None and it.text:
                    self.hub.emit("user_transcript", it.text)
                ot = getattr(sc, "output_transcription", None)
                if ot is not None and ot.text:
                    self.hub.emit("model_transcript", ot.text)
                if getattr(sc, "turn_complete", None):
                    self.hub.emit("turn_complete")

            data = getattr(msg, "data", None)
            if data:
                self.audio_io.play(data)
                if self.on_model_audio:
                    self.on_model_audio(data)
```

- [ ] **Step 5.4: Implement first version of `assistant/app.py` (audio-test + live scaffolding)**

```python
import argparse
import asyncio
import datetime
import os
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

from assistant.audio_io import AudioIO
from assistant.gemini_session import GeminiSession
from assistant.prompts import build_prompt
from assistant.transcript_hub import TranscriptHub

ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(prog="assistant")
    p.add_argument("--audio-test", action="store_true", help="voice only, no glasses")
    p.add_argument("--interface", choices=["usb", "wifi"], default="usb")
    p.add_argument("--device-ip", help="glasses IP for wifi streaming")
    p.add_argument("--record", action="store_true")
    p.add_argument("--task", default="lemonade", choices=["lemonade", "chat"])
    p.add_argument("--audio-device", default="AirPods")
    return p.parse_args()


async def run(args):
    load_dotenv(ROOT / ".env")
    api_key = os.environ["GEMINI_API_KEY"]
    model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
    loop = asyncio.get_running_loop()

    session_dir = None
    recorder = None
    if args.record:
        from assistant.recorder import Recorder  # Task 8

        session_dir = ROOT / "sessions" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        recorder = Recorder(session_dir)

    hub = TranscriptHub(session_dir=session_dir)
    await hub.start_server()
    print("transcript window: http://localhost:8899")
    webbrowser.open("http://localhost:8899")

    audio = AudioIO(loop, prefer=args.audio_device,
                    on_mic_chunk=recorder.add_mic if recorder else None)
    audio.start()

    aria_src = None
    if not args.audio_test:
        from assistant.aria_source import AriaSource  # Task 6

        aria_src = AriaSource(
            interface=args.interface, device_ip=args.device_ip,
            on_status=lambda m: hub.emit("session_event", text=m),
        )
        aria_src.start()

    task = args.task if not args.audio_test else "chat"
    gem = GeminiSession(
        api_key, model, build_prompt(task), hub, audio,
        get_frame_jpeg=aria_src.latest_jpeg if aria_src else None,
        on_model_audio=recorder.add_model_audio if recorder else None,
    )

    tasks = [asyncio.create_task(gem.run())]
    if recorder and aria_src:
        async def frame_loop():
            while True:
                recorder.add_frame(aria_src.latest_frame(), hub.lines)
                await asyncio.sleep(1 / recorder.FPS)

        tasks.append(asyncio.create_task(frame_loop()))

    print("Running — press Ctrl-C to stop.")
    try:
        await asyncio.gather(*tasks)
    finally:
        gem.stop()
        for t in tasks:
            t.cancel()
        audio.stop()
        if aria_src:
            aria_src.stop()
        if recorder:
            recorder.finalize()
            hub.save_txt(session_dir / "transcript.txt")
            print(f"session artifacts: {session_dir}")
        hub.close()


def main():
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nstopped.")
```

- [ ] **Step 5.5: [USER ACTION] Bring-up stage 1 — talk to Gemini, voice only**

Run: `PY -m assistant --audio-test`
Ask the user (AirPods in) to say "Hi, can you hear me? What can you do?"
Expected: Gemini answers audibly through AirPods within ~1–2 s; both sides appear in the browser transcript window; Ctrl-C exits cleanly.
Debug notes: 403/permission error → key not active or `.env` not loaded. Attribute errors on `types.*` → SDK field drift; check the live-guide doc URL in the header. No mic audio → macOS mic permission for the terminal app.

- [ ] **Step 5.6: Commit**

```bash
git add assistant/prompts.py assistant/gemini_session.py assistant/app.py && git commit -m "feat: gemini live session with voice-only audio test mode"
```

---

### Task 6: `aria_source.py` + USB frame check

**Files:** Create `assistant/aria_source.py`.

- [ ] **Step 6.1: Implement `assistant/aria_source.py`** (mirrors the proven sample at `~/ProjectAria/client_sdk_samples/device_stream_cv2.py`; RGB-only subscription)

```python
import threading

import aria.sdk as aria
import cv2
import numpy as np


class AriaSource:
    """Connects to Aria Gen1, subscribes to the RGB stream, exposes the latest
    frame (rotated upright, RGB channel order) and a downscaled JPEG."""

    def __init__(self, interface="usb", device_ip=None, profile="profile18", on_status=None):
        self.interface = interface
        self.device_ip = device_ip
        self.profile = profile
        self.on_status = on_status or (lambda msg: None)
        self._lock = threading.Lock()
        self._frame = None
        self._device = None
        self._client = None
        self._streaming = False
        self._subscribed = False

    # observer callbacks arrive on SDK threads
    def on_image_received(self, image: np.ndarray, record) -> None:
        if record.camera_id == aria.CameraId.Rgb:
            with self._lock:
                self._frame = np.rot90(image, -1)

    def on_streaming_client_failure(self, reason, message: str) -> None:
        self.on_status(f"Aria stream failure: {reason}: {message}")

    def start(self) -> None:
        aria.set_log_level(aria.Level.Warning)
        self._client = aria.DeviceClient()
        cfg = aria.DeviceClientConfig()
        if self.device_ip:
            cfg.ip_v4_address = self.device_ip
        self._client.set_client_config(cfg)
        self._device = self._client.connect()

        mgr = self._device.streaming_manager
        s_cfg = aria.StreamingConfig()
        s_cfg.profile_name = self.profile
        if self.interface == "usb":
            s_cfg.streaming_interface = aria.StreamingInterface.Usb
        s_cfg.security_options.use_ephemeral_certs = True
        mgr.streaming_config = s_cfg

        sub = mgr.streaming_client.subscription_config
        sub.subscriber_data_type = aria.StreamingDataType.Rgb
        sub.message_queue_size[aria.StreamingDataType.Rgb] = 1
        sub.security_options.use_ephemeral_certs = True
        mgr.streaming_client.subscription_config = sub
        mgr.streaming_client.set_streaming_client_observer(self)

        mgr.start_streaming()
        self._streaming = True
        mgr.streaming_client.subscribe()
        self._subscribed = True
        self.on_status(f"Aria streaming via {self.interface}")

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def latest_jpeg(self, max_dim: int = 768, quality: int = 80) -> bytes | None:
        frame = self.latest_frame()
        if frame is None:
            return None
        h, w = frame.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def stop(self) -> None:
        if self._device is not None:
            try:
                if self._subscribed:
                    self._device.streaming_manager.streaming_client.unsubscribe()
                if self._streaming:
                    self._device.streaming_manager.stop_streaming()
            finally:
                self._client.disconnect(self._device)
```

- [ ] **Step 6.2: [USER ACTION] USB frame grab check**

Ask the user to plug the glasses in via USB (and confirm they're on / SDK pairing approved in the Aria app), then run:
```bash
PY - <<'EOF'
import time
from assistant.aria_source import AriaSource

src = AriaSource(interface="usb", on_status=print)
src.start()
for _ in range(50):
    time.sleep(0.2)
    jpeg = src.latest_jpeg()
    if jpeg:
        open("/tmp/aria_test_frame.jpg", "wb").write(jpeg)
        print("wrote /tmp/aria_test_frame.jpg", len(jpeg), "bytes")
        break
else:
    print("NO FRAME after 10s")
src.stop()
EOF
```
Expected: a JPEG lands in `/tmp/aria_test_frame.jpg`; open it and confirm it's an upright, correctly-colored view.

- [ ] **Step 6.3: Commit**

```bash
git add assistant/aria_source.py && git commit -m "feat: aria rgb source with jpeg export"
```

---

### Task 7: Live mode over USB (bring-up stage 2), then WiFi (stage 3)

No new files — integration verification of what Task 5 already wired.

- [ ] **Step 7.1: [USER ACTION] USB vision test**

Run: `PY -m assistant --interface usb --task chat`
User holds up an object and asks "What am I holding?"
Expected: correct answer grounded in the camera view; transcript window updating; `frame_sent` events accumulating in stdout-free silence (they're jsonl-only in record mode; in live mode they simply don't display — that's correct behavior).

- [ ] **Step 7.2: [USER ACTION] Two-minute-limit soak test**

Keep the session from step 7.1 running ≥4 minutes with occasional chat.
Expected: conversation survives past 2:00 (compression working). If the session dies at ~2:00, compression config isn't taking effect — stop and debug before proceeding (check `context_window_compression` in `_config()` against the live-guide doc).

- [ ] **Step 7.3: [USER ACTION] WiFi streaming**

Ask the user to: put Mac + glasses on the same home WiFi/hotspot (glasses WiFi is set in the Aria mobile app), find the glasses' IP in the app (Device settings → WiFi), then:
Run: `PY -m assistant --interface wifi --device-ip <GLASSES_IP> --task chat`
Expected: same behavior as USB, cable-free.
Fallback: if connection fails or frames stall (campus-network isolation), document it and use USB for the demo — the plan works either way.

- [ ] **Step 7.4: Commit any fixes**

```bash
git add -A && git commit -m "fix: adjustments from usb/wifi live bring-up"
```
(Skip if no changes.)

---

### Task 8: `recorder.py` (TDD for pad math) + record mode

**Files:** Create `assistant/recorder.py`, `tests/test_recorder.py`.

- [ ] **Step 8.1: Write failing tests**

`tests/test_recorder.py`:
```python
import wave

import numpy as np

from assistant.recorder import Recorder, gemini_pad_samples


def test_pad_fills_gap_to_wall_clock():
    # 2.0s elapsed, 24000 samples already written, incoming chunk of 4800 samples
    assert gemini_pad_samples(2.0, 24000, 4800 * 2) == 48000 - 24000 - 4800


def test_pad_never_negative():
    assert gemini_pad_samples(0.1, 24000, 9600) == 0


def test_wavs_and_video_created(tmp_path):
    rec = Recorder(tmp_path)
    rec.add_mic(b"\x00\x00" * 640)
    rec.add_model_audio(b"\x00\x00" * 1200)
    frame = np.zeros((1408, 1408, 3), dtype=np.uint8)
    rec.add_frame(frame, [("model", "I suggest testing.")])
    rec.add_frame(None, [])  # missing frame must not crash
    rec.close_writers()
    with wave.open(str(tmp_path / "mic.wav")) as w:
        assert w.getframerate() == 16000 and w.getnframes() == 640
    with wave.open(str(tmp_path / "gemini.wav")) as w:
        assert w.getframerate() == 24000 and w.getnframes() >= 1200
    assert (tmp_path / "demo_video.mp4").stat().st_size > 0
```

- [ ] **Step 8.2: Run to verify failure**

Run: `PY -m pytest tests/test_recorder.py -v`
Expected: FAIL — module not found.

- [ ] **Step 8.3: Implement `assistant/recorder.py`**

```python
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path

import cv2
import numpy as np

from assistant.panel import PANEL_H, PANEL_W, render_panel

VID_W, VID_H = 1200 + PANEL_W, PANEL_H  # 1600 x 900


def gemini_pad_samples(elapsed_s: float, samples_written: int, chunk_bytes: int, rate: int = 24000) -> int:
    """Model audio arrives in bursts; pad with silence so the WAV timeline
    matches wall clock (chunk is placed ending at 'now')."""
    target = int(elapsed_s * rate)
    return max(0, target - samples_written - chunk_bytes // 2)


def _fit(frame: np.ndarray | None, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if frame is None:
        cv2.putText(canvas, "no camera frame", (width // 2 - 120, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 200), 2, cv2.LINE_AA)
        return canvas
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    scale = min(width / w, height / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    x0, y0 = (width - nw) // 2, (height - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


class Recorder:
    FPS = 10

    def __init__(self, session_dir: Path):
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found — brew install ffmpeg")
        self.t0 = time.monotonic()
        self._lock = threading.Lock()
        self._gem_samples = 0
        self._closed = False
        self.mic_wav = wave.open(str(self.dir / "mic.wav"), "wb")
        self.mic_wav.setnchannels(1); self.mic_wav.setsampwidth(2); self.mic_wav.setframerate(16000)
        self.gem_wav = wave.open(str(self.dir / "gemini.wav"), "wb")
        self.gem_wav.setnchannels(1); self.gem_wav.setsampwidth(2); self.gem_wav.setframerate(24000)
        self.video = cv2.VideoWriter(
            str(self.dir / "demo_video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (VID_W, VID_H),
        )

    # called from PortAudio thread
    def add_mic(self, chunk: bytes) -> None:
        with self._lock:
            if not self._closed:
                self.mic_wav.writeframes(chunk)

    # called from asyncio thread
    def add_model_audio(self, chunk: bytes) -> None:
        with self._lock:
            if self._closed:
                return
            pad = gemini_pad_samples(time.monotonic() - self.t0, self._gem_samples, len(chunk))
            if pad:
                self.gem_wav.writeframes(b"\x00\x00" * pad)
                self._gem_samples += pad
            self.gem_wav.writeframes(chunk)
            self._gem_samples += len(chunk) // 2

    def add_frame(self, frame_rgb: np.ndarray | None, lines: list[tuple[str, str]]) -> None:
        composite = np.hstack((_fit(frame_rgb, VID_W - PANEL_W, VID_H), render_panel(lines)))
        with self._lock:
            if not self._closed:
                self.video.write(composite)

    def close_writers(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.mic_wav.close()
            self.gem_wav.close()
            self.video.release()

    def finalize(self) -> None:
        self.close_writers()
        cmd = [
            "ffmpeg", "-y",
            "-i", str(self.dir / "demo_video.mp4"),
            "-i", str(self.dir / "mic.wav"),
            "-i", str(self.dir / "gemini.wav"),
            "-filter_complex",
            "[1:a]aresample=48000[a1];[2:a]aresample=48000[a2];"
            "[a1][a2]amix=inputs=2:duration=longest[aout]",
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libx264", "-preset", "fast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            str(self.dir / "demo.mp4"),
        ]
        print("muxing demo.mp4 ...")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print("ffmpeg failed; raw artifacts are intact. stderr tail:")
            print("\n".join(res.stderr.splitlines()[-8:]))
        else:
            print(f"demo ready: {self.dir / 'demo.mp4'}")
```

- [ ] **Step 8.4: Run tests**

Run: `PY -m pytest tests/ -v`
Expected: all tests pass (panel, hub, recorder).

- [ ] **Step 8.5: [USER ACTION] 30-second record smoke test**

Run: `PY -m assistant --interface usb --task chat --record`
User chats briefly while pointing the glasses at something, Ctrl-C after ~30 s.
Expected: `sessions/<ts>/` contains `demo.mp4` (plays in QuickTime: camera view left, transcript panel right, both voices audible and in sync), `mic.wav`, `gemini.wav`, `transcript.txt`, `events.jsonl`.

- [ ] **Step 8.6: Commit**

```bash
git add assistant/recorder.py tests/test_recorder.py && git commit -m "feat: demo recorder with composite video and wall-clock-aligned audio"
```

---

### Task 9: Rehearsal + lemonade demo (bring-up stages 4–5)

- [ ] **Step 9.1: [USER ACTION] Rehearsal (live mode, no recording)**

Setup checklist for the user: lemon, knife, cutting board, cup, water, sugar, spoon within reach; AirPods in; glasses on; WiFi (or USB) confirmed from Task 7.
Run: `PY -m assistant --interface wifi --device-ip <IP> --task lemonade` (or `--interface usb`)
User performs a partial dry run (1–2 minutes) — deliberately reject one suggestion to confirm the coach adapts without nagging.
Expected: proactive greeting + first suggestion when the scene appears; graceful adaptation on rejection.
Prompt tuning between runs is expected here — edit `assistant/prompts.py`, commit tweaks.

- [ ] **Step 9.2: [USER ACTION] The real demo**

Run: same command with `--record`.
User makes the lemonade start-to-finish (5–6 min), Ctrl-C when done.
Expected: `sessions/<ts>/demo.mp4` = shareable demo; `events.jsonl` = research log.

- [ ] **Step 9.3: Verify artifacts + final commit**

Check: `demo.mp4` duration ≈ session length; audio in sync at start AND end (drift check); transcript readable.
```bash
git add -A && git commit -m "docs: demo session notes"
```

---

## Self-review (done at plan time)

- **Spec coverage:** live mode (T5–7), record mode (T8), transcript window (T3), AirPods (T4), proactive-coach prompt with rejectable suggestions (T5.2), compression + resumption (T5.3, soak-tested T7.2), WiFi-with-USB-fallback (T7.3), event log schema (T3.3), ffmpeg check (T8.3), bring-up stages 1–5 (T5.5, T7.1, T7.3, T9.1, T9.2). No gaps found.
- **Placeholder scan:** none; all code complete.
- **Type consistency:** `hub.emit(type, text)` signature consistent across T3/T5/T6; `lines: list[tuple[str,str]]` consistent between hub, panel, recorder; `latest_jpeg`/`latest_frame` names match between T6 and T5.4's app wiring; `Recorder.FPS` referenced in app frame loop matches T8.
- **Known risk (accepted):** exact `google-genai` message-field names (`session_resumption_update`, `go_away`, `input_transcription`) are per documented API as of 2026-07-03; step 5.5 is the smoke test that catches drift, with the doc URL in the header for repair.
