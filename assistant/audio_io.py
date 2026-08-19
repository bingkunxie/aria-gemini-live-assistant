import asyncio
import threading
import time

import numpy as np
import sounddevice as sd

IN_RATE = 16000   # Gemini Live input requirement (PCM16 mono)
OUT_RATE = 24000  # Gemini Live output format
IN_BLOCK = 640    # 40 ms
OUT_BLOCK = 1200  # 50 ms


class AudioIO:
    """Full-duplex audio. Mic chunks land on an asyncio queue; playback is a
    flushable byte buffer so barge-in interruptions cut Gemini off promptly."""

    def __init__(self, loop: asyncio.AbstractEventLoop, prefer: str = "AirPods", on_mic_chunk=None,
                 echo_guard: float = 0.0):
        self.loop = loop
        self.prefer = prefer
        self.on_mic_chunk = on_mic_chunk  # called on PortAudio thread (recorder tap)
        self.echo_guard = echo_guard      # seconds; >0 enables the half-duplex gate
        self.mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._pending = bytearray()
        self._lock = threading.Lock()
        self._last_output_at = 0.0
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
        guard = f"echo guard: {self.echo_guard:.2f}s" if self.echo_guard > 0 else "echo guard: off"
        print(f"[audio] mic: {in_name}   speaker: {out_name}   {guard}")
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
    def _mic_muted_by_echo(self) -> bool:
        """Half-duplex gate. When mic and speaker share a room (built-in audio),
        the mic picks up Gemini's own voice, which the server hears as barge-in
        and cuts itself off. While playback is active — plus a short hangover
        covering speaker latency and reverb — mic audio is not forwarded.
        Disabled (0) with headphones, where real barge-in should still work."""
        if self.echo_guard <= 0:
            return False
        return (time.monotonic() - self._last_output_at) < self.echo_guard

    def _in_cb(self, indata, frames, t, status) -> None:
        chunk = bytes(indata)
        if self.on_mic_chunk:
            self.on_mic_chunk(chunk)  # recorder keeps the full mic track regardless
        if self._mic_muted_by_echo():
            return
        self.loop.call_soon_threadsafe(self._enqueue, chunk)

    def _enqueue(self, chunk: bytes) -> None:
        try:
            self.mic_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass  # under a stalled session, dropping new mic audio is the right failure mode

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
        if chunk:
            self._last_output_at = time.monotonic()
        chunk = chunk.ljust(need, b"\x00")
        outdata[:] = np.frombuffer(chunk, dtype=np.int16).reshape(-1, 1)
