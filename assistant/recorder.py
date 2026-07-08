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


def frames_due(elapsed_s: float, frames_written: int, fps: int) -> int:
    """Frames the video should contain at this wall-clock moment. Writing
    duplicates when behind keeps the video timeline in sync with the audio."""
    return max(0, int(elapsed_s * fps) - frames_written)


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
        self._mic_samples = 0
        self._vid_frames = 0
        self._closed = False
        self.mic_wav = wave.open(str(self.dir / "mic.wav"), "wb")
        self.mic_wav.setnchannels(1)
        self.mic_wav.setsampwidth(2)
        self.mic_wav.setframerate(16000)
        self.gem_wav = wave.open(str(self.dir / "gemini.wav"), "wb")
        self.gem_wav.setnchannels(1)
        self.gem_wav.setsampwidth(2)
        self.gem_wav.setframerate(24000)
        self.video = cv2.VideoWriter(
            str(self.dir / "demo_video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), self.FPS, (VID_W, VID_H),
        )

    # called from PortAudio thread
    def add_mic(self, chunk: bytes) -> None:
        # anchor to wall clock like the model track: covers the device-open
        # delay at start and Bluetooth clock drift over long recordings
        with self._lock:
            if self._closed:
                return
            pad = gemini_pad_samples(time.monotonic() - self.t0, self._mic_samples,
                                     len(chunk), rate=16000)
            if pad:
                self.mic_wav.writeframes(b"\x00\x00" * pad)
                self._mic_samples += pad
            self.mic_wav.writeframes(chunk)
            self._mic_samples += len(chunk) // 2

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
        due = frames_due(time.monotonic() - self.t0, self._vid_frames, self.FPS)
        if due == 0:
            return
        composite = np.hstack((_fit(frame_rgb, VID_W - PANEL_W, VID_H), render_panel(lines)))
        with self._lock:
            if self._closed:
                return
            for _ in range(due):
                self.video.write(composite)
                self._vid_frames += 1

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
