import time
import wave

import numpy as np

from assistant.recorder import Recorder, frames_due, gemini_pad_samples


def test_frames_due_tracks_wall_clock():
    # at 2.0s and 10fps, 20 frames should exist; 14 written -> 6 due
    assert frames_due(2.0, 14, 10) == 6


def test_frames_due_never_negative():
    assert frames_due(1.0, 25, 10) == 0


def test_pad_fills_gap_to_wall_clock():
    # 2.0s elapsed, 24000 samples already written, incoming chunk of 4800 samples
    assert gemini_pad_samples(2.0, 24000, 4800 * 2) == 48000 - 24000 - 4800


def test_pad_never_negative():
    assert gemini_pad_samples(0.1, 24000, 9600) == 0


def test_mic_padded_to_wall_clock(tmp_path):
    rec = Recorder(tmp_path)
    time.sleep(0.2)
    rec.add_mic(b"\x00\x00" * 640)
    rec.close_writers()
    with wave.open(str(tmp_path / "mic.wav")) as w:
        # ~0.2s of silence padding (3200 samples) + the 640-sample chunk
        assert w.getnframes() >= 0.15 * 16000 + 640


def test_wavs_and_video_created(tmp_path):
    rec = Recorder(tmp_path)
    rec.add_mic(b"\x00\x00" * 640)
    rec.add_model_audio(b"\x00\x00" * 1200)
    frame = np.zeros((1408, 1408, 3), dtype=np.uint8)
    time.sleep(0.25)  # let wall clock advance so frames become due
    rec.add_frame(frame, [("model", "I suggest testing.")])
    assert rec._vid_frames >= 2  # duplicates written to match elapsed time
    rec.add_frame(None, [])  # missing frame must not crash
    rec.close_writers()
    with wave.open(str(tmp_path / "mic.wav")) as w:
        assert w.getframerate() == 16000 and w.getnframes() >= 640
    with wave.open(str(tmp_path / "gemini.wav")) as w:
        assert w.getframerate() == 24000 and w.getnframes() >= 1200
    assert (tmp_path / "demo_video.mp4").stat().st_size > 0
