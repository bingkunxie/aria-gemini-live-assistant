import time

import numpy as np

from assistant.audio_io import AudioIO


def make_io(echo_guard):
    # loop is unused by the paths under test
    return AudioIO(loop=None, echo_guard=echo_guard)


def test_guard_off_never_mutes():
    io = make_io(0.0)
    io._last_output_at = time.monotonic()
    assert io._mic_muted_by_echo() is False


def test_guard_mutes_while_speaker_active():
    io = make_io(0.4)
    io._last_output_at = time.monotonic()
    assert io._mic_muted_by_echo() is True


def test_guard_reopens_after_hangover():
    io = make_io(0.4)
    io._last_output_at = time.monotonic() - 1.0
    assert io._mic_muted_by_echo() is False


def test_silent_playback_does_not_arm_guard():
    io = make_io(0.4)
    out = np.zeros((10, 1), dtype=np.int16)
    io._out_cb(out, 10, None, None)  # nothing queued -> pure silence
    assert io._last_output_at == 0.0
    assert io._mic_muted_by_echo() is False


def test_real_playback_arms_guard():
    io = make_io(0.4)
    out = np.zeros((10, 1), dtype=np.int16)
    io.play(b"\x01\x02" * 10)
    io._out_cb(out, 10, None, None)
    assert io._last_output_at > 0.0
    assert io._mic_muted_by_echo() is True


def test_recorder_tap_still_receives_muted_audio():
    io = make_io(0.4)
    io._last_output_at = time.monotonic()
    captured = []
    io.on_mic_chunk = captured.append
    io._in_cb(np.zeros((640, 1), dtype=np.int16), 640, None, None)
    # gate blocks the send to Gemini, but the recording keeps the full track
    assert len(captured) == 1
