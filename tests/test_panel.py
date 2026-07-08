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
