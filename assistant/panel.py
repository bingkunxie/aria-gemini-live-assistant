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
