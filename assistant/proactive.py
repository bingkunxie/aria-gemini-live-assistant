import asyncio
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable


DECISION_PROMPT = """You are a low-interruption proactive first-person vision assistant.
Decide whether the wearer needs a short spoken reminder based on the latest camera image.

Return exactly one compact JSON object. Do not use Markdown. Do not include text outside JSON.
Required fields:
{
  "visual_context": "one objective sentence about the current image",
  "user_intent": "what the wearer appears to be doing, or unknown",
  "proactive_score": 1,
  "should_interrupt": false,
  "urgency": 0.0,
  "confidence": 0.0,
  "reason": "why you should or should not interrupt",
  "message": "if should_interrupt=true, one short spoken reminder; otherwise empty string"
}

Scoring:
1 = do not interrupt.
2 = weak signal, usually do not interrupt.
3 = potentially helpful, low-priority reminder.
4 = clearly helpful or minor risk, should remind.
5 = safety risk, obvious error, or immediate consequence, must remind.

Only interrupt for clear new information, risk, mistakes, omissions, opportunities, or a helpful next step.
Do not interrupt for ordinary, static, repeated, or unclear scenes."""


@dataclass
class ProactiveConfig:
    interval: float = 5.0
    threshold: float = 4.0
    min_confidence: float = 0.55
    cooldown: float = 15.0
    model: str = "gemini-2.5-flash"
    say: bool = False


def number_or_default(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def bool_or_default(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "1"):
            return True
        if lowered in ("false", "no", "0", ""):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def extract_json_object(text: str) -> str | None:
    text = text.strip()
    if not text:
        return None
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    if text.startswith("{") and text.endswith("}"):
        return text

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def parse_decision(text: str) -> dict | None:
    payload = extract_json_object(text)
    if not payload:
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    if "proactive_score" not in value and "should_interrupt" not in value:
        return None
    return value


class ProactivePolicy:
    def __init__(self, config: ProactiveConfig):
        self.config = config
        self.last_alert_at = 0.0
        self.last_message = ""

    def consider(self, decision: dict) -> str | None:
        score = number_or_default(decision.get("proactive_score"), 0.0)
        confidence = number_or_default(decision.get("confidence"), 0.0)
        should_interrupt = bool_or_default(decision.get("should_interrupt"), False)
        message = str(decision.get("message") or "").strip()
        if not should_interrupt or score < self.config.threshold or confidence < self.config.min_confidence or not message:
            return None

        now = time.monotonic()
        if now - self.last_alert_at < self.config.cooldown:
            return None
        if message == self.last_message:
            return None

        self.last_alert_at = now
        self.last_message = message
        return message


class ProactiveAgent:
    """Runs a separate visual decision loop so JSON decisions are never spoken by Live."""

    def __init__(self, api_key: str, config: ProactiveConfig, hub, long_term_goal: str | None = None):
        from google import genai

        self.client = genai.Client(api_key=api_key)
        self.config = config
        self.hub = hub
        self.long_term_goal = long_term_goal or "No fixed task. Help only when clearly useful."
        self.policy = ProactivePolicy(config)
        self.running = True

    def stop(self) -> None:
        self.running = False

    async def run(self, get_frame_jpeg: Callable[[], bytes | None]) -> None:
        self.hub.emit("session_event", text="Proactive ContextAgent enabled")
        while self.running:
            started = time.monotonic()
            jpeg = get_frame_jpeg()
            if jpeg is not None:
                try:
                    await self._tick(jpeg)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.hub.emit("error", text=f"proactive error: {exc!r}")
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.2, self.config.interval - elapsed))

    async def _tick(self, jpeg: bytes) -> None:
        from google.genai import types

        prompt = (
            f"{DECISION_PROMPT}\n\n"
            f"Long-term task or preference:\n{self.long_term_goal}\n\n"
            "Make one proactive decision from the attached first-person camera image."
        )
        response = await self.client.aio.models.generate_content(
            model=self.config.model,
            contents=[
                types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"),
                types.Part.from_text(text=prompt),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        decision = parse_decision(response.text or "")
        if not decision:
            self.hub.emit("error", text="proactive decision was not valid JSON")
            return

        context = str(decision.get("visual_context") or "")
        reason = str(decision.get("reason") or "")
        score = number_or_default(decision.get("proactive_score"), 0.0)
        confidence = number_or_default(decision.get("confidence"), 0.0)
        self.hub.emit(
            "proactive_decision",
            text=f"score={score:.1f} confidence={confidence:.2f} context={context} reason={reason}",
        )

        message = self.policy.consider(decision)
        if not message:
            return
        self.hub.emit("model_transcript", message)
        if self.config.say:
            speak(message)


def speak(message: str) -> None:
    if sys.platform != "darwin":
        return
    subprocess.Popen(["say", message], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
