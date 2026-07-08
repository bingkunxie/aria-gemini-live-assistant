import time

from assistant.proactive import ProactiveConfig, ProactivePolicy, extract_json_object, parse_decision


def test_extract_json_from_markdown_block():
    text = '```json\n{"proactive_score": 4, "should_interrupt": true}\n```'
    assert extract_json_object(text) == '{"proactive_score": 4, "should_interrupt": true}'


def test_parse_decision_inside_extra_text():
    decision = parse_decision('note {"proactive_score": 3, "should_interrupt": false} tail')
    assert decision["proactive_score"] == 3
    assert decision["should_interrupt"] is False


def test_policy_alerts_when_over_threshold():
    policy = ProactivePolicy(ProactiveConfig(threshold=3, min_confidence=0.5, cooldown=1))
    message = policy.consider({
        "proactive_score": 4,
        "confidence": 0.8,
        "should_interrupt": True,
        "message": "Move the pen slightly left.",
    })
    assert message == "Move the pen slightly left."


def test_policy_suppresses_low_confidence():
    policy = ProactivePolicy(ProactiveConfig(threshold=3, min_confidence=0.7))
    assert policy.consider({
        "proactive_score": 5,
        "confidence": 0.4,
        "should_interrupt": True,
        "message": "Careful.",
    }) is None


def test_policy_cooldown_and_duplicate_suppression():
    policy = ProactivePolicy(ProactiveConfig(threshold=3, min_confidence=0.5, cooldown=60))
    decision = {
        "proactive_score": 4,
        "confidence": 0.8,
        "should_interrupt": True,
        "message": "Turn right.",
    }
    assert policy.consider(decision) == "Turn right."
    assert policy.consider({**decision, "message": "Turn left."}) is None

    policy.last_alert_at = time.monotonic() - 61
    assert policy.consider(decision) is None
