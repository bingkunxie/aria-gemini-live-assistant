import json

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
    ev = json.loads(rows[0])
    assert ev["type"] == "user_transcript" and ev["text"] == "hello"
    assert "t_wall" in ev and "t_session" in ev


def test_transcript_txt(tmp_path):
    hub = make_hub(tmp_path)
    hub.emit("user_transcript", "hi")
    hub.emit("model_transcript", "Hello!")
    hub.save_txt(tmp_path / "transcript.txt")
    assert (tmp_path / "transcript.txt").read_text() == "You: hi\nAI: Hello!\n"
