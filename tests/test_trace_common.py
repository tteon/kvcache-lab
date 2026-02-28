import json

from trace_collector.common import TraceLogger, messages_to_input_text


def test_messages_to_input_text_flattens_multimodal_content():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First part."},
                {"type": "image_url", "image_url": "ignored"},
                {"type": "text", "text": "Second part."},
            ],
        },
    ]

    rendered = messages_to_input_text(messages)

    assert "system: You are helpful." in rendered
    assert "user: First part. Second part." in rendered
    assert "image_url" not in rendered


def test_trace_logger_writes_jsonl_entries(tmp_path):
    output_file = tmp_path / "session.jsonl"

    with TraceLogger(output_file, session_id="unit-test") as logger:
        logger.log("user: hello", "assistant: hi", model="gpt-test", prompt_tokens=4)

    lines = output_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["session_id"] == "unit-test"
    assert payload["input"] == "user: hello"
    assert payload["output"] == "assistant: hi"
    assert payload["model"] == "gpt-test"
    assert payload["prompt_tokens"] == 4
