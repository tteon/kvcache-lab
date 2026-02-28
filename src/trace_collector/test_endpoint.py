"""Verify GPU endpoint supports required capabilities before running collectors."""

import sys

from openai import OpenAI

from .common import LLM_API_BASE, LLM_API_KEY, LLM_MODEL


def _mask_secret(value: str) -> str:
    if not value:
        return "(not set)"
    if len(value) <= 8:
        return f"{value[:2]}***"
    return f"{value[:8]}...{value[-4:]}"


def check_chat_completions(client: OpenAI) -> bool:
    """Test basic chat completions."""
    print("[1/3] Testing chat completions...", end=" ")
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10,
        )
        text = response.choices[0].message.content
        print(f"OK  (response: {text!r})")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def check_tool_calling(client: OpenAI) -> bool:
    """Test tool/function calling (needed for mem0)."""
    print("[2/3] Testing tool calling...", end=" ")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_entities",
                "description": "Extract entities from text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity": {"type": "string"},
                                    "entity_type": {"type": "string"},
                                },
                                "required": ["entity", "entity_type"],
                            },
                        }
                    },
                    "required": ["entities"],
                },
            },
        }
    ]
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Marie Curie was a physicist."}],
            tools=tools,
            tool_choice="auto",
            max_tokens=200,
        )
        msg = response.choices[0].message
        has_tool_calls = msg.tool_calls is not None and len(msg.tool_calls) > 0
        has_content = msg.content is not None and len(msg.content) > 0
        if has_tool_calls:
            print(f"OK  (tool_calls: {msg.tool_calls[0].function.name})")
        elif has_content:
            print(f"WARN (no tool_calls, got content instead: {msg.content[:80]!r})")
        else:
            print("WARN (empty response)")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def check_json_mode(client: OpenAI) -> bool:
    """Test JSON response format (needed for graphiti)."""
    print("[3/3] Testing JSON mode...", end=" ")
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds in JSON format.",
                },
                {
                    "role": "user",
                    "content": 'Return a JSON object with key "status" and value "ok".',
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=50,
        )
        text = response.choices[0].message.content
        print(f"OK  (response: {text!r})")
        return True
    except Exception as e:
        print(f"FAIL ({e})")
        return False


def main():
    print(f"Endpoint: {LLM_API_BASE}")
    print(f"Model:    {LLM_MODEL}")
    print(f"API Key:  {_mask_secret(LLM_API_KEY)}")
    print()

    if not LLM_API_KEY:
        print("OPENAI_API_KEY/LLM_API_KEY is not configured.")
        sys.exit(1)

    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)

    results = [
        check_chat_completions(client),
        check_tool_calling(client),
        check_json_mode(client),
    ]

    print()
    if all(results):
        print("All checks passed. Endpoint is ready for trace collection.")
    else:
        failed = sum(1 for r in results if not r)
        print(f"{failed}/3 checks failed. Some collectors may not work.")
        sys.exit(1)


if __name__ == "__main__":
    main()
