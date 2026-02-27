"""Collect LLM call traces from mem0 graph memory system.

mem0 makes exactly 3 LLM calls per add():
  1. Entity extraction (EXTRACT_ENTITIES_TOOL)
  2. Relation extraction (RELATIONS_TOOL)
  3. Delete decision (DELETE_MEMORY_TOOL_GRAPH)

Interception: Uses built-in response_callback in OpenAIConfig.
"""

import json
import logging

from mem0 import Memory

from .common import (
    LLM_MODEL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    OPENAI_API_KEY,
    TEST_CORPUS,
    TRACES_DIR,
    TraceLogger,
    messages_to_input_text,
)

logger = logging.getLogger(__name__)

MEM0_DB = "mem0store"


def _make_callback(trace_logger: TraceLogger):
    """Create a response_callback closure that logs to trace_logger."""

    def callback(llm_instance, response, params):
        messages = params.get("messages", [])
        input_text = messages_to_input_text(messages)

        # Extract output text from response
        output_text = ""
        choice = response.choices[0].message
        call_type = None
        if choice.tool_calls:
            # Tool calling mode: serialize tool call arguments
            tool_outputs = []
            for tc in choice.tool_calls:
                tool_outputs.append(
                    json.dumps({"name": tc.function.name, "arguments": tc.function.arguments})
                )
            output_text = "\n".join(tool_outputs)
            call_type = choice.tool_calls[0].function.name
        elif choice.content:
            output_text = choice.content

        # Extract metadata from response
        metadata = {
            "model": getattr(response, "model", None),
            "finish_reason": response.choices[0].finish_reason,
        }
        if call_type:
            metadata["call_type"] = call_type

        usage = getattr(response, "usage", None)
        if usage:
            metadata["prompt_tokens"] = usage.prompt_tokens
            metadata["completion_tokens"] = usage.completion_tokens
            metadata["total_tokens"] = usage.total_tokens
            # OpenAI cached_tokens (server-side prefix cache)
            details = getattr(usage, "prompt_tokens_details", None)
            if details:
                metadata["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0

        trace_logger.log(input_text, output_text, **metadata)

    return callback


def collect(user_id: str = "trace_user") -> str:
    """Run mem0 trace collection. Returns path to output JSONL."""
    output_path = TRACES_DIR / "mem0_graph" / "mem0_graph_session.jsonl"

    with TraceLogger(output_path, session_id="mem0_graph") as trace_logger:
        callback = _make_callback(trace_logger)

        llm_config = {
            "provider": "openai",
            "config": {
                "model": LLM_MODEL,
                "api_key": OPENAI_API_KEY,
                "response_callback": callback,
            },
        }

        config = {
            "llm": llm_config,
            "graph_store": {
                "provider": "neo4j",
                "config": {
                    "url": NEO4J_URI,
                    "username": NEO4J_USERNAME,
                    "password": NEO4J_PASSWORD,
                    "database": MEM0_DB,
                },
                "llm": llm_config,
            },
            "embedder": {
                "provider": "huggingface",
                "config": {
                    "model": "multi-qa-MiniLM-L6-cos-v1",
                    "embedding_dims": 384,
                },
            },
            "version": "v1.1",
        }

        m = Memory.from_config(config)

        print(f"[mem0] Collecting traces for {len(TEST_CORPUS)} items...")
        for i, text in enumerate(TEST_CORPUS):
            print(f"  [{i + 1}/{len(TEST_CORPUS)}] {text[:60]}...")
            try:
                m.add(text, user_id=user_id)
            except Exception as e:
                logger.warning(f"  mem0 add() failed for item {i + 1}: {e}")

    print(f"[mem0] Traces written to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    collect()
