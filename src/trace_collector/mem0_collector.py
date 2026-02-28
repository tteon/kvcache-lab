"""Collect LLM call traces from mem0 graph memory system.

mem0 makes exactly 3 LLM calls per add():
  1. Entity extraction (EXTRACT_ENTITIES_TOOL)
  2. Relation extraction (RELATIONS_TOOL)
  3. Delete decision (DELETE_MEMORY_TOOL_GRAPH)

Interception: Uses built-in response_callback in OpenAIConfig.
"""

import json
import logging
import time
from hashlib import sha1
from pathlib import Path
from typing import Any

from mem0 import Memory

from .common import (
    LLM_API_BASE,
    LLM_API_KEY,
    LLM_MODEL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    TEST_CORPUS,
    TRACES_DIR,
    TraceLogger,
    messages_to_input_text,
)
from .neo4j_metrics import BreakdownLogger, capture_db_snapshot, patch_neo4j_calls

logger = logging.getLogger(__name__)

MEM0_DB = "mem0store"


def _make_callback(trace_logger: TraceLogger, breakdown_logger: BreakdownLogger | None = None):
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
        if breakdown_logger is not None:
            breakdown_logger.log_event(
                "openai",
                "chat_completion",
                prompt_hash=sha1(input_text.encode("utf-8")).hexdigest()[:12],
                prompt_preview=input_text[:240],
                prompt_size_chars=len(input_text),
                output_size_chars=len(output_text),
                call_type=metadata.get("call_type"),
                prompt_tokens=metadata.get("prompt_tokens"),
                completion_tokens=metadata.get("completion_tokens"),
                total_tokens=metadata.get("total_tokens"),
                cached_tokens=metadata.get("cached_tokens"),
                prompt_text=input_text,
            )

    return callback


def collect(
    user_id: str = "trace_user",
    corpus: list[str] | None = None,
    output_path: str | Path | None = None,
    session_id: str = "mem0_graph",
    database: str = MEM0_DB,
    breakdown_path: str | Path | None = None,
    breakdown_context: dict[str, Any] | None = None,
) -> str:
    """Run mem0 trace collection. Returns path to output JSONL."""
    if output_path is None:
        output_path = TRACES_DIR / "mem0_graph" / "mem0_graph_session.jsonl"
    output_path = Path(output_path)
    rows = corpus if corpus is not None else TEST_CORPUS
    b_logger = (
        BreakdownLogger(
            breakdown_path,
            session_id=session_id,
            collector="mem0",
            **(breakdown_context or {}),
        )
        if breakdown_path is not None
        else None
    )

    try:
        with TraceLogger(output_path, session_id=session_id) as trace_logger:
            callback = _make_callback(trace_logger, b_logger)

            llm_config = {
                "provider": "openai",
                "config": {
                    "model": LLM_MODEL,
                    "api_key": LLM_API_KEY,
                    "openai_base_url": LLM_API_BASE,
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
                        "database": database,
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
            if b_logger is not None:
                b_logger.log_event("collector", "start", item_count=len(rows), neo4j_database=database)
                capture_db_snapshot(
                    b_logger,
                    uri=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    database=database,
                    stage="before_collection",
                )

            print(f"[mem0] Collecting traces for {len(rows)} items...")
            with patch_neo4j_calls(b_logger):
                for i, text in enumerate(rows):
                    print(f"  [{i + 1}/{len(rows)}] {text[:60]}...")
                    started = time.monotonic()
                    try:
                        m.add(text, user_id=user_id)
                    except Exception as e:
                        logger.warning(f"  mem0 add() failed for item {i + 1}: {e}")
                        if b_logger is not None:
                            b_logger.log_event(
                                "mem0",
                                "add",
                                status="error",
                                duration_ms=(time.monotonic() - started) * 1000.0,
                                step=i + 1,
                                input_size_chars=len(text),
                                error=str(e),
                            )
                        continue
                    if b_logger is not None:
                        b_logger.log_event(
                            "mem0",
                            "add",
                            duration_ms=(time.monotonic() - started) * 1000.0,
                            step=i + 1,
                            input_size_chars=len(text),
                        )

            if b_logger is not None:
                capture_db_snapshot(
                    b_logger,
                    uri=NEO4J_URI,
                    username=NEO4J_USERNAME,
                    password=NEO4J_PASSWORD,
                    database=database,
                    stage="after_collection",
                )
                b_logger.log_event("collector", "finish")
    finally:
        if b_logger is not None:
            b_logger.close()

    print(f"[mem0] Traces written to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    collect()
