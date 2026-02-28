"""Collect baseline traces from direct OpenAI chat completions (no memory scaffold)."""

from __future__ import annotations

import json
import logging
import time
from hashlib import sha1
from pathlib import Path
from typing import Any

from openai import OpenAI

from .common import LLM_API_BASE, LLM_API_KEY, LLM_MODEL, TEST_CORPUS, TRACES_DIR, TraceLogger
from .neo4j_metrics import BreakdownLogger

logger = logging.getLogger(__name__)


def _build_messages(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a factual extraction assistant. "
                "Return compact JSON with keys: summary, entities, relations."
            ),
        },
        {"role": "user", "content": text},
    ]


def _messages_to_input_text(messages: list[dict]) -> str:
    return "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages)


def collect(
    corpus: list[str] | None = None,
    output_path: str | Path | None = None,
    session_id: str = "openai_base",
    breakdown_path: str | Path | None = None,
    breakdown_context: dict[str, Any] | None = None,
) -> str:
    """Run baseline trace collection. Returns path to output JSONL."""
    if output_path is None:
        output_path = TRACES_DIR / "openai_base" / "openai_base_session.jsonl"
    output_path = Path(output_path)
    rows = corpus if corpus is not None else TEST_CORPUS

    client = OpenAI(base_url=LLM_API_BASE, api_key=LLM_API_KEY)
    b_logger = (
        BreakdownLogger(
            breakdown_path,
            session_id=session_id,
            collector="openai_base",
            **(breakdown_context or {}),
        )
        if breakdown_path is not None
        else None
    )

    try:
        with TraceLogger(output_path, session_id=session_id) as trace_logger:
            if b_logger is not None:
                b_logger.log_event("collector", "start", item_count=len(rows))
            print(f"[openai-base] Collecting traces for {len(rows)} items...")
            for i, text in enumerate(rows):
                print(f"  [{i + 1}/{len(rows)}] {text[:60]}...")
                messages = _build_messages(text)
                input_text = _messages_to_input_text(messages)
                started = time.monotonic()
                try:
                    response = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        temperature=0.0,
                        max_tokens=400,
                        response_format={"type": "json_object"},
                    )
                except Exception as e:
                    logger.warning(f"  openai_base call failed for item {i + 1}: {e}")
                    if b_logger is not None:
                        b_logger.log_event(
                            "openai",
                            "chat_completion",
                            status="error",
                            duration_ms=(time.monotonic() - started) * 1000.0,
                            step=i + 1,
                            prompt_size_chars=len(input_text),
                            error=str(e),
                        )
                    continue

                choice = response.choices[0]
                output_text = choice.message.content or ""
                if choice.message.tool_calls:
                    output_text = "\n".join(
                        json.dumps({"name": tc.function.name, "arguments": tc.function.arguments})
                        for tc in choice.message.tool_calls
                    )

                metadata = {
                    "model": getattr(response, "model", None),
                    "finish_reason": choice.finish_reason,
                }
                usage = getattr(response, "usage", None)
                if usage:
                    metadata["prompt_tokens"] = usage.prompt_tokens
                    metadata["completion_tokens"] = usage.completion_tokens
                    metadata["total_tokens"] = usage.total_tokens
                    details = getattr(usage, "prompt_tokens_details", None)
                    if details:
                        metadata["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0

                trace_logger.log(input_text, output_text, **metadata)
                if b_logger is not None:
                    b_logger.log_event(
                        "openai",
                        "chat_completion",
                        duration_ms=(time.monotonic() - started) * 1000.0,
                        step=i + 1,
                        prompt_size_chars=len(input_text),
                        prompt_hash=sha1(input_text.encode("utf-8")).hexdigest()[:12],
                        call_type=choice.finish_reason,
                        output_size_chars=len(output_text),
                        prompt_tokens=metadata.get("prompt_tokens"),
                        completion_tokens=metadata.get("completion_tokens"),
                        total_tokens=metadata.get("total_tokens"),
                        cached_tokens=metadata.get("cached_tokens"),
                        prompt_preview=input_text[:240],
                        prompt_text=input_text,
                    )
            if b_logger is not None:
                b_logger.log_event("collector", "finish")
    finally:
        if b_logger is not None:
            b_logger.close()

    print(f"[openai-base] Traces written to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    collect()
