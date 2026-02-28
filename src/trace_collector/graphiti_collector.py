"""Collect LLM call traces from graphiti graph memory system.

graphiti makes 5-12 LLM calls per add_episode() depending on entity/edge count:
  - Node extraction, dedup, edge extraction, edge dedup, attributes, summaries

Interception: Subclass OpenAIGenericClient, override _generate_response to log traces.
Uses OpenAIGenericClient (not OpenAIClient) to avoid responses.parse() which custom
endpoints don't support.
"""

import asyncio
import json
import logging
import time
import typing
from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any

import openai

from graphiti_core import Graphiti
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.errors import RateLimitError
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.prompts.models import Message
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

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
)
from .neo4j_metrics import BreakdownLogger, capture_db_snapshot, patch_neo4j_calls

logger = logging.getLogger(__name__)

GRAPHITI_DB = "graphitistore"

EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"


class LocalEmbedder(EmbedderClient):
    """Local HuggingFace sentence-transformers embedder (no API key needed)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        if isinstance(input_data, str):
            return self.model.encode(input_data).tolist()
        if isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            return self.model.encode(input_data[0]).tolist()
        return self.model.encode(str(input_data)).tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(input_data_list)
        return [e.tolist() for e in embeddings]


class TracingOpenAIGenericClient(OpenAIGenericClient):
    """OpenAIGenericClient subclass that logs all LLM calls to a TraceLogger."""

    def __init__(
        self,
        config: LLMConfig,
        trace_logger: TraceLogger,
        breakdown_logger: BreakdownLogger | None = None,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        self.trace_logger = trace_logger
        self.breakdown_logger = breakdown_logger

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int = 16384,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        # Build input text from messages (before _clean_input modifies them)
        input_parts = []
        for m in messages:
            input_parts.append(f"{m.role}: {m.content}")
        input_text = "\n".join(input_parts)

        # Replicate parent logic to access raw response for metadata.
        # Parent (OpenAIGenericClient._generate_response) is ~15 lines:
        # clean input -> build openai_messages -> set response_format -> API call -> json.loads
        openai_messages: list[dict[str, Any]] = []
        for m in messages:
            m.content = self._clean_input(m.content)
            if m.role == "user":
                openai_messages.append({"role": "user", "content": m.content})
            elif m.role == "system":
                openai_messages.append({"role": "system", "content": m.content})

        response_format: dict[str, Any] = {"type": "json_object"}
        if response_model is not None:
            schema_name = getattr(response_model, "__name__", "structured_response")
            json_schema = response_model.model_json_schema()
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": json_schema},
            }

        try:
            t0 = time.monotonic()
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format=response_format,
            )
            latency_ms = round((time.monotonic() - t0) * 1000)

            result_text = response.choices[0].message.content or ""
            result = json.loads(result_text)
        except openai.RateLimitError as e:
            raise RateLimitError from e
        except Exception as e:
            logger.error(f"Error in generating LLM response: {e}")
            raise

        # Build metadata from raw response
        metadata: dict[str, Any] = {
            "model": getattr(response, "model", None),
            "finish_reason": response.choices[0].finish_reason,
            "latency_ms": latency_ms,
        }
        usage = getattr(response, "usage", None)
        if usage:
            metadata["prompt_tokens"] = usage.prompt_tokens
            metadata["completion_tokens"] = usage.completion_tokens
            metadata["total_tokens"] = usage.total_tokens
            details = getattr(usage, "prompt_tokens_details", None)
            if details:
                metadata["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0

        output_text = json.dumps(result) if result else ""
        self.trace_logger.log(input_text, output_text, **metadata)
        if self.breakdown_logger is not None:
            self.breakdown_logger.log_event(
                "openai",
                "chat_completion",
                duration_ms=latency_ms,
                prompt_hash=sha1(input_text.encode("utf-8")).hexdigest()[:12],
                prompt_preview=input_text[:240],
                prompt_size_chars=len(input_text),
                output_size_chars=len(output_text),
                prompt_tokens=metadata.get("prompt_tokens"),
                completion_tokens=metadata.get("completion_tokens"),
                total_tokens=metadata.get("total_tokens"),
                cached_tokens=metadata.get("cached_tokens"),
                prompt_text=input_text,
            )

        return result


async def _collect_async(
    user_id: str = "trace_user",
    corpus: list[str] | None = None,
    output_path: str | Path | None = None,
    session_id: str = "graphiti_graph",
    database: str = GRAPHITI_DB,
    group_id: str | None = None,
    breakdown_path: str | Path | None = None,
    breakdown_context: dict[str, Any] | None = None,
) -> str:
    """Async implementation of graphiti trace collection."""
    if output_path is None:
        output_path = TRACES_DIR / "graphiti_graph" / "graphiti_graph_session.jsonl"
    output_path = Path(output_path)
    rows = corpus if corpus is not None else TEST_CORPUS
    group = group_id if group_id is not None else database
    b_logger = (
        BreakdownLogger(
            breakdown_path,
            session_id=session_id,
            collector="graphiti",
            **(breakdown_context or {}),
        )
        if breakdown_path is not None
        else None
    )

    try:
        with TraceLogger(output_path, session_id=session_id) as trace_logger:
            llm_config = LLMConfig(
                api_key=LLM_API_KEY,
                base_url=LLM_API_BASE,
                model=LLM_MODEL,
                small_model=LLM_MODEL,
            )

            llm_client = TracingOpenAIGenericClient(
                config=llm_config,
                trace_logger=trace_logger,
                breakdown_logger=b_logger,
            )

            driver = Neo4jDriver(
                uri=NEO4J_URI,
                user=NEO4J_USERNAME,
                password=NEO4J_PASSWORD,
                database=database,
            )

            embedder = LocalEmbedder()

            graphiti = Graphiti(
                uri=None,
                user=None,
                password=None,
                llm_client=llm_client,
                graph_driver=driver,
                embedder=embedder,
            )

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

            try:
                with patch_neo4j_calls(b_logger):
                    build_started = time.monotonic()
                    try:
                        await graphiti.build_indices_and_constraints()
                        if b_logger is not None:
                            b_logger.log_event(
                                "graphiti",
                                "build_indices",
                                duration_ms=(time.monotonic() - build_started) * 1000.0,
                            )
                    except Exception as e:
                        logger.warning(f"  Failed to build indices (may already exist): {e}")
                        if b_logger is not None:
                            b_logger.log_event(
                                "graphiti",
                                "build_indices",
                                status="error",
                                duration_ms=(time.monotonic() - build_started) * 1000.0,
                                error=str(e),
                            )

                    print(f"[graphiti] Collecting traces for {len(rows)} items...")
                    for i, text in enumerate(rows):
                        print(f"  [{i + 1}/{len(rows)}] {text[:60]}...")
                        started = time.monotonic()
                        try:
                            await graphiti.add_episode(
                                name=f"fact_{i + 1}",
                                episode_body=text,
                                source_description="trace_collection_corpus",
                                reference_time=datetime.now(timezone.utc),
                                group_id=group,
                            )
                        except Exception as e:
                            logger.warning(f"  graphiti add_episode() failed for item {i + 1}: {e}")
                            if b_logger is not None:
                                b_logger.log_event(
                                    "graphiti",
                                    "add_episode",
                                    status="error",
                                    duration_ms=(time.monotonic() - started) * 1000.0,
                                    step=i + 1,
                                    input_size_chars=len(text),
                                    error=str(e),
                                )
                            continue
                        if b_logger is not None:
                            b_logger.log_event(
                                "graphiti",
                                "add_episode",
                                duration_ms=(time.monotonic() - started) * 1000.0,
                                step=i + 1,
                                input_size_chars=len(text),
                            )
            except Exception as e:
                if b_logger is not None:
                    b_logger.log_event("graphiti", "collection", status="error", error=str(e))
                raise
            finally:
                await graphiti.close()

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

    print(f"[graphiti] Traces written to {output_path}")
    return str(output_path)


def collect(
    user_id: str = "trace_user",
    corpus: list[str] | None = None,
    output_path: str | Path | None = None,
    session_id: str = "graphiti_graph",
    database: str = GRAPHITI_DB,
    group_id: str | None = None,
    breakdown_path: str | Path | None = None,
    breakdown_context: dict[str, Any] | None = None,
) -> str:
    """Run graphiti trace collection. Returns path to output JSONL."""
    return asyncio.run(
        _collect_async(
            user_id=user_id,
            corpus=corpus,
            output_path=output_path,
            session_id=session_id,
            database=database,
            group_id=group_id,
            breakdown_path=breakdown_path,
            breakdown_context=breakdown_context,
        )
    )


if __name__ == "__main__":
    collect()
