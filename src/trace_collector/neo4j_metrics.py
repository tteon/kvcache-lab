"""Workload breakdown helpers for Neo4j-intensive collectors.

This module provides:
- JSONL event logger for workload spans and query metrics
- Runtime monkeypatching for neo4j Driver/Session query methods
- Lightweight database snapshot helper (indexes, counts, property-size estimates)
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterator


def _json_default(obj: Any) -> str:
    return str(obj)


def _estimate_size_bytes(payload: Any) -> int:
    try:
        return len(json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8"))
    except Exception:
        return len(str(payload).encode("utf-8"))


def _extract_query_text(query_obj: Any) -> str:
    if query_obj is None:
        return ""
    text = getattr(query_obj, "text", None)
    if isinstance(text, str):
        return text
    return str(query_obj)


def _normalize_query_text(query: str) -> str:
    return " ".join(query.strip().split())


def cypher_hash(query: str) -> str:
    normalized = _normalize_query_text(query)
    return sha1(normalized.encode("utf-8")).hexdigest()[:12]


def classify_cypher_query(query: str) -> str:
    q = _normalize_query_text(query).lower()
    if not q:
        return "unknown"
    if (
        "create index" in q
        or "drop index" in q
        or "show indexes" in q
        or "create constraint" in q
        or "drop constraint" in q
    ):
        return "indexing"
    if "vector.similarity" in q or "db.index.fulltext.query" in q or "vector.querynodes" in q:
        return "search"
    if q.startswith("merge") or q.startswith("create") or q.startswith("delete") or q.startswith("set"):
        return "write"
    if q.startswith("match") or q.startswith("with match") or " return " in q:
        return "read"
    return "other"


def _extract_counter(summary: Any, field: str) -> int:
    counters = getattr(summary, "counters", None)
    value = getattr(counters, field, 0) if counters is not None else 0
    return int(value or 0)


def _extract_records_and_summary(result: Any) -> tuple[list[Any], Any]:
    records = getattr(result, "records", None)
    summary = getattr(result, "summary", None)
    if records is not None:
        return list(records), summary

    # neo4j Driver.execute_query may also be unpacked like (records, summary, keys)
    if isinstance(result, tuple) and len(result) >= 2:
        raw_records = result[0]
        summary = result[1]
        try:
            return list(raw_records), summary
        except Exception:
            return [], summary
    return [], summary


def _records_to_size(records: list[Any]) -> int:
    rows: list[Any] = []
    for row in records:
        if hasattr(row, "data"):
            try:
                rows.append(row.data())
                continue
            except Exception:
                pass
        rows.append(str(row))
    return _estimate_size_bytes(rows)


def _summary_fields(summary: Any) -> dict[str, Any]:
    if summary is None:
        return {}

    fields: dict[str, Any] = {
        "result_available_after_ms": getattr(summary, "result_available_after", None),
        "result_consumed_after_ms": getattr(summary, "result_consumed_after", None),
    }
    db_obj = getattr(summary, "database", None)
    if db_obj is not None:
        fields["database"] = getattr(db_obj, "name", str(db_obj))

    plan = getattr(summary, "plan", None)
    if plan is not None:
        fields["plan_operator"] = getattr(plan, "operator_type", None)

    profile = getattr(summary, "profile", None)
    if profile is not None:
        fields["profile_operator"] = getattr(profile, "operator_type", None)

    fields["nodes_created"] = _extract_counter(summary, "nodes_created")
    fields["nodes_deleted"] = _extract_counter(summary, "nodes_deleted")
    fields["relationships_created"] = _extract_counter(summary, "relationships_created")
    fields["relationships_deleted"] = _extract_counter(summary, "relationships_deleted")
    fields["properties_set"] = _extract_counter(summary, "properties_set")
    fields["labels_added"] = _extract_counter(summary, "labels_added")
    fields["indexes_added"] = _extract_counter(summary, "indexes_added")
    fields["indexes_removed"] = _extract_counter(summary, "indexes_removed")
    return fields


class BreakdownLogger:
    """JSONL logger for workload breakdown events."""

    def __init__(self, output_path: str | Path, **context: Any):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.output_path, "w", encoding="utf-8")
        self.context = context

    def log_event(
        self,
        component: str,
        op: str,
        *,
        status: str = "ok",
        duration_ms: float | None = None,
        **fields: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "timestamp_us": int(time.time() * 1_000_000),
            "component": component,
            "op": op,
            "status": status,
            **self.context,
            **fields,
        }
        if duration_ms is not None:
            payload["duration_ms"] = round(duration_ms, 3)
        self._file.write(json.dumps(payload, default=_json_default) + "\n")
        self._file.flush()

    @contextmanager
    def span(self, component: str, op: str, **fields: Any) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        except Exception as exc:
            self.log_event(
                component,
                op,
                status="error",
                duration_ms=(time.monotonic() - started) * 1000.0,
                error=str(exc),
                **fields,
            )
            raise
        self.log_event(component, op, duration_ms=(time.monotonic() - started) * 1000.0, **fields)

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "BreakdownLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


@contextmanager
def patch_neo4j_calls(logger: BreakdownLogger | None) -> Iterator[None]:
    """Monkeypatch neo4j query methods to emit per-query breakdown events."""
    if logger is None:
        yield
        return

    from neo4j import AsyncDriver, AsyncSession, Driver, Session

    original_driver_execute_query = Driver.execute_query
    original_async_driver_execute_query = AsyncDriver.execute_query
    original_session_run = Session.run
    original_async_session_run = AsyncSession.run

    def _log_query_event(
        *,
        query_text: str,
        params: Any,
        duration_ms: float,
        status: str,
        error: str | None = None,
        result: Any = None,
    ) -> None:
        query_tag = classify_cypher_query(query_text)
        event: dict[str, Any] = {
            "query_hash": cypher_hash(query_text),
            "query_tag": query_tag,
            "query_text": query_text,
            "query_preview": _normalize_query_text(query_text)[:240],
            "params_size_bytes": _estimate_size_bytes(params or {}),
        }
        if result is not None:
            records, summary = _extract_records_and_summary(result)
            event["records_count"] = len(records)
            event["records_size_bytes"] = _records_to_size(records)
            event.update(_summary_fields(summary))
        if error:
            event["error"] = error
        logger.log_event(
            "neo4j",
            "cypher_query",
            status=status,
            duration_ms=duration_ms,
            **event,
        )

    def traced_driver_execute_query(self: Any, *args: Any, **kwargs: Any) -> Any:
        query_obj = kwargs.get("query_")
        if query_obj is None and args:
            query_obj = args[0]
        params = kwargs.get("parameters_")
        if params is None and len(args) > 1:
            params = args[1]
        query_text = _extract_query_text(query_obj)

        started = time.monotonic()
        try:
            result = original_driver_execute_query(self, *args, **kwargs)
        except Exception as exc:
            _log_query_event(
                query_text=query_text,
                params=params,
                duration_ms=(time.monotonic() - started) * 1000.0,
                status="error",
                error=str(exc),
            )
            raise

        _log_query_event(
            query_text=query_text,
            params=params,
            duration_ms=(time.monotonic() - started) * 1000.0,
            status="ok",
            result=result,
        )
        return result

    async def traced_async_driver_execute_query(self: Any, *args: Any, **kwargs: Any) -> Any:
        query_obj = kwargs.get("query_")
        if query_obj is None and args:
            query_obj = args[0]
        params = kwargs.get("parameters_")
        if params is None and len(args) > 1:
            params = args[1]
        query_text = _extract_query_text(query_obj)

        started = time.monotonic()
        try:
            result = await original_async_driver_execute_query(self, *args, **kwargs)
        except Exception as exc:
            _log_query_event(
                query_text=query_text,
                params=params,
                duration_ms=(time.monotonic() - started) * 1000.0,
                status="error",
                error=str(exc),
            )
            raise

        _log_query_event(
            query_text=query_text,
            params=params,
            duration_ms=(time.monotonic() - started) * 1000.0,
            status="ok",
            result=result,
        )
        return result

    def traced_session_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        query_obj = kwargs.get("query")
        if query_obj is None and args:
            query_obj = args[0]
        params = kwargs.get("parameters")
        if params is None and len(args) > 1:
            params = args[1]
        query_text = _extract_query_text(query_obj)

        started = time.monotonic()
        try:
            result = original_session_run(self, *args, **kwargs)
        except Exception as exc:
            _log_query_event(
                query_text=query_text,
                params=params,
                duration_ms=(time.monotonic() - started) * 1000.0,
                status="error",
                error=str(exc),
            )
            raise

        logger.log_event(
            "neo4j",
            "cypher_run",
            duration_ms=(time.monotonic() - started) * 1000.0,
            query_hash=cypher_hash(query_text),
            query_tag=classify_cypher_query(query_text),
            query_text=query_text,
            query_preview=_normalize_query_text(query_text)[:240],
            params_size_bytes=_estimate_size_bytes(params or {}),
            result_type=type(result).__name__,
        )
        return result

    async def traced_async_session_run(self: Any, *args: Any, **kwargs: Any) -> Any:
        query_obj = kwargs.get("query")
        if query_obj is None and args:
            query_obj = args[0]
        params = kwargs.get("parameters")
        if params is None and len(args) > 1:
            params = args[1]
        query_text = _extract_query_text(query_obj)

        started = time.monotonic()
        try:
            result = await original_async_session_run(self, *args, **kwargs)
        except Exception as exc:
            _log_query_event(
                query_text=query_text,
                params=params,
                duration_ms=(time.monotonic() - started) * 1000.0,
                status="error",
                error=str(exc),
            )
            raise

        logger.log_event(
            "neo4j",
            "cypher_run",
            duration_ms=(time.monotonic() - started) * 1000.0,
            query_hash=cypher_hash(query_text),
            query_tag=classify_cypher_query(query_text),
            query_text=query_text,
            query_preview=_normalize_query_text(query_text)[:240],
            params_size_bytes=_estimate_size_bytes(params or {}),
            result_type=type(result).__name__,
        )
        return result

    Driver.execute_query = traced_driver_execute_query
    AsyncDriver.execute_query = traced_async_driver_execute_query
    Session.run = traced_session_run
    AsyncSession.run = traced_async_session_run
    try:
        yield
    finally:
        Driver.execute_query = original_driver_execute_query
        AsyncDriver.execute_query = original_async_driver_execute_query
        Session.run = original_session_run
        AsyncSession.run = original_async_session_run


def capture_db_snapshot(
    logger: BreakdownLogger | None,
    *,
    uri: str,
    username: str,
    password: str,
    database: str,
    stage: str,
) -> None:
    """Capture coarse DB state for indexing/search/storage breakdown."""
    if logger is None:
        return

    from neo4j import GraphDatabase

    started = time.monotonic()
    try:
        with GraphDatabase.driver(uri, auth=(username, password)) as driver:
            idx_rows, _, _ = driver.execute_query(
                (
                    "SHOW INDEXES YIELD name, type, entityType, state, populationPercent, readCount "
                    "RETURN name, type, entityType, state, populationPercent, readCount"
                ),
                database_=database,
            )
            index_entries = [row.data() if hasattr(row, "data") else dict(row) for row in idx_rows]

            nodes, _, _ = driver.execute_query("MATCH (n) RETURN count(n) AS c", database_=database)
            rels, _, _ = driver.execute_query("MATCH ()-[r]->() RETURN count(r) AS c", database_=database)

            node_props, _, _ = driver.execute_query(
                (
                    "MATCH (n) UNWIND keys(n) AS k "
                    "RETURN count(*) AS prop_count, sum(size(toString(n[k]))) AS prop_chars"
                ),
                database_=database,
            )
            rel_props, _, _ = driver.execute_query(
                (
                    "MATCH ()-[r]->() UNWIND keys(r) AS k "
                    "RETURN count(*) AS prop_count, sum(size(toString(r[k]))) AS prop_chars"
                ),
                database_=database,
            )

            node_count = int(nodes[0]["c"]) if nodes else 0
            rel_count = int(rels[0]["c"]) if rels else 0
            node_prop_count = int(node_props[0]["prop_count"] or 0) if node_props else 0
            node_prop_chars = int(node_props[0]["prop_chars"] or 0) if node_props else 0
            rel_prop_count = int(rel_props[0]["prop_count"] or 0) if rel_props else 0
            rel_prop_chars = int(rel_props[0]["prop_chars"] or 0) if rel_props else 0

        logger.log_event(
            "neo4j",
            "db_snapshot",
            stage=stage,
            duration_ms=(time.monotonic() - started) * 1000.0,
            index_count=len(index_entries),
            online_indexes=sum(1 for row in index_entries if str(row.get("state", "")).upper() == "ONLINE"),
            building_indexes=sum(
                1 for row in index_entries if str(row.get("state", "")).upper() not in {"ONLINE", ""}
            ),
            index_entries=index_entries[:30],
            node_count=node_count,
            relationship_count=rel_count,
            node_property_count=node_prop_count,
            node_property_chars=node_prop_chars,
            relationship_property_count=rel_prop_count,
            relationship_property_chars=rel_prop_chars,
        )
    except Exception as exc:
        logger.log_event(
            "neo4j",
            "db_snapshot",
            status="error",
            stage=stage,
            duration_ms=(time.monotonic() - started) * 1000.0,
            error=str(exc),
        )
