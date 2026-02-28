import json

import pytest

from trace_collector import analyze_matrix, run_matrix
from trace_collector.matrix_report import _compute_breakdown_metrics, _compute_rates


def test_run_matrix_build_output_path(monkeypatch, tmp_path):
    monkeypatch.setattr(run_matrix, "TRACES_DIR", tmp_path)

    output = run_matrix._build_output_path("mem0", "corpus50")

    assert output == tmp_path / "mem0_corpus50" / "mem0_corpus50_session.jsonl"


def test_run_matrix_build_breakdown_path(monkeypatch, tmp_path):
    monkeypatch.setattr(run_matrix, "TRACES_DIR", tmp_path)

    output = run_matrix._build_breakdown_path("mem0", "corpus50")

    assert output == tmp_path / "mem0_corpus50" / "mem0_corpus50_breakdown.jsonl"


def test_analyze_matrix_trace_and_result_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(analyze_matrix, "TRACES_DIR", tmp_path)

    trace_path = analyze_matrix._trace_path("graphiti", "tau2_airline")
    png_path, matches_path = analyze_matrix._result_paths("graphiti", "tau2_airline")

    assert trace_path == tmp_path / "graphiti_tau2_airline" / "graphiti_tau2_airline_session.jsonl"
    assert png_path == (
        tmp_path / "graphiti_tau2_airline_result" / "graphiti_tau2_airline_hit_rate.png"
    )
    assert matches_path == (
        tmp_path / "graphiti_tau2_airline_result" / "graphiti_tau2_airline_matches.jsonl"
    )
    assert png_path.parent.exists()


def test_compute_rates_handles_overlap_and_prefix(tmp_path):
    matches_path = tmp_path / "matches.jsonl"
    entries = [
        {
            "InputLen": 10,
            "Matches": [
                {"MatchStart": 0, "MatchEnd": 3},
                {"MatchStart": 2, "MatchEnd": 5},
                {"MatchStart": 7, "MatchEnd": 9},
            ],
        },
        {"InputLen": 8, "Matches": [{"MatchStart": 2, "MatchEnd": 4}]},
    ]
    matches_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

    rates = _compute_rates(matches_path)

    assert rates["count"] == 2
    assert rates["avg_tokens"] == pytest.approx(9.0)
    assert rates["prefix"] == pytest.approx(5 / 18)
    assert rates["substring"] == pytest.approx(9 / 18)
    assert rates["gap"] == pytest.approx(4 / 18)


def test_compute_rates_zero_tokens(tmp_path):
    matches_path = tmp_path / "matches.jsonl"
    matches_path.write_text(
        json.dumps({"InputLen": 0, "Matches": [{"MatchStart": 0, "MatchEnd": 3}]}) + "\n",
        encoding="utf-8",
    )

    rates = _compute_rates(matches_path)

    assert rates["count"] == 1
    assert rates["avg_tokens"] == 0.0
    assert rates["prefix"] == 0.0
    assert rates["substring"] == 0.0
    assert rates["gap"] == 0.0


def test_compute_breakdown_metrics(tmp_path):
    breakdown_path = tmp_path / "breakdown.jsonl"
    events = [
        {
            "component": "openai",
            "op": "chat_completion",
            "prompt_hash": "p1",
            "prompt_preview": "system: test",
            "prompt_size_chars": 200,
        },
        {
            "component": "neo4j",
            "op": "cypher_query",
            "duration_ms": 10.0,
            "query_hash": "q1",
            "query_tag": "search",
            "query_preview": "MATCH ...",
            "records_count": 2,
            "records_size_bytes": 120,
            "params_size_bytes": 30,
        },
        {
            "component": "neo4j",
            "op": "cypher_query",
            "duration_ms": 30.0,
            "query_hash": "q1",
            "query_tag": "search",
            "query_preview": "MATCH ...",
            "records_count": 4,
            "records_size_bytes": 220,
            "params_size_bytes": 40,
        },
        {
            "component": "neo4j",
            "op": "db_snapshot",
            "stage": "before_collection",
            "node_count": 5,
            "relationship_count": 4,
            "node_property_chars": 100,
            "relationship_property_chars": 50,
            "online_indexes": 1,
            "building_indexes": 0,
        },
        {
            "component": "neo4j",
            "op": "db_snapshot",
            "stage": "after_collection",
            "node_count": 9,
            "relationship_count": 11,
            "node_property_chars": 140,
            "relationship_property_chars": 90,
            "online_indexes": 2,
            "building_indexes": 0,
        },
    ]
    breakdown_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

    metrics = _compute_breakdown_metrics(breakdown_path)

    assert metrics["events"] == 5
    assert metrics["prompt_calls"] == 1
    assert metrics["neo4j_queries"] == 2
    assert metrics["search_queries"] == 2
    assert metrics["query_p50_ms"] == pytest.approx(20.0)
    assert metrics["query_p95_ms"] == pytest.approx(29.0)
    assert metrics["avg_records_per_query"] == pytest.approx(3.0)
    assert metrics["avg_result_bytes_per_query"] == pytest.approx(170.0)
    assert metrics["node_delta"] == 4
    assert metrics["relationship_delta"] == 7
    assert metrics["index_online_after"] == 2
