"""Generate markdown report for dataset x baseline matrix runs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from .common import LLM_API_BASE, LLM_MODEL, NEO4J_URI, NEO4J_USERNAME, TRACES_DIR
from .datasets import DATASET_CHOICES, dataset_description
from .run_matrix import BASELINE_CHOICES


def _matches_path(baseline: str, dataset: str) -> Path:
    key = f"{baseline}_{dataset}"
    return TRACES_DIR / f"{key}_result" / f"{key}_matches.jsonl"


def _trace_path(baseline: str, dataset: str) -> Path:
    key = f"{baseline}_{dataset}"
    return TRACES_DIR / key / f"{key}_session.jsonl"


def _breakdown_path(baseline: str, dataset: str) -> Path:
    key = f"{baseline}_{dataset}"
    return TRACES_DIR / key / f"{key}_breakdown.jsonl"


def _legacy_trace_path(baseline: str, dataset: str) -> Path | None:
    if dataset == "corpus50" and baseline == "mem0":
        return TRACES_DIR / "mem0_graph" / "mem0_graph_session.jsonl"
    if dataset == "corpus50" and baseline == "graphiti":
        return TRACES_DIR / "graphiti_graph" / "graphiti_graph_session.jsonl"
    if baseline == "openai_base" and dataset == "tau2_airline":
        return TRACES_DIR / "tau2_airline" / "tau2_airline_session.jsonl"
    if baseline == "openai_base" and dataset == "tau2_retail":
        return TRACES_DIR / "tau2_retail" / "tau2_retail_session.jsonl"
    if baseline == "openai_base" and dataset == "tau2_telecom":
        return TRACES_DIR / "tau2_telecom" / "tau2_telecom_session.jsonl"
    return None


def _legacy_matches_path(baseline: str, dataset: str) -> Path | None:
    if dataset == "corpus50" and baseline == "mem0":
        return TRACES_DIR / "mem0_result" / "mem0_matches.jsonl"
    if dataset == "corpus50" and baseline == "graphiti":
        return TRACES_DIR / "graphiti_result" / "graphiti_matches.jsonl"
    if baseline == "openai_base" and dataset == "tau2_airline":
        return TRACES_DIR / "tau2_airline_result" / "tau2_airline_matches.jsonl"
    if baseline == "openai_base" and dataset == "tau2_retail":
        return TRACES_DIR / "tau2_retail_result" / "tau2_retail_matches.jsonl"
    if baseline == "openai_base" and dataset == "tau2_telecom":
        return TRACES_DIR / "tau2_telecom_result" / "tau2_telecom_matches.jsonl"
    return None


def _compute_rates(matches_path: Path) -> dict:
    total_input_tokens = 0
    total_prefix_matched = 0
    total_substring_matched = 0
    count = 0

    with open(matches_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            input_len = entry.get("InputLen", 0)
            matches = entry.get("Matches", [])
            count += 1
            total_input_tokens += input_len
            if input_len == 0 or not matches:
                continue

            matched_tokens = set()
            for m in matches:
                start = max(0, m["MatchStart"])
                end = min(input_len, m["MatchEnd"])
                for t in range(start, end):
                    matched_tokens.add(t)
            total_substring_matched += len(matched_tokens)

            sorted_ranges = sorted(
                [(max(0, m["MatchStart"]), min(input_len, m["MatchEnd"])) for m in matches],
                key=lambda x: x[0],
            )
            prefix_end = 0
            for start, end in sorted_ranges:
                if start <= prefix_end:
                    prefix_end = max(prefix_end, end)
                else:
                    break
            total_prefix_matched += prefix_end

    if total_input_tokens == 0 or count == 0:
        return {
            "count": count,
            "avg_tokens": 0.0,
            "prefix": 0.0,
            "substring": 0.0,
            "gap": 0.0,
        }

    prefix = total_prefix_matched / total_input_tokens
    substring = total_substring_matched / total_input_tokens
    return {
        "count": count,
        "avg_tokens": total_input_tokens / count,
        "prefix": prefix,
        "substring": substring,
        "gap": substring - prefix,
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = max(0.0, min(1.0, p)) * (len(sorted_values) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return float(sorted_values[lo])
    weight = rank - lo
    return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)


def _compute_breakdown_metrics(breakdown_path: Path) -> dict:
    event_count = 0
    durations: list[float] = []
    query_records: list[int] = []
    query_result_bytes: list[int] = []
    query_param_bytes: list[int] = []
    query_events = 0
    indexing_queries = 0
    search_queries = 0
    prompt_events = 0
    prompt_sizes: list[int] = []
    top_prompts: dict[str, dict] = {}
    top_queries: dict[str, dict] = {}
    snapshots: dict[str, dict] = {}

    with open(breakdown_path, encoding="utf-8") as f:
        for line in f:
            event_count += 1
            event = json.loads(line)
            op = event.get("op")
            component = event.get("component")
            if component == "openai" and op == "chat_completion":
                prompt_events += 1
                prompt_size = event.get("prompt_size_chars")
                if isinstance(prompt_size, (int, float)):
                    prompt_sizes.append(int(prompt_size))
                phash = str(event.get("prompt_hash", "unknown"))
                pentry = top_prompts.setdefault(
                    phash,
                    {
                        "prompt_hash": phash,
                        "prompt_preview": str(event.get("prompt_preview", "")),
                        "count": 0,
                    },
                )
                pentry["count"] += 1

            if component == "neo4j" and op in {"cypher_query", "cypher_run"}:
                query_events += 1
                dur = event.get("duration_ms")
                if isinstance(dur, (int, float)):
                    durations.append(float(dur))
                records_count = event.get("records_count")
                if isinstance(records_count, int):
                    query_records.append(records_count)
                result_bytes = event.get("records_size_bytes")
                if isinstance(result_bytes, int):
                    query_result_bytes.append(result_bytes)
                params_bytes = event.get("params_size_bytes")
                if isinstance(params_bytes, int):
                    query_param_bytes.append(params_bytes)

                query_tag = str(event.get("query_tag", "unknown"))
                if query_tag == "indexing":
                    indexing_queries += 1
                if query_tag == "search":
                    search_queries += 1

                qhash = str(event.get("query_hash", "unknown"))
                qentry = top_queries.setdefault(
                    qhash,
                    {
                        "query_hash": qhash,
                        "query_tag": query_tag,
                        "query_preview": str(event.get("query_preview", "")),
                        "count": 0,
                        "durations": [],
                    },
                )
                qentry["count"] += 1
                if isinstance(dur, (int, float)):
                    qentry["durations"].append(float(dur))

            if component == "neo4j" and op == "db_snapshot":
                stage = str(event.get("stage", "unknown"))
                snapshots[stage] = event

    top_items = sorted(top_queries.values(), key=lambda x: x["count"], reverse=True)[:3]
    top_queries_rows = []
    for item in top_items:
        top_queries_rows.append(
            {
                "query_hash": item["query_hash"],
                "query_tag": item["query_tag"],
                "count": item["count"],
                "avg_ms": mean(item["durations"]) if item["durations"] else 0.0,
                "query_preview": item["query_preview"],
            }
        )
    top_prompt_rows = sorted(top_prompts.values(), key=lambda x: x["count"], reverse=True)[:3]

    before = snapshots.get("before_collection", {})
    after = snapshots.get("after_collection", {})
    node_delta = int(after.get("node_count", 0) or 0) - int(before.get("node_count", 0) or 0)
    rel_delta = int(after.get("relationship_count", 0) or 0) - int(
        before.get("relationship_count", 0) or 0
    )
    node_prop_chars_delta = int(after.get("node_property_chars", 0) or 0) - int(
        before.get("node_property_chars", 0) or 0
    )
    rel_prop_chars_delta = int(after.get("relationship_property_chars", 0) or 0) - int(
        before.get("relationship_property_chars", 0) or 0
    )

    return {
        "status": "analyzed",
        "events": event_count,
        "neo4j_queries": query_events,
        "indexing_queries": indexing_queries,
        "search_queries": search_queries,
        "query_p50_ms": _percentile(durations, 0.5),
        "query_p95_ms": _percentile(durations, 0.95),
        "avg_records_per_query": mean(query_records) if query_records else 0.0,
        "avg_result_bytes_per_query": mean(query_result_bytes) if query_result_bytes else 0.0,
        "avg_params_bytes_per_query": mean(query_param_bytes) if query_param_bytes else 0.0,
        "prompt_calls": prompt_events,
        "avg_prompt_chars": mean(prompt_sizes) if prompt_sizes else 0.0,
        "node_delta": node_delta,
        "relationship_delta": rel_delta,
        "node_prop_chars_delta": node_prop_chars_delta,
        "relationship_prop_chars_delta": rel_prop_chars_delta,
        "index_online_after": int(after.get("online_indexes", 0) or 0),
        "index_building_after": int(after.get("building_indexes", 0) or 0),
        "top_queries": top_queries_rows,
        "top_prompts": top_prompt_rows,
    }


def _empty_breakdown(status: str = "not_collected") -> dict:
    return {
        "status": status,
        "events": 0,
        "neo4j_queries": 0,
        "indexing_queries": 0,
        "search_queries": 0,
        "query_p50_ms": 0.0,
        "query_p95_ms": 0.0,
        "avg_records_per_query": 0.0,
        "avg_result_bytes_per_query": 0.0,
        "avg_params_bytes_per_query": 0.0,
        "prompt_calls": 0,
        "avg_prompt_chars": 0.0,
        "node_delta": 0,
        "relationship_delta": 0,
        "node_prop_chars_delta": 0,
        "relationship_prop_chars_delta": 0,
        "index_online_after": 0,
        "index_building_after": 0,
        "top_queries": [],
        "top_prompts": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate markdown matrix report")
    parser.add_argument(
        "-o",
        "--output",
        default=str(Path("docs") / "matrix_breakdown.md"),
        help="Output markdown path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    breakdown_rows: list[dict] = []
    cypher_rows: list[dict] = []
    prompt_rows: list[dict] = []
    for dataset in DATASET_CHOICES:
        for baseline in BASELINE_CHOICES:
            trace_candidates = [_trace_path(baseline, dataset)]
            legacy_trace = _legacy_trace_path(baseline, dataset)
            if legacy_trace is not None:
                trace_candidates.append(legacy_trace)
            trace_file = next((p for p in trace_candidates if p.exists()), trace_candidates[0])

            matches_candidates = [_matches_path(baseline, dataset)]
            legacy_matches = _legacy_matches_path(baseline, dataset)
            if legacy_matches is not None:
                matches_candidates.append(legacy_matches)
            matches_file = next((p for p in matches_candidates if p.exists()), matches_candidates[0])
            breakdown_file = _breakdown_path(baseline, dataset)
            if breakdown_file.exists():
                b = _compute_breakdown_metrics(breakdown_file)
                breakdown_rows.append({"dataset": dataset, "baseline": baseline, **b})
                for q in b["top_queries"]:
                    cypher_rows.append({"dataset": dataset, "baseline": baseline, **q})
                for p in b["top_prompts"]:
                    prompt_rows.append({"dataset": dataset, "baseline": baseline, **p})
            else:
                breakdown_rows.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        **_empty_breakdown(),
                    }
                )

            if not trace_file.exists():
                rows.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "status": "not_collected",
                        "count": 0,
                        "avg_tokens": 0.0,
                        "prefix": 0.0,
                        "substring": 0.0,
                        "gap": 0.0,
                    }
                )
                continue
            if not matches_file.exists():
                rows.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "status": "collected_not_analyzed",
                        "count": 0,
                        "avg_tokens": 0.0,
                        "prefix": 0.0,
                        "substring": 0.0,
                        "gap": 0.0,
                    }
                )
                continue

            metrics = _compute_rates(matches_file)
            rows.append(
                {
                    "dataset": dataset,
                    "baseline": baseline,
                    "status": "analyzed",
                    **metrics,
                }
            )

    lines: list[str] = []
    lines.append("# Matrix Breakdown Report")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("- Datasets: `corpus50`, `tau2_airline`, `tau2_retail`, `tau2_telecom`, `taubench_legacy`")
    lines.append("- Baselines: `openai_base`, `mem0`, `graphiti`")
    lines.append("")
    lines.append("## Active Config")
    lines.append("")
    lines.append(f"- `LLM_API_BASE`: `{LLM_API_BASE}`")
    lines.append(f"- `LLM_MODEL`: `{LLM_MODEL}`")
    lines.append(f"- `NEO4J_URI`: `{NEO4J_URI}`")
    lines.append(f"- `NEO4J_USERNAME`: `{NEO4J_USERNAME}`")
    lines.append("- Tokenizer in analysis: `meta-llama/Llama-3.1-8B`")
    lines.append("")
    lines.append("## Dataset Notes")
    lines.append("")
    for dataset in DATASET_CHOICES:
        lines.append(f"- `{dataset}`: {dataset_description(dataset)}")
    lines.append("")
    lines.append("## Matrix Status")
    lines.append("")
    lines.append("| Dataset | Baseline | Status | Calls | Avg input tokens | Prefix | Substring | Gap |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")

    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["dataset"],
                    r["baseline"],
                    r["status"],
                    str(r["count"]),
                    f"{r['avg_tokens']:.1f}",
                    f"{r['prefix']*100:.2f}%",
                    f"{r['substring']*100:.2f}%",
                    f"{r['gap']*100:.2f}%",
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Workload Breakdown")
    lines.append("")
    lines.append(
        "| Dataset | Baseline | Status | Prompt calls | Avg prompt chars | Neo4j queries | Search | Indexing | p50 ms | p95 ms | Avg rec/query | Avg result bytes/query | Node delta | Rel delta | Online idx(after) | Building idx(after) |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in breakdown_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r["dataset"],
                    r["baseline"],
                    r["status"],
                    str(r["prompt_calls"]),
                    f"{r['avg_prompt_chars']:.1f}",
                    str(r["neo4j_queries"]),
                    str(r["search_queries"]),
                    str(r["indexing_queries"]),
                    f"{r['query_p50_ms']:.1f}",
                    f"{r['query_p95_ms']:.1f}",
                    f"{r['avg_records_per_query']:.2f}",
                    f"{r['avg_result_bytes_per_query']:.1f}",
                    str(r["node_delta"]),
                    str(r["relationship_delta"]),
                    str(r["index_online_after"]),
                    str(r["index_building_after"]),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Top Cypher Patterns")
    lines.append("")
    lines.append("| Dataset | Baseline | Query hash | Tag | Calls | Avg ms | Query preview |")
    lines.append("|---|---|---|---|---:|---:|---|")
    if not cypher_rows:
        lines.append("| - | - | - | - | 0 | 0.0 | no breakdown data |")
    else:
        for row in cypher_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["dataset"],
                        row["baseline"],
                        row["query_hash"],
                        row["query_tag"],
                        str(row["count"]),
                        f"{row['avg_ms']:.1f}",
                        row["query_preview"].replace("|", "\\|"),
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.append("## Top Prompt Patterns")
    lines.append("")
    lines.append("| Dataset | Baseline | Prompt hash | Calls | Prompt preview |")
    lines.append("|---|---|---|---:|---|")
    if not prompt_rows:
        lines.append("| - | - | - | 0 | no breakdown data |")
    else:
        for row in prompt_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["dataset"],
                        row["baseline"],
                        row["prompt_hash"],
                        str(row["count"]),
                        row["prompt_preview"].replace("|", "\\|"),
                    ]
                )
                + " |"
            )

    lines.append("")
    lines.append("## Interpretation Hints")
    lines.append("")
    lines.append("- High prefix + small gap: prompt prefixes are stable.")
    lines.append("- Low prefix + large gap: prompt blocks move, substring reuse dominates.")
    lines.append("- Low both: low cross-call reuse in prompt content.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to: {output_path}")


if __name__ == "__main__":
    main()
