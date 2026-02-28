"""Generate markdown report for dataset x baseline matrix runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import LLM_API_BASE, LLM_MODEL, NEO4J_URI, NEO4J_USERNAME, TRACES_DIR
from .datasets import DATASET_CHOICES, dataset_description
from .run_matrix import BASELINE_CHOICES


def _matches_path(baseline: str, dataset: str) -> Path:
    key = f"{baseline}_{dataset}"
    return TRACES_DIR / f"{key}_result" / f"{key}_matches.jsonl"


def _trace_path(baseline: str, dataset: str) -> Path:
    key = f"{baseline}_{dataset}"
    return TRACES_DIR / key / f"{key}_session.jsonl"


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
