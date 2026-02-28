"""Run prefix-analysis for dataset x baseline matrix traces."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .common import PROJECT_ROOT, TRACES_DIR
from .datasets import DATASET_CHOICES
from .run_matrix import BASELINE_CHOICES

ANALYSIS_SCRIPT = PROJECT_ROOT / "lmcache-agent-trace" / "prefix_analysis.py"
TOKENIZER = "meta-llama/Llama-3.1-8B"


def _trace_path(baseline: str, dataset: str) -> Path:
    key = f"{baseline}_{dataset}"
    return TRACES_DIR / key / f"{key}_session.jsonl"


def _result_paths(baseline: str, dataset: str) -> tuple[Path, Path]:
    key = f"{baseline}_{dataset}"
    result_dir = TRACES_DIR / f"{key}_result"
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{key}_hit_rate.png", result_dir / f"{key}_matches.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze matrix traces with prefix_analysis.py")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES + ["all"],
        default="all",
        help="Dataset to analyze (default: all)",
    )
    parser.add_argument(
        "--baseline",
        choices=BASELINE_CHOICES + ["all"],
        default="all",
        help="Baseline to analyze (default: all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = DATASET_CHOICES if args.dataset == "all" else [args.dataset]
    baselines = BASELINE_CHOICES if args.baseline == "all" else [args.baseline]

    print("=== Matrix Analysis ===")
    results: list[dict] = []
    for dataset in datasets:
        for baseline in baselines:
            trace_file = _trace_path(baseline, dataset)
            if not trace_file.exists():
                print(f"[{baseline} x {dataset}] missing trace: {trace_file}")
                results.append(
                    {"dataset": dataset, "baseline": baseline, "status": "missing", "error": ""}
                )
                continue

            line_count = sum(1 for _ in open(trace_file, encoding="utf-8"))
            if line_count == 0:
                print(f"[{baseline} x {dataset}] empty trace")
                results.append(
                    {"dataset": dataset, "baseline": baseline, "status": "empty", "error": ""}
                )
                continue

            output_png, match_jsonl = _result_paths(baseline, dataset)
            cmd = [
                sys.executable,
                str(ANALYSIS_SCRIPT),
                "-i",
                str(trace_file),
                "-o",
                str(output_png),
                "--log-matches",
                str(match_jsonl),
                "--tokenizer",
                TOKENIZER,
            ]
            print(f"[{baseline} x {dataset}] analyzing ({line_count} entries)...")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            except subprocess.TimeoutExpired:
                print(f"[{baseline} x {dataset}] timeout")
                results.append(
                    {"dataset": dataset, "baseline": baseline, "status": "timeout", "error": ""}
                )
                continue

            if proc.returncode == 0:
                print(f"[{baseline} x {dataset}] OK -> {output_png}")
                results.append(
                    {"dataset": dataset, "baseline": baseline, "status": "ok", "error": ""}
                )
            else:
                err = proc.stderr.strip().splitlines()
                tail = err[-1] if err else f"exit={proc.returncode}"
                print(f"[{baseline} x {dataset}] FAILED: {tail}")
                results.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "status": "error",
                        "error": tail,
                    }
                )

    print("\n=== Analysis Summary ===")
    for r in results:
        print(f"{r['dataset']:>14} | {r['baseline']:<11} | {r['status']}")

    bad_states = {"error", "timeout"}
    if any(r["status"] in bad_states for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
