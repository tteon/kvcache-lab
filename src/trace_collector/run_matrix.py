"""Run dataset x baseline matrix experiments for trace collection."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .common import TRACES_DIR
from .datasets import DATASET_CHOICES, dataset_description, load_dataset

BASELINE_CHOICES = ["openai_base", "mem0", "graphiti"]


def _build_output_path(baseline: str, dataset: str) -> Path:
    subdir = f"{baseline}_{dataset}"
    return TRACES_DIR / subdir / f"{subdir}_session.jsonl"


def _build_breakdown_path(baseline: str, dataset: str) -> Path:
    subdir = f"{baseline}_{dataset}"
    return TRACES_DIR / subdir / f"{subdir}_breakdown.jsonl"


def _run_openai_base(
    dataset: str,
    rows: list[str],
    output_path: Path,
    breakdown_path: Path | None = None,
    breakdown_context: dict | None = None,
) -> str:
    from .openai_base_collector import collect

    return collect(
        corpus=rows,
        output_path=output_path,
        session_id=f"openai_base_{dataset}",
        breakdown_path=breakdown_path,
        breakdown_context=breakdown_context,
    )


def _run_mem0(
    dataset: str,
    rows: list[str],
    output_path: Path,
    breakdown_path: Path | None = None,
    breakdown_context: dict | None = None,
) -> str:
    from .mem0_collector import collect

    return collect(
        user_id=f"trace_{dataset}",
        corpus=rows,
        output_path=output_path,
        session_id=f"mem0_{dataset}",
        breakdown_path=breakdown_path,
        breakdown_context=breakdown_context,
    )


def _run_graphiti(
    dataset: str,
    rows: list[str],
    output_path: Path,
    breakdown_path: Path | None = None,
    breakdown_context: dict | None = None,
) -> str:
    from .graphiti_collector import collect

    return collect(
        user_id=f"trace_{dataset}",
        corpus=rows,
        output_path=output_path,
        session_id=f"graphiti_{dataset}",
        group_id=f"graphiti_{dataset}",
        breakdown_path=breakdown_path,
        breakdown_context=breakdown_context,
    )


RUNNERS = {
    "openai_base": _run_openai_base,
    "mem0": _run_mem0,
    "graphiti": _run_graphiti,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run matrix experiments across datasets and baselines")
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES + ["all"],
        default="all",
        help="Dataset to run (default: all)",
    )
    parser.add_argument(
        "--baseline",
        choices=BASELINE_CHOICES + ["all"],
        default="all",
        help="Baseline to run (default: all)",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=None,
        help="Optional cap on number of dataset rows",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a matrix cell if output trace already exists",
    )
    parser.add_argument(
        "--with-breakdown",
        action="store_true",
        help="Collect workload breakdown (prompts, Cypher, indexing/search/storage snapshots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets = DATASET_CHOICES if args.dataset == "all" else [args.dataset]
    baselines = BASELINE_CHOICES if args.baseline == "all" else [args.baseline]

    print("=== Trace Matrix Collection ===")
    print(f"datasets:  {', '.join(datasets)}")
    print(f"baselines: {', '.join(baselines)}")
    if args.num_items is not None:
        print(f"num_items: {args.num_items}")
    if args.with_breakdown:
        print("workload_breakdown: enabled")
    print()

    results: list[dict] = []
    run_id = f"matrix_{int(time.time())}"
    for dataset in datasets:
        try:
            rows = load_dataset(dataset, num_items=args.num_items)
        except Exception as e:
            for baseline in baselines:
                results.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "status": "error",
                        "error": f"dataset load failed: {e}",
                        "path": "",
                        "time": 0.0,
                        "rows": 0,
                    }
                )
            continue

        print(f"[dataset] {dataset} ({len(rows)} rows)")
        print(f"  - {dataset_description(dataset)}")
        for baseline in baselines:
            output_path = _build_output_path(baseline, dataset)
            breakdown_path = _build_breakdown_path(baseline, dataset) if args.with_breakdown else None
            breakdown_context = (
                {
                    "run_id": f"{run_id}:{baseline}_{dataset}",
                    "dataset": dataset,
                    "baseline": baseline,
                    "matrix_key": f"{baseline}_{dataset}",
                }
                if args.with_breakdown
                else None
            )
            if args.skip_existing and output_path.exists():
                print(f"  [{baseline}] SKIP (exists): {output_path}")
                results.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "status": "skipped",
                        "error": "",
                        "path": str(output_path),
                        "time": 0.0,
                        "rows": len(rows),
                        "breakdown_path": str(breakdown_path) if breakdown_path else "",
                    }
                )
                continue

            print(f"  [{baseline}] running...")
            t0 = time.monotonic()
            try:
                path = RUNNERS[baseline](
                    dataset,
                    rows,
                    output_path,
                    breakdown_path=breakdown_path,
                    breakdown_context=breakdown_context,
                )
                elapsed = time.monotonic() - t0
                print(f"  [{baseline}] OK ({elapsed:.1f}s) -> {path}")
                if breakdown_path is not None:
                    print(f"  [{baseline}] breakdown -> {breakdown_path}")
                results.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "status": "ok",
                        "error": "",
                        "path": path,
                        "time": elapsed,
                        "rows": len(rows),
                        "breakdown_path": str(breakdown_path) if breakdown_path else "",
                    }
                )
            except Exception as e:
                elapsed = time.monotonic() - t0
                print(f"  [{baseline}] FAILED ({elapsed:.1f}s): {e}")
                results.append(
                    {
                        "dataset": dataset,
                        "baseline": baseline,
                        "status": "error",
                        "error": str(e),
                        "path": str(output_path),
                        "time": elapsed,
                        "rows": len(rows),
                        "breakdown_path": str(breakdown_path) if breakdown_path else "",
                    }
                )
        print()

    print("=== Matrix Summary ===")
    for r in results:
        if r["status"] == "ok":
            print(
                f"{r['dataset']:>14} | {r['baseline']:<11} | OK      | "
                f"{r['rows']:>4} rows | {r['time']:>6.1f}s | {r['path']}"
            )
        elif r["status"] == "skipped":
            print(
                f"{r['dataset']:>14} | {r['baseline']:<11} | SKIPPED | "
                f"{r['rows']:>4} rows | {'-':>6} | {r['path']}"
            )
        else:
            print(
                f"{r['dataset']:>14} | {r['baseline']:<11} | ERROR   | "
                f"{r['rows']:>4} rows | {r['time']:>6.1f}s | {r['error']}"
            )

    if any(r["status"] == "error" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
