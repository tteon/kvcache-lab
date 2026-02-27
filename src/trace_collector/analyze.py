"""Wrapper around lmcache-agent-trace/prefix_analysis.py for trace analysis.

Runs prefix_analysis.py on collected trace files and generates PNG plots
and optional match detail JSONL files.

Usage:
    python -m src.trace_collector.analyze --system all
    python -m src.trace_collector.analyze --system mem0
    python -m src.trace_collector.analyze --system graphiti
    python -m src.trace_collector.analyze --system tau2_telecom
"""

import argparse
import subprocess
import sys

from .common import PROJECT_ROOT, TRACES_DIR

ANALYSIS_SCRIPT = PROJECT_ROOT / "lmcache-agent-trace" / "prefix_analysis.py"

SYSTEMS = ["mem0", "graphiti", "tau2_telecom", "tau2_airline", "tau2_retail"]

TRACE_FILES = {
    "mem0": TRACES_DIR / "mem0_graph" / "mem0_graph_session.jsonl",
    "graphiti": TRACES_DIR / "graphiti_graph" / "graphiti_graph_session.jsonl",
    "tau2_telecom": TRACES_DIR / "tau2_telecom" / "tau2_telecom_session.jsonl",
    "tau2_airline": TRACES_DIR / "tau2_airline" / "tau2_airline_session.jsonl",
    "tau2_retail": TRACES_DIR / "tau2_retail" / "tau2_retail_session.jsonl",
}


def analyze_system(system: str) -> bool:
    """Run prefix_analysis.py on a single system's traces. Returns True on success."""
    trace_file = TRACE_FILES[system]
    result_dir = TRACES_DIR / f"{system}_result"
    result_dir.mkdir(parents=True, exist_ok=True)

    output_png = result_dir / f"{system}_hit_rate.png"
    match_jsonl = result_dir / f"{system}_matches.jsonl"

    if not trace_file.exists():
        print(f"  [{system}] Trace file not found: {trace_file}")
        return False

    # Count lines in trace file
    with open(trace_file) as f:
        line_count = sum(1 for _ in f)
    print(f"  [{system}] Found {line_count} trace entries in {trace_file.name}")

    if line_count == 0:
        print(f"  [{system}] Empty trace file, skipping analysis")
        return False

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
        "gpt2",
    ]

    print(f"  [{system}] Running prefix_analysis.py...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"  [{system}] Analysis complete:")
            print(f"    Plot: {output_png}")
            print(f"    Matches: {match_jsonl}")
            return True
        else:
            print(f"  [{system}] Analysis failed (exit code {result.returncode})")
            if result.stderr:
                print(f"    stderr: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [{system}] Analysis timed out (>600s)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run cache hit rate analysis on collected traces")
    parser.add_argument(
        "--system",
        choices=SYSTEMS + ["all"],
        default="all",
        help="Which system(s) to analyze (default: all)",
    )
    args = parser.parse_args()

    systems = SYSTEMS if args.system == "all" else [args.system]

    print(f"=== Cache Hit Rate Analysis: {', '.join(systems)} ===\n")

    results = {}
    for system in systems:
        print(f"--- {system} ---")
        ok = analyze_system(system)
        results[system] = ok
        print()

    # Summary
    print("=== Analysis Summary ===")
    for system, ok in results.items():
        status = "OK" if ok else "FAILED/SKIPPED"
        print(f"  {system}: {status}")

    if not any(results.values()):
        print("\nNo analyses completed. Run trace collection first:")
        print("  python -m src.trace_collector.run_all --system all")
        sys.exit(1)


if __name__ == "__main__":
    main()
