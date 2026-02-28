"""Orchestrator CLI for running trace collectors across all systems.

Usage:
    python -m src.trace_collector.run_all --system all
    python -m src.trace_collector.run_all --system openai_base
    python -m src.trace_collector.run_all --system mem0
    python -m src.trace_collector.run_all --system graphiti
    python -m src.trace_collector.run_all --system tau2_telecom
    python -m src.trace_collector.run_all --system tau2_airline
    python -m src.trace_collector.run_all --system tau2_retail
"""

import argparse
import sys
import time

SYSTEMS = ["openai_base", "mem0", "graphiti", "tau2_telecom", "tau2_airline", "tau2_retail"]


def run_openai_base():
    from .openai_base_collector import collect

    return collect()


def run_mem0():
    from .mem0_collector import collect

    return collect()


def run_graphiti():
    from .graphiti_collector import collect

    return collect()


def run_tau2_telecom():
    from .tau2_collector import collect

    return collect(domain="telecom")


def run_tau2_airline():
    from .tau2_collector import collect

    return collect(domain="airline")


def run_tau2_retail():
    from .tau2_collector import collect

    return collect(domain="retail")


COLLECTORS = {
    "openai_base": run_openai_base,
    "mem0": run_mem0,
    "graphiti": run_graphiti,
    "tau2_telecom": run_tau2_telecom,
    "tau2_airline": run_tau2_airline,
    "tau2_retail": run_tau2_retail,
}


def main():
    parser = argparse.ArgumentParser(description="Run LLM trace collectors for graph memory systems")
    parser.add_argument(
        "--system",
        choices=SYSTEMS + ["all"],
        default="all",
        help="Which system(s) to collect traces from (default: all)",
    )
    args = parser.parse_args()

    systems = SYSTEMS if args.system == "all" else [args.system]

    print(f"=== Trace Collection: {', '.join(systems)} ===\n")

    results = {}
    for system in systems:
        print(f"--- {system} ---")
        start = time.time()
        try:
            path = COLLECTORS[system]()
            elapsed = time.time() - start
            results[system] = {"status": "ok", "path": path, "time": elapsed}
            print(f"  Completed in {elapsed:.1f}s\n")
        except Exception as e:
            elapsed = time.time() - start
            results[system] = {"status": "error", "error": str(e), "time": elapsed}
            print(f"  FAILED after {elapsed:.1f}s: {e}\n")

    # Summary
    print("=== Summary ===")
    for system, r in results.items():
        if r["status"] == "ok":
            print(f"  {system}: OK ({r['time']:.1f}s) -> {r['path']}")
        else:
            print(f"  {system}: FAILED ({r['time']:.1f}s) -> {r['error']}")

    if any(r["status"] == "error" for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
