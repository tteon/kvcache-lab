"""Generate a comparison chart across all analyzed systems.

Reads match JSONL files from each system's result directory and produces
a grouped bar chart comparing prefix vs substring hit rates + gap.

Usage:
    python -m src.trace_collector.compare_chart
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import TRACES_DIR

SYSTEMS = {
    "openai_base": {
        "match_candidates": [
            TRACES_DIR / "openai_base_result" / "openai_base_matches.jsonl",
            TRACES_DIR / "openai_base_result" / "openai_base_matches_8gb.jsonl",
        ],
        "label": "openai\nbase",
    },
    "mem0": {
        "match_candidates": [
            TRACES_DIR / "mem0_result" / "mem0_matches.jsonl",
            TRACES_DIR / "mem0_result" / "mem0_matches_8gb.jsonl",
        ],
        "label": "mem0\n(graph memory)",
    },
    "graphiti": {
        "match_candidates": [
            TRACES_DIR / "graphiti_result" / "graphiti_matches.jsonl",
            TRACES_DIR / "graphiti_result" / "graphiti_matches_8gb.jsonl",
        ],
        "label": "graphiti\n(temporal KG)",
    },
    "tau2_airline": {
        "match_candidates": [
            TRACES_DIR / "tau2_airline_result" / "tau2_airline_matches.jsonl",
            TRACES_DIR / "tau2_airline_result" / "tau2_airline_matches_8gb.jsonl",
        ],
        "label": "tau2\nairline",
    },
    "tau2_retail": {
        "match_candidates": [
            TRACES_DIR / "tau2_retail_result" / "tau2_retail_matches.jsonl",
            TRACES_DIR / "tau2_retail_result" / "tau2_retail_matches_8gb.jsonl",
        ],
        "label": "tau2\nretail",
    },
    "tau2_telecom": {
        "match_candidates": [
            TRACES_DIR / "tau2_telecom_result" / "tau2_telecom_matches.jsonl",
            TRACES_DIR / "tau2_telecom_result" / "tau2_telecom_matches_8gb.jsonl",
        ],
        "label": "tau2\ntelecom",
    },
}


def _compute_hit_rates(matches_path: Path) -> dict:
    """Compute average prefix and substring hit rates from matches JSONL.

    Format per line: {"StepID": int, "InputLen": int, "Matches": [{MatchStart, MatchEnd, ...}]}
    - Prefix hit: longest contiguous match from token 0 (each chunk is 16 tokens)
    - Substring hit: union of all matched token ranges
    """
    total_input_tokens = 0
    total_prefix_matched = 0
    total_substring_matched = 0
    count = 0

    with open(matches_path) as f:
        for line in f:
            entry = json.loads(line)
            input_len = entry.get("InputLen", 0)
            matches = entry.get("Matches", [])
            count += 1
            total_input_tokens += input_len

            if input_len == 0 or not matches:
                continue

            # Substring: union of all matched ranges in current input
            matched_tokens = set()
            for m in matches:
                for t in range(m["MatchStart"], m["MatchEnd"]):
                    matched_tokens.add(t)
            total_substring_matched += len(matched_tokens)

            # Prefix: longest contiguous match starting from token 0
            # Sort matched ranges that start at 0 and extend contiguously
            sorted_ranges = sorted(
                [(m["MatchStart"], m["MatchEnd"]) for m in matches],
                key=lambda x: x[0],
            )
            prefix_end = 0
            for start, end in sorted_ranges:
                if start <= prefix_end:
                    prefix_end = max(prefix_end, end)
                else:
                    break
            total_prefix_matched += prefix_end

    if total_input_tokens == 0:
        return {"prefix": 0, "substring": 0, "gap": 0, "count": 0, "avg_tokens": 0}

    prefix = total_prefix_matched / total_input_tokens
    substring = total_substring_matched / total_input_tokens
    return {
        "prefix": prefix,
        "substring": substring,
        "gap": substring - prefix,
        "count": count,
        "avg_tokens": total_input_tokens / count,
    }


def main():
    output_path = TRACES_DIR / "comparison_chart.png"

    results = {}
    for name, info in SYSTEMS.items():
        matches_path = next((p for p in info["match_candidates"] if p.exists()), None)
        if matches_path is None:
            print(f"  [{name}] No matches file, skipping")
            continue
        rates = _compute_hit_rates(matches_path)
        rates["label"] = info["label"]
        results[name] = rates
        print(
            f"  [{name}] {rates['count']} calls, avg {rates['avg_tokens']:.0f} tokens, "
            f"prefix {rates['prefix']*100:.1f}%, substring {rates['substring']*100:.1f}%, "
            f"gap {rates['gap']*100:.1f}%"
        )

    if not results:
        print("No results to plot!")
        return

    # --- Plot ---
    names = list(results.keys())
    labels = [results[n]["label"] for n in names]
    prefix_vals = [results[n]["prefix"] * 100 for n in names]
    substring_vals = [results[n]["substring"] * 100 for n in names]
    gap_vals = [results[n]["gap"] * 100 for n in names]

    x = np.arange(len(names))
    width = 0.3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1],
                                     gridspec_kw={"hspace": 0.3})

    # Top: prefix vs substring bars
    bars1 = ax1.bar(x - width / 2, prefix_vals, width, label="Prefix Matching",
                     color="#4A90D9", edgecolor="white", linewidth=0.5)
    bars2 = ax1.bar(x + width / 2, substring_vals, width, label="Substring Matching",
                     color="#D94A7A", edgecolor="white", linewidth=0.5)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#4A90D9")
    for bar in bars2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color="#D94A7A")

    ax1.set_ylabel("Cache Hit Rate (%)", fontsize=12)
    ax1.set_title("LMCache Prefix vs Substring Hit Rate by Agent Scaffolding\n"
                   "(Llama-3.1-8B tokenizer, unlimited pool)", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # Add entry counts as secondary labels
    for i, name in enumerate(names):
        r = results[name]
        ax1.text(i, -6, f"{r['count']} calls\n{r['avg_tokens']:.0f} tok/call",
                 ha="center", va="top", fontsize=8, color="gray")

    # Bottom: gap bars (substring - prefix)
    colors = ["#FF6B35" if g > 5 else "#888888" for g in gap_vals]
    bars3 = ax2.bar(x, gap_vals, width * 1.5, color=colors, edgecolor="white", linewidth=0.5)
    for bar, g in zip(bars3, gap_vals):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.3, f"+{g:.1f}%",
                 ha="center", va="bottom", fontsize=10, fontweight="bold",
                 color="#FF6B35" if g > 5 else "#888888")

    ax2.set_ylabel("Substring - Prefix Gap (%)", fontsize=12)
    ax2.set_title("Where LMCache Substring Matching Adds Value", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0, max(gap_vals) * 1.3 + 2)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison chart saved to: {output_path}")


if __name__ == "__main__":
    main()
