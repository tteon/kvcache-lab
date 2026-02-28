# lmcache-contributor

Trace collection and cache-hit analysis workspace for memory-augmented agent workloads.

This repository compares prompt-cache behavior across:
- `mem0` graph memory
- `graphiti` temporal graph memory
- `tau2-bench` conversational workloads (`airline`, `retail`, `telecom`)

The core question is:
- "When prompt prefixes are unstable, how much can substring/block caching recover?"

Primary focus:
- `mem0 (graph)` vs `graphiti`
- `tau2` domains are supporting baselines, not the main comparison target.

## What This Framework Does

End-to-end pipeline:

1. Run agent workloads and intercept every LLM call.
2. Normalize calls into a shared JSONL trace format.
3. Run `lmcache-agent-trace/prefix_analysis.py`.
4. Compare `prefix` vs `substring` hit rates and their gap.

Matrix extension:
- Run `dataset x baseline` experiments with `openai_base`, `mem0`, `graphiti`.
- Datasets include `corpus50`, `tau2_*`, and `taubench_legacy` input replay.
- In matrix mode, `tau2_*` and `taubench_legacy` are replayed as prompt-text datasets
  (not full tau2 environment simulation loops).

Data flow:

`collector -> trace jsonl -> prefix_analysis -> matches jsonl + plot -> comparison chart`

## Framework Components

| Layer | Role | Key files |
|---|---|---|
| Collector | Executes workload and captures LLM calls | `src/trace_collector/*_collector.py` |
| Normalizer | Writes unified trace schema | `src/trace_collector/common.py` |
| Analyzer | Runs LMCache analysis script | `src/trace_collector/analyze.py` |
| Aggregator | Builds cross-system chart | `src/trace_collector/compare_chart.py` |
| Analysis engine | Prefix/substring hit computation | `lmcache-agent-trace/prefix_analysis.py` |

Interception strategies by system:

| System | How calls are intercepted | Output trace |
|---|---|---|
| `mem0` | OpenAI `response_callback` | `data/traces/mem0_graph/mem0_graph_session.jsonl` |
| `graphiti` | `OpenAIGenericClient` subclass override | `data/traces/graphiti_graph/graphiti_graph_session.jsonl` |
| `tau2` | `litellm.completion` monkeypatch | `data/traces/tau2_<domain>/tau2_<domain>_session.jsonl` |

## Main Comparison: mem0 (graph) vs graphiti

If you want to analyze the core framework behavior, start here first:

1. `mem0` and `graphiti` only collection
2. Per-system prefix/substring result comparison
3. Gap interpretation at architecture level

Fast path commands:

```bash
uv run python -m src.trace_collector.run_all --system mem0
uv run python -m src.trace_collector.run_all --system graphiti

uv run python -m src.trace_collector.analyze --system mem0
uv run python -m src.trace_collector.analyze --system graphiti
```

Matrix mode commands:

```bash
# 1) Collect matrix traces (all datasets x all baselines)
uv run python -m src.trace_collector.run_matrix --dataset all --baseline all

# 2) Analyze matrix traces
uv run python -m src.trace_collector.analyze_matrix --dataset all --baseline all

# 3) Build markdown report
uv run python -m src.trace_collector.matrix_report -o docs/matrix_breakdown.md
```

Core outputs:
- `data/traces/mem0_graph/mem0_graph_session.jsonl`
- `data/traces/graphiti_graph/graphiti_graph_session.jsonl`
- `data/traces/mem0_result/mem0_matches.jsonl`
- `data/traces/graphiti_result/graphiti_matches.jsonl`

Interpretation guide:
- `mem0` high prefix + small gap -> stable scaffold/template reuse
- `graphiti` lower prefix + larger gap -> dynamic context injection with reusable moved blocks

## Repository Layout

```text
src/trace_collector/
  common.py            # env resolution, test corpus, TraceLogger
  datasets.py          # dataset loaders for matrix experiments
  run_all.py           # collector orchestrator
  run_matrix.py        # dataset x baseline trace orchestrator
  mem0_collector.py    # mem0 trace collection
  graphiti_collector.py# graphiti trace collection
  openai_base_collector.py # direct OpenAI baseline collection
  tau2_collector.py    # tau2 trace collection
  analyze.py           # wrapper around prefix_analysis.py
  analyze_matrix.py    # matrix trace analyzer
  matrix_report.py     # matrix markdown report generator
  compare_chart.py     # cross-system comparison chart

data/traces/
  */*.jsonl            # raw traces
  *_result/*.jsonl     # substring match logs
  *_result/*.png       # per-system hit-rate plots
  comparison_chart.png # combined chart

lmcache-agent-trace/
  prefix_analysis.py   # core algorithm (tokenize + prefix/substring scoring)
```

## Requirements

- Python `>=3.10`
- `uv`
- Docker + Docker Compose (for DozerDB/Neo4j)
- LLM API key and endpoint

## Setup

1. Install dependencies:

```bash
uv sync --dev
```

2. Prepare Neo4j directories/plugins and start DB:

```bash
./setup.sh
docker-compose up -d
```

3. Create isolated databases:

```cypher
CREATE DATABASE mem0store;
CREATE DATABASE graphitistore;
```

4. Configure `.env`:

```bash
OPENAI_API_KEY=...

# Optional compatibility settings
GPU_API_KEY=...
GPU_ENDPOINT=https://api.openai.com/v1
GPU_MODEL=gpt-4o-mini

# Preferred runtime overrides
LLM_API_KEY=...
LLM_API_BASE=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

Environment variable resolution:
- `LLM_API_KEY` -> `GPU_API_KEY` -> `OPENAI_API_KEY`
- `LLM_API_BASE` -> `GPU_ENDPOINT` -> `OPENAI_BASE_URL` -> OpenAI default
- `LLM_MODEL` -> `GPU_MODEL`

## Run Workflow

1. Verify endpoint capability (chat, tool calling, JSON mode):

```bash
uv run python -m src.trace_collector.test_endpoint
```

2. Collect traces:

```bash
uv run python -m src.trace_collector.run_all --system all
```

3. Analyze hit rates:

```bash
uv run python -m src.trace_collector.analyze --system all
```

4. Build cross-system chart:

```bash
uv run python -m src.trace_collector.compare_chart
```

For core analysis only (recommended first pass):

```bash
uv run python -m src.trace_collector.run_all --system mem0
uv run python -m src.trace_collector.run_all --system graphiti
uv run python -m src.trace_collector.analyze --system mem0
uv run python -m src.trace_collector.analyze --system graphiti
```

For dataset x baseline matrix:

```bash
uv run python -m src.trace_collector.run_matrix --dataset all --baseline all
uv run python -m src.trace_collector.analyze_matrix --dataset all --baseline all
uv run python -m src.trace_collector.matrix_report -o docs/matrix_breakdown.md
```

## How To Analyze The Whole Framework

Use this order to break down results like a MemGPT-style report.

1. Topline by system
- Open `data/traces/comparison_chart.png`.
- Compare `prefix`, `substring`, and `gap = substring - prefix`.

2. Inspect per-system raw traces
- Check prompt construction behavior in raw JSONL:
  - `input`: what changes turn-to-turn
  - `output`: tool calls / structured responses

3. Inspect substring match logs
- Read `*_matches.jsonl` for:
  - `InputLen`
  - `Matches[]` (`MatchStart`, `MatchEnd`, `PrevStep`, `PrevMatchStart`, `PrevMatchEnd`)

4. Classify each system pattern
- High prefix, low gap: stable prompt prefixes.
- Low prefix, high substring: dynamic insertion/reordering with reusable blocks.
- Low both: little cross-call reuse.

For `mem0` vs `graphiti`, prioritize these questions:
- Where does prefix break earliest?
- Which repeated blocks are recovered only by substring?
- Is the gap driven by context mutation, retrieval position shift, or tool output pattern?

5. Write conclusions at two levels
- Architecture level: why that system creates this shape.
- Operations level: expected impact on prefill latency and compute reuse.

## Code Reading Order (Recommended)

If your goal is framework-level understanding, read in this order:

1. `src/trace_collector/run_all.py`
2. `src/trace_collector/mem0_collector.py`
3. `src/trace_collector/graphiti_collector.py`
4. `src/trace_collector/tau2_collector.py`
5. `src/trace_collector/common.py`
6. `src/trace_collector/analyze.py`
7. `lmcache-agent-trace/prefix_analysis.py`
8. `src/trace_collector/compare_chart.py`

## Testing

Run project tests only:

```bash
uv run pytest -q
```

`pytest` ignores reference/generated directories (`codebase`, `vendor`, `data`, `lmcache-agent-trace`) and collects from `tests/`.

## Artifact Policy

- Treat runtime analysis outputs as generated artifacts:
  - `data/traces/*_result/`
  - `data/traces/*/*.jsonl.bak*`
- Keep intentional fixtures only.
- Prefer committing scripts/config that regenerate outputs over large result blobs.
