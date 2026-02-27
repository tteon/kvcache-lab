# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LMCache Contributor analyzes how different AI agent workloads (graph memory systems) affect LLM prompt cache hit rates. It compares prefix vs substring caching across two graph memory systems (mem0, graphiti) and existing baselines (tau-bench, MemGPT) using the [lmcache-agent-trace](https://github.com/LMCache/lmcache-agent-trace) analysis toolkit.

**Core hypothesis**: Graphiti's dynamic context injection (episode history in prompts) destroys prefix caching but creates repeating substrings, producing the largest prefix-substring gap between the two systems.

## Tech Stack

- **Python 3.10** with UV package manager
- **Neo4j/DozerDB 5.26.3** - Graph database (Docker, multi-database isolation)
- **Mem0** (`mem0ai[graph]`) - Graph memory with OpenAI tool calling
- **Graphiti** (`graphiti-core`) - Temporal knowledge graph with Pydantic structured output
- **LMCache Agent Trace** - Prefix/substring cache hit rate analysis (Llama-3.1-8B tokenizer)

## Commands

```bash
# Infrastructure
./setup.sh                                        # Create Neo4j dirs, download plugins
docker-compose up -d                              # Start DozerDB

# Trace Collection
python -m src.trace_collector.test_endpoint       # Verify GPU endpoint connectivity
python -m src.trace_collector.run_all --system all # Collect traces from all systems
python -m src.trace_collector.run_all --system mem0      # Collect mem0 only
python -m src.trace_collector.run_all --system graphiti  # Collect graphiti only

# Analysis
python -m src.trace_collector.analyze --system all       # Run prefix_analysis.py on all traces
python -m src.trace_collector.analyze --system graphiti  # Analyze single system

# Legacy
python src/main.py                                # Multi-database isolation test
python src/test_agent.py                          # Agent and DB health checks
```

## Project Structure

```
src/
├── main.py                           # Multi-DB isolation test (mem0 + graphiti factory methods)
├── test_agent.py                     # OpenAI Agent + DozerDB health checks
└── trace_collector/                  # LLM trace collection pipeline
    ├── common.py                     # TraceLogger, TEST_CORPUS, config constants
    ├── test_endpoint.py              # GPU endpoint verification (chat, tools, JSON mode)
    ├── mem0_collector.py             # mem0 traces via response_callback
    ├── graphiti_collector.py         # graphiti traces via TracingOpenAIGenericClient subclass
    ├── run_all.py                    # CLI orchestrator
    └── analyze.py                    # prefix_analysis.py wrapper

data/traces/                          # Collected trace output
├── mem0_graph/                       # ~30 lines (10 items x 3 calls/add)
├── graphiti_graph/                   # ~50-120 lines (10 items x 5-12 calls/episode)
├── mem0_graph_result/                # Analysis output (PNG + match JSONL)
└── graphiti_graph_result/

lmcache-agent-trace/                  # Analysis toolkit (cloned from LMCache)
├── prefix_analysis.py                # Core: LRU token pool, prefix/substring matching
├── merge_matches.py                  # Post-processing: merge adjacent matches
├── combine_jsonl.py                  # Utility: merge multiple JSONL files
├── taubench/                         # Baseline: 25 sessions, 519 entries (airline+retail)
├── taubench_result/                  # 24 PNG plots
├── memgpt/                           # Baseline: 2 sessions, 19 entries
├── miniswe/                          # Baseline: 20 sessions, 402 entries (Django PRs)
└── magagent/                         # Baseline: 25 sessions, 746 entries (travel planning)

codebase/                             # Cloned reference repos (gitignored)
├── mem0/                             # https://github.com/mem0ai/mem0
└── graphiti/                         # https://github.com/getzep/graphiti

example_codebase/                     # GNN+LLM reference (Neo4j + PCST + Llama)
```

## Trace Format (JSONL)

All traces use the same format, compatible with `prefix_analysis.py`:

```jsonl
{"timestamp": 1760587752017385, "input": "system: You are...\nuser: Marie Curie was...", "output": "{\"name\": \"extract_entities\", ...}", "session_id": "mem0_graph"}
```

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | int | Unix microseconds (`int(time.time() * 1_000_000)`) |
| `input` | string | Full prompt text (role-prefixed messages concatenated with `\n`) |
| `output` | string | LLM response text (or serialized tool call JSON) |
| `session_id` | string | Groups related trace entries per system |

## Database Isolation

Each system writes to an isolated database to enable post-hoc comparison:

| System | Database | Storage | Notes |
|--------|----------|---------|-------|
| mem0 | `mem0store` (Neo4j) | DozerDB | `database` param in Neo4jConfig |
| graphiti | `graphitistore` (Neo4j) | DozerDB | `database` param in Neo4jDriver constructor |

**Prerequisite**: `mem0store` and `graphitistore` databases must exist in DozerDB before trace collection. Create via Cypher:

```cypher
CREATE DATABASE mem0store;
CREATE DATABASE graphitistore;
```

## LLM Call Interception Methods

### mem0: response_callback (built-in)

```python
# OpenAIConfig has native callback support: response_callback(llm_instance, response, params)
config = {
    "graph_store": {
        "llm": {"provider": "openai", "config": {
            "model": GPU_MODEL, "openai_base_url": GPU_ENDPOINT,
            "response_callback": my_callback,
        }}
    }
}
```

- 3 fixed LLM calls per `add()`: entity extraction, relation extraction, delete decision
- Source: `codebase/mem0/mem0/llms/openai.py:140-146`

### graphiti: TracingOpenAIGenericClient (subclass)

```python
# Subclass OpenAIGenericClient, override _generate_response()
# Uses OpenAIGenericClient (not OpenAIClient) to avoid responses.parse()
class TracingOpenAIGenericClient(OpenAIGenericClient):
    async def _generate_response(self, messages, ...):
        result = await super()._generate_response(messages, ...)
        trace_logger.log(input_text, json.dumps(result))
        return result
```

- 5-12 variable LLM calls per `add_episode()` depending on entity/edge count
- Source: `codebase/graphiti/graphiti_core/llm_client/openai_generic_client.py`

## Environment

The `.env` file contains:
- `OPENAI_API_KEY` - Used as default for GPU endpoint API key and OpenAI embeddings
- `GPU_ENDPOINT` (optional) - Override GPU endpoint URL (default: `http://89.169.103.68:30080/v1`)
- `GPU_MODEL` (optional) - Override model name (default: `openai/gpt-oss-120b`)
- `GPU_API_KEY` (optional) - Override API key (default: falls back to `OPENAI_API_KEY`)
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` - Database connection (defaults: `bolt://localhost:7687`, `neo4j`, `password`)

## Experimental Design

### Independent Variable
Graph memory system: mem0, graphiti + baselines (tau-bench, MemGPT, mini-swe, MetaGPT)

### Controlled Variables
- Same GPU endpoint (`openai/gpt-oss-120b`)
- Same 10-item test corpus (`TEST_CORPUS` in `common.py`)
- Same Neo4j backend (DozerDB with isolated databases)
- Same tokenizer for analysis (Llama-3.1-8B)

### Dependent Variables
- **Prefix cache hit rate**: Fraction of input tokens matching a cached prefix
- **Substring cache hit rate**: Fraction of input tokens matching any cached substring
- **Prefix-substring gap**: Where LMCache's substring matching adds the most value
- **LLM calls per operation**: API call count per write cycle

### Predicted Cache Hit Rates

| System | Prefix | Substring | Gap | Rationale |
|--------|--------|-----------|-----|-----------|
| mem0 | 40-60% | 30-50% | Small | Fixed system prompts, 3 calls |
| graphiti | 15-35% | 50-70% | **Large** | Episode context breaks prefix; templates repeat as substrings |

### Key Comparisons
1. **mem0 vs graphiti**: Fixed prompts vs dynamic context injection -> prefix gap
2. **Graph memory vs baselines**: Graph overhead vs traditional agent workloads

### Baselines (from lmcache-agent-trace)

| System | Source | Sessions | Entries | Domain |
|--------|--------|----------|---------|--------|
| tau-bench | [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench) | 25 | 519 | Airline + retail customer service |
| MemGPT | lmcache-agent-trace | 2 | 19 | Stateful memory agents |
| mini-swe | lmcache-agent-trace | 20 | 402 | Software engineering (Django PRs) |
| MetaGPT | lmcache-agent-trace | 25 | 746 | Multi-agent travel planning |

## GNN + LLM Pipeline (Reference)

Reference implementation in `example_codebase/neo4j-gnn-llm-example/`.

```
Question -> Vector Search (top-K) -> Cypher Multi-Hop -> PCST Pruning -> GNN Encoding -> LLM -> Answer
```

Key files: `train.py`, `STaRKQADataset.py`, `compute_pcst.py`, `data-loading/load_data.py`

Training: `python train.py --gnn_hidden_channels 1536 --num_gnn_layers 4 --lr 1e-5 --epochs 2 --batch_size 4 --llama_version llama3.1-8b --checkpointing`
