# Matrix Breakdown Report

## Scope

- Datasets: `corpus50`, `tau2_airline`, `tau2_retail`, `tau2_telecom`, `taubench_legacy`
- Baselines: `openai_base`, `mem0`, `graphiti`

## Active Config

- `LLM_API_BASE`: `https://api.openai.com/v1`
- `LLM_MODEL`: `gpt-4o-mini`
- `NEO4J_URI`: `bolt://localhost:7687`
- `NEO4J_USERNAME`: `neo4j`
- Tokenizer in analysis: `meta-llama/Llama-3.1-8B`

## Dataset Notes

- `corpus50`: Shared 50-item factual corpus from trace_collector.common.TEST_CORPUS
- `tau2_airline`: tau2 airline domain tasks (base split)
- `tau2_retail`: tau2 retail domain tasks (base split)
- `tau2_telecom`: tau2 telecom domain tasks (base split)
- `taubench_legacy`: Legacy taubench trace inputs from lmcache-agent-trace/taubench/*.jsonl

## Matrix Status

| Dataset | Baseline | Status | Calls | Avg input tokens | Prefix | Substring | Gap |
|---|---|---|---:|---:|---:|---:|---:|
| corpus50 | openai_base | not_collected | 0 | 0.0 | 0.00% | 0.00% | 0.00% |
| corpus50 | mem0 | analyzed | 201 | 455.9 | 86.14% | 88.46% | 2.32% |
| corpus50 | graphiti | analyzed | 451 | 817.4 | 49.82% | 86.45% | 36.63% |
| tau2_airline | openai_base | analyzed | 51 | 1613.8 | 86.43% | 89.72% | 3.29% |
| tau2_airline | mem0 | not_collected | 0 | 0.0 | 0.00% | 0.00% | 0.00% |
| tau2_airline | graphiti | not_collected | 0 | 0.0 | 0.00% | 0.00% | 0.00% |
| tau2_retail | openai_base | analyzed | 76 | 2257.5 | 86.86% | 94.80% | 7.94% |
| tau2_retail | mem0 | not_collected | 0 | 0.0 | 0.00% | 0.00% | 0.00% |
| tau2_retail | graphiti | not_collected | 0 | 0.0 | 0.00% | 0.00% | 0.00% |
| tau2_telecom | openai_base | analyzed | 516 | 9796.9 | 19.26% | 19.35% | 0.09% |
| tau2_telecom | mem0 | not_collected | 0 | 0.0 | 0.00% | 0.00% | 0.00% |
| tau2_telecom | graphiti | not_collected | 0 | 0.0 | 0.00% | 0.00% | 0.00% |
| taubench_legacy | openai_base | not_collected | 0 | 0.0 | 0.00% | 0.00% | 0.00% |
| taubench_legacy | mem0 | not_collected | 0 | 0.0 | 0.00% | 0.00% | 0.00% |
| taubench_legacy | graphiti | not_collected | 0 | 0.0 | 0.00% | 0.00% | 0.00% |

## Interpretation Hints

- High prefix + small gap: prompt prefixes are stable.
- Low prefix + large gap: prompt blocks move, substring reuse dominates.
- Low both: low cross-call reuse in prompt content.
