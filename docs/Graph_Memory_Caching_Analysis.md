# Graph Memory Agents: Where Prefix Caching Fails and Substring Caching Succeeds

## TL;DR

Graph memory agents that dynamically inject retrieved context into LLM prompts destroy traditional prefix caching. Graphiti's temporal knowledge graph architecture produces a **36.4% gap** between prefix and substring caching, while mem0's fixed-prompt design maintains high prefix reuse. Conversational agents without graph memory (tau2-bench baseline) show only **0.8-9.2%** gaps.

### Key Metrics at a Glance

| System | LLM Calls | Input Tokens | Duration | Prefix | Substring | Gap |
|--------|----------:|-------------:|---------:|-------:|----------:|----:|
| **mem0** | 201 (#1-#201) | ~104K | — | 87.6% | 88.5% | +0.9% |
| **Graphiti** | 451 (#1-#451) | ~428K | 12.5 min | 50.1% | 86.5% | **+36.4%** |
| tau2 airline | 51 (#1-#51) | ~126K | 1.3 min | 86.5% | 89.7% | +3.2% |
| tau2 retail | 76 (#1-#76) | ~275K | 2.7 min | 85.6% | 94.8% | +9.2% |
| tau2 telecom | 516 (#1-#516) | ~6.1M | 12.7 min | 98.8% | 99.6% | +0.8% |
| **Total** | **1,295** | **~7.1M** | **~29 min** | | | |

- **Largest gap**: Graphiti at **+36.4%** — substring caching recovers 36 percentage points that prefix caching cannot capture
- **Highest volume**: tau2-bench telecom consumes ~6.1M input tokens across 516 calls, requiring bounded pool sizes (8 GB) to avoid OOM during analysis
- **Most prefix-friendly**: tau2-bench telecom at 98.8% — append-only conversation history with a dominant system prompt

![Comparison Chart](data/traces/comparison_chart.png)

---

## Introduction

Graph memory systems enable LLM agents to persist knowledge across conversations by storing entities and relationships in a graph database. Unlike simple retrieval-augmented generation (RAG), graph memory agents perform **write-time LLM processing**---extracting entities, deduplicating facts, and maintaining relationship graphs---that generates distinct patterns of LLM calls.

This analysis studies how two graph memory systems (**mem0** and **Graphiti**) affect KV cache hit rates, compared to standard conversational agents (**tau2-bench**) that do not use graph memory. The key question: **does graph memory's dynamic context injection create opportunities for substring caching that prefix caching cannot capture?**

### Why tau2-bench as Baseline?

tau2-bench is a conversational agent benchmark that evaluates multi-turn dialogue without any external graph memory. It represents the "traditional" agent pattern: a system prompt + growing conversation history. By comparing graph memory agents against this baseline, we can isolate the caching impact of graph memory's additional LLM processing overhead.

| | Graph Memory Agents (mem0, Graphiti) | Baseline Agent (tau2-bench) |
|---|---|---|
| **Memory** | External graph database (Neo4j) | None (stateless per task) |
| **Write-time LLM calls** | 3-12 calls per knowledge ingestion | 0 (no knowledge storage) |
| **Prompt structure** | Multiple specialized prompts (extraction, dedup, summary) | Single conversational prompt growing over turns |
| **Context mutation** | Dynamic entity/relation injection between calls | Append-only conversation history |

---

## System Architectures

### mem0: Fixed-Prompt Graph Memory

mem0 uses a **fixed 3-call pipeline** for each knowledge ingestion (`add()`) operation:

```
                 mem0 add() Pipeline
                 ════════════════════

  Input: "Marie Curie was a physicist who won the Nobel Prize"
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   ┌─────────────┐ ┌──────────┐ ┌──────────┐
   │ Call 1:     │ │ Call 2:  │ │ Call 3:  │
   │ Entity      │ │ Relation │ │ Delete   │
   │ Extraction  │ │ Extract  │ │ Decision │
   │ (180 tok)   │ │ (364 tok)│ │ (568 tok)│
   └─────────────┘ └──────────┘ └──────────┘
   │  FIXED       │  FIXED      │  FIXED
   │  System      │  System     │  System
   │  Prompt      │  Prompt     │  Prompt
   └──────────────┴─────────────┘
         │
         ▼
   ✅ Same system prompts across ALL 50 items
   ✅ Only user content varies (short factual text)
   ✅ Prefix caching works well
```

**Key properties:**
- 3 LLM calls per `add()`: entity extraction, relation extraction, delete decision
- Each call uses a **fixed system prompt** that is identical across all items
- Only the user message (the factual text being ingested) changes between items
- Average prompt length: **517 tokens** (short, template-dominated)

### Graphiti: Dynamic-Context Temporal Knowledge Graph

Graphiti uses a **variable 5-12 call pipeline** per `add_episode()` that injects previously processed episodes into prompts:

```
               Graphiti add_episode() Pipeline
               ════════════════════════════════

  Input: "Marie Curie was a physicist who won the Nobel Prize"
                        │
    ┌───────────────────┼───────────────────────────┐
    ▼                   ▼                           ▼
┌──────────┐    ┌──────────────┐          ┌──────────────┐
│ Entity   │    │ Entity Dedup │          │ Fact         │
│ Extract  │    │ vs EXISTING  │          │ Extraction   │
│          │    │ entities     │          │ + PREVIOUS   │
│ (630 tok)│    │ (877 tok)    │          │ MESSAGES     │
└──────────┘    └──────────────┘          │ (1065 tok)   │
                       │                  └──────────────┘
              ┌────────┤                         │
              ▼        ▼                         ▼
        ┌──────────┐ ┌──────────┐    ┌──────────────────┐
        │ Edge     │ │ Fact     │    │ Entity Summaries │
        │ Dedup    │ │ Dedup    │    │ (per entity)     │
        │          │ │ EXISTING │    │ (470-500 tok)    │
        └──────────┘ └──────────┘    └──────────────────┘
                                             ×N

  ⚠️ PREVIOUS MESSAGES grows with each item processed
  ⚠️ EXISTING entities/facts change after each episode
  ⚠️ Number of calls varies (5-12) based on entity count
```

**Key properties:**
- 5-12 LLM calls per `add_episode()` (variable based on entity/edge count)
- Prompts inject `<PREVIOUS MESSAGES>`: a growing list of all previously processed episodes
- Prompts inject `EXISTING` entities/facts for deduplication (changes after each episode)
- Average prompt length: **951 tokens** (grows over time: 756 avg for first 50 calls -> 1011 avg for last 50)

### tau2-bench: Conversational Agent Baseline (No Graph Memory)

tau2-bench evaluates multi-turn conversational agents without any external memory system:

```
            tau2-bench Conversation Flow
            ════════════════════════════

  Task: "Customer wants to cancel reservation EHGLP3"
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
     ┌───────────┐      ┌───────────┐
     │ Agent LLM │      │ User Sim  │
     │ (system + │◄────►│ (system + │
     │  history) │      │  scenario)│
     └───────────┘      └───────────┘
           │
     Turn 1: [system] + [user_msg1]           → 417 tok
     Turn 2: [system] + [user_msg1, agent1]   → 3277 tok
     Turn 3: [system] + [..., user_msg2]      → 3354 tok
     ...
     Turn N: [system] + [full history]        → 5706 tok

  ✅ System prompt is constant prefix across all turns
  ✅ Conversation history is append-only (no mutations)
  ✅ Prefix caching works well for system prompt + prior turns
```

**Key properties:**
- Two LLM actors: agent + user simulator (both make LLM calls)
- Prompts grow monotonically (append-only conversation history)
- No context mutations mid-prompt---only new messages appended at the end
- Prompt sizes vary by domain: airline avg 2,489 tok, retail 3,630 tok, telecom 11,874 tok

---

## Why Prefix Caching Fails for Graphiti

Prefix caching relies on **exact prefix matching**---reusing cached KV states only when the beginning of the prompt is identical. Graphiti's architecture breaks this in two ways:

### 1. Dynamic Context Injection (`<PREVIOUS MESSAGES>`)

Graphiti injects all previously processed episodes into extraction prompts. This content **changes after every item**, breaking the prefix after the system instructions:

```
ITEM 1: Entity Extraction                    ITEM 10: Entity Extraction
═══════════════════════                      ══════════════════════════

┌─────────────────────────┐                  ┌─────────────────────────┐
│ system: "You are an AI  │                  │ system: "You are an AI  │
│ assistant that extracts │ ← MATCH          │ assistant that extracts │
│ entity nodes from..."   │                  │ entity nodes from..."   │
├─────────────────────────┤                  ├─────────────────────────┤
│ user:                   │                  │ user:                   │
│ <ENTITY TYPES>          │ ← MATCH          │ <ENTITY TYPES>          │
│ [type definitions...]   │                  │ [type definitions...]   │
│                         │                  │                         │
│ <PREVIOUS MESSAGES>     │                  │ <PREVIOUS MESSAGES>     │
│ []                      │ ← BREAKS HERE ✗  │ ["Marie Curie was...",  │
│                         │                  │  "Albert Einstein...",  │
│                         │                  │  "Python programming..",│
│                         │                  │  ... 9 episodes ...]    │
│ <CURRENT MESSAGE>       │                  │ <CURRENT MESSAGE>       │
│ "Marie Curie was..."    │                  │ "SpaceX, founded by..." │
└─────────────────────────┘                  └─────────────────────────┘

     630 tokens                                    748 tokens (+19%)
```

### 2. Growing EXISTING Entity/Fact Lists

Deduplication prompts include all previously extracted entities and facts, which **accumulate over the session**:

```
ITEM 1: Entity Dedup                         ITEM 30: Entity Dedup
════════════════════                         ═════════════════════

┌─────────────────────────┐                  ┌─────────────────────────┐
│ system: "determines     │                  │ system: "determines     │
│ whether ENTITIES are    │ ← MATCH          │ whether ENTITIES are    │
│ duplicates of..."       │                  │ duplicates of..."       │
├─────────────────────────┤                  ├─────────────────────────┤
│ user:                   │                  │ user:                   │
│ EXISTING ENTITIES:      │                  │ EXISTING ENTITIES:      │
│ [Marie Curie,           │ ← BREAKS HERE ✗  │ [Marie Curie,           │
│  Nobel Prize]           │                  │  Nobel Prize,           │
│                         │                  │  Albert Einstein,       │
│                         │                  │  ... 60+ entities ...]  │
│ NEW ENTITIES:           │                  │ NEW ENTITIES:           │
│ [Marie Curie,           │                  │ [NVIDIA, Jensen Huang,  │
│  Nobel Prize]           │                  │  GPU]                   │
└─────────────────────────┘                  └─────────────────────────┘

     877 tokens                                    ~1800 tokens (+105%)
```

### Prompt Token Growth Over Session

Graphiti's average prompt tokens grow as the knowledge graph accumulates:

| Call Range | Avg Prompt Tokens | Growth |
|-----------|------------------:|-------:|
| Calls 1-50 | 756 | baseline |
| Calls 51-100 | 899 | +19% |
| Calls 101-150 | 974 | +29% |
| Calls 201-250 | 1,000 | +32% |
| Calls 401-450 | 1,011 | +34% |

This growing dynamic content breaks prefix matching at increasingly earlier positions in the prompt, as more and more of the prefix becomes "stale" with each new episode.

---

## Why Prefix Caching Succeeds for mem0 and tau2-bench

### mem0: Template-Dominated Short Prompts

mem0's 3-call pipeline uses **identical system prompts** across all items. Only the short user message (the factual text) varies:

```
ITEM 1: Entity Extraction                   ITEM 50: Entity Extraction
═════════════════════════                    ══════════════════════════

┌─────────────────────────┐                  ┌─────────────────────────┐
│ system: "You are a      │                  │ system: "You are a      │
│ smart assistant who     │ ← MATCH          │ smart assistant who     │
│ understands entities    │                  │ understands entities    │
│ and their types..."     │ ← MATCH          │ and their types..."     │
├─────────────────────────┤                  ├─────────────────────────┤
│ user: "Marie Curie was  │ ← VARIES         │ user: "Netflix was      │
│ a physicist..."         │  (only here!)    │ founded by Reed..."     │
└─────────────────────────┘                  └─────────────────────────┘

     180 tokens                                    183 tokens
     ~170 tok prefix match                         ~170 tok prefix match
     (94% of prompt is reusable prefix)
```

**Result:** The system prompt dominates the total token count. Since it never changes, prefix caching captures almost all cacheable content. Gap = **0.9%**.

### tau2-bench: Append-Only Conversation History

tau2-bench prompts grow by appending new messages at the end. The system prompt + all prior turns form a **valid prefix** for the next turn:

```
TURN 3                                       TURN 4
══════                                       ══════

┌─────────────────────────┐                  ┌─────────────────────────┐
│ system: "You are a      │                  │ system: "You are a      │
│ customer service..."    │ ← PREFIX MATCH   │ customer service..."    │
│ + tool definitions      │                  │ + tool definitions      │
├─────────────────────────┤                  ├─────────────────────────┤
│ user: "Cancel EHGLP3"   │ ← PREFIX MATCH   │ user: "Cancel EHGLP3"   │
│ agent: "Let me check"   │ ← PREFIX MATCH   │ agent: "Let me check"   │
│ user: "Yes, confirm"    │ ← PREFIX MATCH   │ user: "Yes, confirm"    │
│                         │                  │ agent: "Cancelled."     │ ← NEW
│                         │                  │ user: "Thanks"          │ ← NEW
└─────────────────────────┘                  └─────────────────────────┘

     3718 tokens                                   3905 tokens
     (prior 3354 tokens = prefix match = 91%)
```

**Result:** Each new turn reuses the entire prior conversation as a prefix. The longer the conversation, the higher the prefix reuse ratio. Gap = **0.8-9.2%** across domains.

---

## Substring Caching: The Solution for Graphiti

Unlike prefix caching, **substring (block) caching** can match and reuse **any contiguous token block** regardless of position. This is critical for Graphiti:

### Position-Invariant Matching

```
ITEM 1                                       ITEM 10
══════                                       ═══════

Position 0-400:                              Position 0-400:
┌─────────────────────────┐                  ┌─────────────────────────┐
│ system: "You are an AI  │                  │ system: "You are an AI  │
│ assistant that extracts │ ◄─ CACHE HIT ──► │ assistant that extracts │
│ entity nodes..."        │                  │ entity nodes..."        │
└─────────────────────────┘                  └─────────────────────────┘

Position 400-420:                            Position 400-600:
┌─────────────────────────┐                  ┌─────────────────────────┐
│ <PREVIOUS MESSAGES>     │                  │ <PREVIOUS MESSAGES>     │
│ []                      │                  │ ["Marie Curie...",      │
│                         │   DIFFERENT!     │  "Albert Einstein...",  │
│                         │                  │  ... ]                  │
│                         │                  │ ◄─ Contains substrings  │
│                         │                  │    from Items 1-9 ──►   │
│                         │                  │    CACHE HITS!          │
└─────────────────────────┘                  └─────────────────────────┘

Position 420-630:                            Position 600-748:
┌─────────────────────────┐                  ┌─────────────────────────┐
│ <CURRENT MESSAGE>       │                  │ <CURRENT MESSAGE>       │
│ "Marie Curie was a      │                  │ "SpaceX, founded by     │
│ physicist..."           │                  │ Elon Musk..."           │
└─────────────────────────┘                  └─────────────────────────┘

   ✅ SUBSTRING CACHE: Matches by CONTENT, not position
   ✅ System prompts reused regardless of PREVIOUS MESSAGES length
   ✅ PREVIOUS MESSAGES content itself is cacheable (seen in prior items)
   ✅ EXISTING entities/facts overlap with previously cached content
```

### What Gets Cached in Each System

| Block | Graphiti | mem0 | tau2-bench |
|-------|----------|------|------------|
| System instructions | CACHED (same across calls of same type) | CACHED (same across all items) | CACHED (same across all turns) |
| Template structure | CACHED (Pydantic schemas, entity types) | CACHED (tool definitions) | CACHED (tool/function defs) |
| Dynamic context | PARTIALLY CACHED (PREVIOUS MESSAGES contain prior items) | N/A (no dynamic context) | N/A (append-only) |
| User content | PARTIALLY CACHED (entity overlap clusters) | LOW cache (unique facts) | CACHED (prior turns are prefix) |

### Quantified Impact

For Graphiti's average prompt (~951 tokens):

```
PREFIX CACHING:     ~475 / 951 = ~50.1%
                    (Only system prompt + template before dynamic injection point)

SUBSTRING CACHING:  ~821 / 951 = ~86.5%
                    (System prompt + templates + PREVIOUS MESSAGES content +
                     EXISTING entity/fact overlaps, all position-invariant)

GAP:                86.5% - 50.1% = 36.4%
                    (Recovered by matching content at different positions)
```

---

## Empirical Results

### Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Tokenizer** | meta-llama/Llama-3.1-8B |
| **KV Cache Pool** | 8 GB (LRU eviction) |
| **Analysis Tool** | [lmcache-agent-trace](https://github.com/LMCache/lmcache-agent-trace) `prefix_analysis.py` |
| **Graph Memory Corpus** | 50 factual statements across 5 domains with entity overlap clusters |
| **tau2-bench Tasks** | 5 tasks per domain (airline, retail, telecom) |
| **Graph Database** | DozerDB (Neo4j-compatible) with per-system database isolation |
| **LLM** | gpt-4o-mini via OpenAI-compatible endpoint |

### Workload Characteristics

| System | LLM Calls | Avg Prompt Tokens | Total Prompt Tokens | Call Pattern |
|--------|----------:|------------------:|--------------------:|--------------|
| mem0 | 201 | 517 | 104,010 | 3 fixed calls per item (entity, relation, delete) |
| Graphiti | 451 | 951 | 428,686 | 5-12 variable calls per item (extract, dedup, summarize) |
| tau2 airline | 51 | 2,489 | 126,919 | Multi-turn dialogue (agent + user simulator) |
| tau2 retail | 76 | 3,630 | 275,906 | Multi-turn dialogue (agent + user simulator) |
| tau2 telecom | 516 | 11,874 | 6,126,897 | Multi-turn dialogue (agent + user simulator) |

### Cache Hit Rate Results (8 GB Pool, Llama-3.1-8B Tokenizer)

| System | Prefix | Substring | Gap | Key Insight |
|--------|-------:|----------:|----:|-------------|
| **mem0** | 87.6% | 88.5% | **0.9%** | Fixed prompts = prefix-friendly |
| **Graphiti** | 50.1% | 86.5% | **36.4%** | Dynamic context breaks prefix; substring recovers |
| tau2 airline | 86.5% | 89.7% | 3.2% | Append-only history = stable prefix |
| tau2 retail | 85.6% | 94.8% | 9.2% | Longer conversations = more reusable content |
| tau2 telecom | 98.8% | 99.6% | 0.8% | Very long prompts with dominant system prefix |

### Per-System Hit Rate Charts

#### mem0 (Graph Memory - Fixed Prompts)

![mem0 hit rate](data/traces/mem0_result/mem0_hit_rate_8gb.png)

mem0's near-zero gap (0.9%) confirms that its fixed 3-call pipeline is inherently prefix-friendly. The system prompt dominates token count, and only the short user message varies between items.

#### Graphiti (Graph Memory - Dynamic Context)

![Graphiti hit rate](data/traces/graphiti_result/graphiti_hit_rate_8gb.png)

Graphiti shows the largest gap of any system tested (**36.4%**). Prefix caching captures only ~50% because dynamic `<PREVIOUS MESSAGES>` and `EXISTING` entity lists break the prefix early. Substring caching recovers to 86.5% by matching system prompts, templates, and repeated episode content at arbitrary positions.

#### tau2-bench Airline (Baseline)

![tau2 airline hit rate](data/traces/tau2_airline_result/tau2_airline_hit_rate_8gb.png)

#### tau2-bench Retail (Baseline)

![tau2 retail hit rate](data/traces/tau2_retail_result/tau2_retail_hit_rate_8gb.png)

#### tau2-bench Telecom (Baseline)

![tau2 telecom hit rate](data/traces/tau2_telecom_result/tau2_telecom_hit_rate.png)

tau2-bench domains show consistently high prefix hit rates (85-99%) with small gaps. This is expected: conversational agents use append-only history, so each turn's prompt is a strict prefix of the next turn's prompt.

---

## Analysis: Graph Memory vs Conversational Agents

### Why Graph Memory Creates the Largest Caching Gap

The fundamental difference lies in **how prompts evolve between LLM calls**:

| Pattern | Prompt Evolution | Prefix Behavior | Substring Benefit |
|---------|-----------------|-----------------|-------------------|
| **Append-only** (tau2-bench) | New content added at END | Prior turns remain valid prefix | Minimal (prefix already captures most) |
| **Fixed-template** (mem0) | Only user message changes | System prompt is long stable prefix | Minimal (template dominates) |
| **Dynamic injection** (Graphiti) | Content INSERTED in MIDDLE | Prefix breaks at injection point | **Massive** (templates + prior content reusable) |

Graphiti's architecture is uniquely problematic for prefix caching because:
1. **Mid-prompt mutations**: `<PREVIOUS MESSAGES>` is injected between the system prompt and current message, breaking the prefix at the earliest possible point
2. **Accumulating context**: The injected content grows with each item, meaning the "stale prefix" fraction decreases over the session
3. **Repeated content across call types**: The same episode text appears in extraction, dedup, and summary prompts---cacheable as substrings but at different positions

### The Spectrum of Caching Friendliness

```
PREFIX-FRIENDLY                                          PREFIX-HOSTILE
◄──────────────────────────────────────────────────────────────────────►

tau2-telecom    tau2-airline    mem0        tau2-retail    Graphiti
  (0.8% gap)     (3.2% gap)   (0.9% gap)   (9.2% gap)   (36.4% gap)
  append-only    append-only   fixed tmpl   append-only   dynamic inject

                         ┌─────────────────────────────────┐
                         │ Substring caching most valuable  │
                         │ for dynamic injection patterns   │
                         │ (Graphiti-style architectures)   │
                         └─────────────────────────────────┘
```

---

## Practical Considerations: Pool Size and Memory Pressure

### Analysis Tool Memory Characteristics

The cache simulation tool (`prefix_analysis.py`) maintains an **LRU token pool** that stores all input tokens for prefix/substring matching. With an unlimited pool, every token from every LLM call is retained in memory. This creates a critical scalability constraint for large workloads.

### OOM on tau2-bench Telecom

During our experiments, the tau2-bench telecom workload caused **Out-of-Memory (OOM) kills** when analyzed with an unlimited pool:

| System | Total Tokens | Unlimited Pool Analysis | Result |
|--------|------------:|------------------------:|--------|
| mem0 | 104K | ~0.4 GB | OK |
| graphiti | 429K | ~1.6 GB | OK |
| tau2 airline | 127K | ~0.5 GB | OK |
| tau2 retail | 276K | ~1.0 GB | OK |
| **tau2 telecom** | **6,127K** | **~21 GB** | **OOM killed** |

The root cause is twofold:

1. **Unlimited pool = no eviction**: All 6.1M tokens from 516 LLM calls remain in memory. With each token requiring storage for the token ID plus metadata for matching, memory usage scales linearly with total token count.

2. **Substring matching is O(N x M)**: For each new call with M input tokens, the analyzer scans the entire pool of N cached tokens for substring matches. With 516 calls averaging 11,874 tokens each, the cumulative matching work grows quadratically:

```
Call 1:    Match 11,874 tokens against pool of 0        →  trivial
Call 100:  Match 11,874 tokens against pool of ~1.2M    →  moderate
Call 300:  Match 11,874 tokens against pool of ~3.6M    →  heavy
Call 516:  Match 11,874 tokens against pool of ~6.1M    →  OOM killed

Memory growth (unlimited pool):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Call 100   ▓▓▓▓░░░░░░░░░░░░░░░░ ~4 GB
Call 200   ▓▓▓▓▓▓▓▓░░░░░░░░░░░░ ~8 GB
Call 300   ▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░ ~12 GB
Call 400   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░ ~17 GB
Call 516   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ ~21 GB → OOM ✗
```

### Solution: Bounded Pool Sizes

Limiting the pool to `--pool-sizes 1 2 4 8` (GB) enables LRU eviction, which keeps memory bounded while retaining the most recently used tokens:

```
Pool Size = 8 GB (LRU eviction)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Call 1-100:    Pool fills to ~4 GB       → no eviction yet
Call 100-250:  Pool fills to 8 GB cap    → LRU eviction begins
Call 250-516:  Pool stays at 8 GB        → oldest tokens evicted

Memory: stable at 8 GB ✓
Result: prefix 98.8%, substring 99.6%   (vs unlimited: would be ~same)
```

The 8 GB pool achieves nearly identical hit rates to an unlimited pool because tau2-bench telecom's prompts are dominated by a **large, stable system prompt** (~2,300 tokens) that is always in the LRU cache. The conversation history from recent turns also fits within the 8 GB budget.

### Implications for Real Deployments

This OOM behavior in the simulation mirrors a real concern for LMCache deployments:

| Scenario | Token Volume | Recommendation |
|----------|-------------|----------------|
| Graph memory (mem0) | Low (104K per 50 items) | Unlimited pool feasible |
| Graph memory (Graphiti) | Medium (429K per 50 items) | 8 GB pool sufficient |
| Short conversations (airline) | Low (127K per 5 tasks) | Unlimited pool feasible |
| Long conversations (telecom) | **Very high (6.1M per 5 tasks)** | **Bounded pool required** |

For production systems serving telecom-scale workloads (long multi-turn conversations with large system prompts), **bounded KV cache pools with LRU eviction are not just a memory optimization---they are a necessity**. The marginal hit rate improvement from unlimited pools does not justify the memory cost.

---

## Conclusion

Graph memory agents present a unique challenge for LLM inference caching. Our analysis of five workloads reveals a clear pattern:

1. **Fixed-prompt systems (mem0)** are inherently prefix-friendly. With identical system prompts across all operations, prefix caching achieves 87.6% hit rate with only 0.9% left on the table for substring matching.

2. **Dynamic context injection (Graphiti)** fundamentally breaks prefix caching. By inserting growing `<PREVIOUS MESSAGES>` and `EXISTING` entity lists into the middle of prompts, Graphiti reduces prefix cache hit rate to 50.1%. Substring caching recovers to 86.5%---a **36.4% improvement** that represents the single largest gap across all systems tested.

3. **Conversational agents (tau2-bench)** fall between these extremes. Their append-only conversation history naturally preserves prefixes, yielding 0.8-9.2% gaps depending on conversation length and domain complexity.

**For LLM serving systems like LMCache, substring (block) caching is not just an optimization---it is essential for graph memory agent workloads.** As more AI applications adopt knowledge graph architectures with dynamic context injection (Graphiti, similar temporal KG systems), the demand for position-invariant cache matching will only grow.

---

## References

- [LMCache Agent Trace Analysis Toolkit](https://github.com/LMCache/lmcache-agent-trace)
- [mem0: Memory for AI Agents](https://github.com/mem0ai/mem0)
- [Graphiti: Temporal Knowledge Graph](https://github.com/getzep/graphiti)
- [tau2-bench: Conversational Agent Benchmark](https://github.com/sierra-research/tau2-bench)
- [MemGPT Caching Analysis](https://github.com/kobe0938/blog/blob/master/mem-gpt/MemGPT_Caching_Analysis.md)
- [LMCache: Disaggregated KV Cache for LLM Serving](https://github.com/LMCache/LMCache)
