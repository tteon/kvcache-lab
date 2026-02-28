# MemGPT: Where Prefix Caching Fails and Substring Caching Succeeds

## TL;DR

MemGPT's dynamic memory architecture breaks traditional prefix caching due to frequent context mutations. Substring caching achieves **~93.4% cache hit rate** compared to only **~43.9%** with prefix caching.

| Caching Strategy | Cache Hit Rate |
|------------------|----------------|
| **Prefix Caching** | ~43.9% |
| **Substring Caching** | ~93.4% |

![LMCache MemGPT benchmark results](https://raw.githubusercontent.com/LMCache/lmcache-agent-trace/main/memgpt_result/memgpt1.png)

---

## Introduction

MemGPT is an innovative system that enables LLMs to manage their own memory, effectively creating "virtual context" that extends beyond the finite context window. By treating the context window like an operating system manages virtual memory, MemGPT allows persistent conversations and document analysis that would otherwise exceed token limits.

## MemGPT Memory Architecture

![MemGPT Architecture](image.png)

The architecture consists of three main layers:

### 1. LLM Context Window

The context window is divided into distinct sections:

| Section | Type | Description |
|---------|------|-------------|
| **System Instructions** | Read-Only (Static) | The MemGPT system prompt defining behavior and available functions |
| **Working Context** | Read-Write | Current user/persona information, updated via function calls |
| **FIFO Queue** | Read-Write | Recent conversation history, managed by Queue Manager |

### 2. External Storage Systems

- **Archival Storage**: Long-term memory for facts, documents, and persistent information
- **Recall Storage**: Searchable history of past conversations

### 3. Middleware Components

- **Function Executor**: Handles read/write operations to Archival Storage and Working Context
- **Queue Manager**: Controls the FIFO Queue eviction policy and Recall Storage writes

---

## Why Prefix Caching Fails for MemGPT

Prefix caching relies on **exact prefix matching**—reusing cached KV states only when the beginning of the prompt is identical. MemGPT's architecture fundamentally breaks this assumption:

### The Dynamic Context Problem

```
TURN 1 PROMPT                              TURN 2 PROMPT
═══════════════════                        ═══════════════════

┌───────────────────┐                      ┌───────────────────┐
│ System Instruct.  │ ← MATCH              │ System Instruct.  │
├───────────────────┤                      ├───────────────────┤
│ Working Context:  │                      │ Working Context:  │
│ User: "Alice"     │ ← MATCH              │ User: "Alice"     │
│                   │                      │ Born in: 1999     │ ← PREFIX BREAKS HERE ✗
│ Archival: Doc_A   │                      │ Archival: Doc_A   │ ← SAME CONTENT (CACHEABLE)
│ "The capital..."  │                      │ "The capital..."  │   
│ Func Definitions  │                      │ Func Definitions  │ ← SAME (CACHEABLE)
├───────────────────┤                      ├───────────────────┤
│ FIFO Queue:       │                      │ FIFO Queue:       │
│ [msg1, msg2, msg3]│                      │ [msg2, msg3, msg4]│ ← SHIFTED!
│ Recalled Conv     │                      │ Recalled Conv     │ ← CACHEABLE
└───────────────────┘                      └───────────────────┘
         │                                          │
         └──────────────┬───────────────────────────┘
                        ▼
        ┌───────────────────────────────────────┐
        │  ❌ PREFIX CACHE: very low reuse rate │
        │                                       │
        │  • Working Context updates break      │
        │    prefix after system instructions   │
        │  • FIFO Queue shifts every turn       │
        │  • Archival Docs & Func Defs same     │
        │    content but different positions    │
        └───────────────────────────────────────┘
```

### Key Reasons for Prefix Cache Failure

1. **Working Context Mutations**: 

    **Information Update**: The agent updates persona/user info via `core_memory_replace()` and `core_memory_append()`, changing mid-prompt content.
    
    **Archival Retrieval Variability**: Retrieved documents appear at different positions depending on current queue length

    **Function Call Results**: Tool outputs are injected dynamically, changing prompt structure unpredictably

2. **FIFO Queue Dynamics**: New messages push old ones out—the queue shifts every turn, destroying any prefix after system instructions


---

## Non-Prefix (Substring) Caching: The Solution

Unlike prefix caching, **substring/block caching** can match and reuse **any contiguous token block** regardless of position. This is transformative for MemGPT:

### What Can Be Cached and Reused

### Position-Invariant Matching

```
TURN 1                                    TURN 5
═══════                                   ═══════

Position 0-2,000:                          Position 0-2,000:
┌─────────────────┐                       ┌─────────────────┐
│ System Instruct │ ◄─── CACHE HIT ────►  │ System Instruct │
└─────────────────┘                       └─────────────────┘

Position 2,000-4,500:                    Position 2,000-5,000:
┌─────────────────┐                       ┌─────────────────┐
│ Working Context │ ◄─── CACHE HIT ────►  │ Working Context │
│ (Archival Docs, │   (Archival Docs &    │ (Archival Docs, │
│  Func Defs)     │    Func Defs cached)  │  Func Defs)     │
└─────────────────┘                       │ Part updated/   │
                                          │ added, +500 tok │
                                          └─────────────────┘

Position 4,500-16,000:                   Position 5,000-16,000:
┌─────────────────┐                       ┌─────────────────┐
│ FIFO Queue      │ ◄─── CACHE HIT ────►  │ FIFO Queue      │
│ (Recalled Conv) │   (different pos!)    │ (Recalled Conv) │
└─────────────────┘                       └─────────────────┘

    ✅ NON-PREFIX CACHE: Matches by CONTENT, not position
```

---

## Empirical Results: LMCache Agent Trace

The [LMCache MemGPT benchmark results](https://github.com/LMCache/lmcache-agent-trace/tree/main/memgpt_result) demonstrate the dramatic difference:

![LMCache MemGPT benchmark results](https://raw.githubusercontent.com/LMCache/lmcache-agent-trace/main/memgpt_result/memgpt1.png)

### Observed Cache Performance

| Caching Strategy | Cache Hit Rate | Explanation |
|------------------|----------------|-------------|
| **Prefix Caching** | ~43.9% | Only system prompt prefix matches |
| **Substring Caching** | ~93.4% | System prompt + archival docs + function defs all reused |


### Quantified Benefits

### What Can Be Cached and Reused

CACHEABLE BLOCKS + TYPICAL MEMGPT PROMPT (~16,000 tokens)

| Block | Cacheability | Token share | Notes |
|-------|--------------|-------------|-------|
| System Instructions | CACHED | ~2000 | Static across turns (reusable) |
| Working Context(including Archival Documents and Function Definitions) | DYNAMIC | ~2500 | Archival Documents and Function Definitions Cachable |
| FIFO Queue (Recent Messages & Recalled Conversations) | DYNAMIC(recent) and CACHED(recalled) | ~11,500 | Recent messages & Recalled Conversations putting into Working Context, reusable if recalled |

PREFIX CACHING:     (2000 + 1/2 * 2500) / 16,000 = ~20% (suppose prefix matching break at middle of working context).

SUBSTRING CACHING: > 15,500 / 16,000 = 95%+ (system + archival + functions, minus 500 from 16,000 context length)

---

## Conclusion

MemGPT's memory management paradigm—with its dynamic Working Context, shifting FIFO Queue, and variable archival retrievals—fundamentally breaks prefix caching assumptions. The content that could be reused (system prompts, retrieved documents, function definitions) appears at different positions across turns.

**Substring/block caching** solves this by matching token sequences regardless of position, enabling:
- **Higher cache hit rates** (93.4% vs 43.9%)
- **Reduced prefill latency** for repeated archival retrievals
- **Lower computational costs** for long-running agent sessions

For memory-augmented LLM agents like MemGPT, non-prefix caching isn't just an optimization—it's essential for practical deployment at scale.

---

## References

- [LMCache MemGPT Trace Results](https://github.com/LMCache/lmcache-agent-trace/tree/main/memgpt_result)
- [MemGPT: Towards LLMs as Operating Systems](https://memgpt.ai/)
