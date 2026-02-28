# Framework Tracing Issues: What We Changed and Why

This document records the framework-level issues encountered while building LLM call interception for each system, and the workarounds we implemented.

---

## Overview

To analyze KV cache hit rates, we need **full trace logs** of every LLM call: the complete input prompt, output text, and token usage metadata. None of the three frameworks (mem0, graphiti, tau2-bench) were designed with external trace collection in mind, so each required a different interception strategy.

| Framework | Interception Method | Difficulty | Key Issue |
|-----------|-------------------|------------|-----------|
| **mem0** | Built-in `response_callback` | Low | Callback exists but is undocumented; metadata access requires internal knowledge |
| **Graphiti** | Subclass `OpenAIGenericClient` | High | Default client (`OpenAIClient`) uses `responses.parse()` incompatible with custom endpoints; had to rewrite `_generate_response()` entirely |
| **tau2-bench** | Monkeypatch `litellm.completion` | Medium | No hook mechanism; endpoint configuration requires aligning multiple env vars simultaneously |

---

## mem0

### AS-IS (Framework Default)

mem0's `OpenAIConfig` supports a `response_callback` parameter that is called after every LLM completion:

```python
# mem0/llms/openai.py (line ~140-146)
if self.config.response_callback:
    self.config.response_callback(self, response, params)
```

**Issues:**
1. **Undocumented API**: `response_callback` is not in mem0's public docs. We discovered it by reading the source code at `codebase/mem0/mem0/llms/openai.py`.
2. **Callback signature is opaque**: The callback receives `(llm_instance, response, params)` but the types are not documented. You need to know that:
   - `params["messages"]` contains the OpenAI-format messages list
   - `response` is a raw `openai.ChatCompletion` object
   - `response.usage.prompt_tokens_details.cached_tokens` exists for server-side cache stats
3. **No call type identification**: mem0 doesn't pass which pipeline stage (entity extraction, relation extraction, delete decision) triggered the call. We infer it from `tool_calls[0].function.name` when present.

### TO-BE (Our Implementation)

```python
# src/trace_collector/mem0_collector.py

def callback(llm_instance, response, params):
    messages = params.get("messages", [])          # ← undocumented access
    input_text = messages_to_input_text(messages)

    # Extract call_type from tool call function name
    choice = response.choices[0].message
    if choice.tool_calls:
        call_type = choice.tool_calls[0].function.name  # e.g., "extract_entities"

    # Extract metadata from raw response
    usage = response.usage
    metadata = {
        "prompt_tokens": usage.prompt_tokens,
        "cached_tokens": usage.prompt_tokens_details.cached_tokens,  # ← deep nesting
    }
    trace_logger.log(input_text, output_text, **metadata)

config = {
    "llm": {"provider": "openai", "config": {
        "response_callback": callback,  # ← plug in directly
    }}
}
```

**What we changed:** Nothing in mem0's source. We only use the built-in callback. The difficulty was **discovering and understanding** the undocumented callback interface.

---

## Graphiti

### AS-IS (Framework Default)

Graphiti provides two LLM client classes:

| Client | API Method | Structured Output | Custom Endpoint |
|--------|-----------|-------------------|-----------------|
| `OpenAIClient` | `client.beta.chat.completions.parse()` | Native Pydantic parsing | **Incompatible** (requires OpenAI-specific beta API) |
| `OpenAIGenericClient` | `client.chat.completions.create()` + `json.loads()` | Manual JSON schema + parsing | Compatible |

**Issues:**

1. **`OpenAIClient.parse()` breaks on custom endpoints**: Graphiti's default `OpenAIClient` uses OpenAI's beta `responses.parse()` API for Pydantic structured output. Custom/GPU endpoints (vLLM, TGI, etc.) don't implement this beta API, causing immediate failures.

2. **`OpenAIGenericClient._generate_response()` doesn't expose the raw response**: The parent method calls the OpenAI API and returns only the parsed JSON dict, discarding the `response` object that contains usage metadata (prompt_tokens, cached_tokens, latency):

   ```python
   # graphiti_core/llm_client/openai_generic_client.py (simplified)
   class OpenAIGenericClient:
       async def _generate_response(self, messages, response_model, ...):
           response = await self.client.chat.completions.create(...)
           result = json.loads(response.choices[0].message.content)
           return result  # ← raw response object is lost here
   ```

3. **Cannot use `super()._generate_response()`**: Because the parent discards the raw response, calling `super()` and then trying to access metadata is impossible. We must **replicate the entire parent logic** to capture the response before it's discarded.

### TO-BE (Our Implementation)

```python
# src/trace_collector/graphiti_collector.py

class TracingOpenAIGenericClient(OpenAIGenericClient):
    """Must rewrite _generate_response entirely --- cannot use super()."""

    async def _generate_response(self, messages, response_model, ...):
        # 1. Capture input BEFORE _clean_input modifies messages
        input_text = "\n".join(f"{m.role}: {m.content}" for m in messages)

        # 2. Replicate parent logic (clean input → build messages → set format)
        openai_messages = []
        for m in messages:
            m.content = self._clean_input(m.content)
            openai_messages.append({"role": m.role, "content": m.content})

        response_format = {"type": "json_object"}
        if response_model is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "schema": response_model.model_json_schema(),
                },
            }

        # 3. Make API call ourselves (not via super) to capture raw response
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format=response_format,
        )

        # 4. Now we have the raw response for metadata
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens,
            "cached_tokens": response.usage.prompt_tokens_details.cached_tokens,
            "latency_ms": latency_ms,
        }

        result = json.loads(response.choices[0].message.content)
        self.trace_logger.log(input_text, json.dumps(result), **metadata)
        return result
```

**What we changed:**
- Used `OpenAIGenericClient` instead of `OpenAIClient` (avoids `responses.parse()`)
- Completely rewrote `_generate_response()` (~50 lines) to replicate parent logic while retaining the raw response object
- This is **fragile**: if Graphiti updates `_generate_response()` internals, our subclass may break

### Upstream Improvement Suggestion

Graphiti could expose a hook or return the raw response alongside the parsed result:

```python
# Proposed: return (result, response) or add a callback parameter
async def _generate_response(self, messages, ..., on_response=None):
    response = await self.client.chat.completions.create(...)
    if on_response:
        on_response(response)
    return json.loads(response.choices[0].message.content)
```

---

## tau2-bench

### AS-IS (Framework Default)

tau2-bench uses `litellm.completion()` for all LLM calls (both agent and user simulator). There is no built-in callback, hook, or tracing mechanism.

**Issues:**

1. **No interception point**: tau2's `generate()` function calls `litellm.completion()` directly. There's no middleware layer, event system, or callback parameter to hook into.

2. **Endpoint configuration is fragmented**: tau2/litellm reads API configuration from multiple sources simultaneously:
   - `os.environ["OPENAI_API_KEY"]`
   - `os.environ["OPENAI_API_BASE"]`
   - `os.environ["OPENAI_BASE_URL"]`
   - `litellm.api_key`
   - `litellm.api_base`

   All must be set consistently, and restored after collection to avoid side effects.

3. **Model name routing**: litellm has its own model name routing logic. Passing a model like `gpt-4o-mini` works, but custom model names may require `openai/` prefix depending on the endpoint.

### TO-BE (Our Implementation)

```python
# src/trace_collector/tau2_collector.py

# 1. Monkeypatch litellm.completion
def _patch_litellm(trace_logger):
    original_completion = litellm.completion

    def traced_completion(*args, **kwargs):
        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
        input_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages)

        response = original_completion(*args, **kwargs)   # ← call original

        # Extract metadata from response
        metadata = {"prompt_tokens": response.usage.prompt_tokens, ...}
        trace_logger.log(input_text, output_text, **metadata)
        return response

    litellm.completion = traced_completion
    return lambda: setattr(litellm, "completion", original_completion)

# 2. Align endpoint configuration across all sources
def _configure_litellm_endpoint():
    previous = {key: os.environ.get(key) for key in env_keys}
    os.environ["OPENAI_API_KEY"] = LLM_API_KEY
    os.environ["OPENAI_API_BASE"] = LLM_API_BASE
    os.environ["OPENAI_BASE_URL"] = LLM_API_BASE
    litellm.api_key = LLM_API_KEY
    litellm.api_base = LLM_API_BASE
    return restore_function   # ← restores all env vars after collection
```

**What we changed:**
- Monkeypatch `litellm.completion` at runtime (no source modifications)
- Manage 5 separate configuration points (3 env vars + 2 litellm attrs) with save/restore
- Import tau2 modules **after** patching to ensure all calls go through our wrapper

---

## Summary: Interception Difficulty Spectrum

```
EASY                                                    HARD
◄──────────────────────────────────────────────────────────►

mem0                    tau2-bench              Graphiti
(built-in callback)     (monkeypatch)           (subclass + rewrite)

✅ No source changes     ✅ No source changes    ⚠️ No source changes BUT
✅ Clean API             ⚠️ Fragile patch          full method rewrite
⚠️ Undocumented         ⚠️ Multi-point config   ⚠️ Tightly coupled to
                                                   internal implementation
```

### Key Takeaway

None of the three frameworks provide first-class support for **LLM call observability**. For the growing ecosystem of agent trace analysis (LMCache, prompt caching research, cost monitoring), frameworks should consider:

1. **Standardized callback/hook API**: A `on_llm_call(request, response)` hook at the framework level
2. **Raw response access**: Don't discard the API response object before users can inspect it
3. **Call type annotation**: Label which pipeline stage triggered each LLM call (extraction, dedup, etc.)

---

## File References

| File | Framework | Lines of Interest |
|------|-----------|-------------------|
| `src/trace_collector/mem0_collector.py` | mem0 | `_make_callback()` (L39-97) |
| `src/trace_collector/graphiti_collector.py` | Graphiti | `TracingOpenAIGenericClient._generate_response()` (L88-173) |
| `src/trace_collector/tau2_collector.py` | tau2-bench | `_patch_litellm()` (L70-130), `_configure_litellm_endpoint()` (L39-67) |
| `codebase/mem0/mem0/llms/openai.py` | mem0 source | `response_callback` invocation (L140-146) |
| `codebase/graphiti/graphiti_core/llm_client/openai_generic_client.py` | Graphiti source | `_generate_response()` parent method |
