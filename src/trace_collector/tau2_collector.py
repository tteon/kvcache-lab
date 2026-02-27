"""Collect LLM call traces from tau2-bench benchmark simulations.

tau2-bench evaluates conversational agents across domains (airline, retail, telecom).
Both the agent and user simulator make LLM calls via litellm.completion().

Interception: Monkeypatch litellm.completion to capture all LLM calls before/after
tau2's generate() function processes them.

Domains:
  - airline: Flight booking and customer service
  - retail: E-commerce and product support
  - telecom: Telecommunications service management (new in tau2)
"""

import json
import logging
import time

import litellm

from .common import (
    LLM_MODEL,
    TRACES_DIR,
    TraceLogger,
)

logger = logging.getLogger(__name__)

# Supported tau2 domains
TAU2_DOMAINS = ["airline", "retail", "telecom"]

# Default number of tasks per domain (each task = 1 multi-turn conversation)
DEFAULT_NUM_TASKS = 10


def _patch_litellm(trace_logger: TraceLogger):
    """Monkeypatch litellm.completion to capture all LLM calls.

    Returns a restore function to undo the patch.
    """
    original_completion = litellm.completion

    def traced_completion(*args, **kwargs):
        # Extract messages (positional arg or kwarg)
        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])

        # Build input text matching prefix_analysis.py format
        input_parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "") or ""
            input_parts.append(f"{role}: {content}")
        input_text = "\n".join(input_parts)

        t0 = time.monotonic()
        response = original_completion(*args, **kwargs)
        latency_ms = round((time.monotonic() - t0) * 1000)

        # Extract output text
        choice = response.choices[0]
        output_text = ""
        call_type = None
        if choice.message.tool_calls:
            tool_outputs = []
            for tc in choice.message.tool_calls:
                tool_outputs.append(
                    json.dumps({"name": tc.function.name, "arguments": tc.function.arguments})
                )
            output_text = "\n".join(tool_outputs)
            call_type = choice.message.tool_calls[0].function.name
        elif choice.message.content:
            output_text = choice.message.content

        # Build metadata
        metadata = {
            "model": getattr(response, "model", None),
            "finish_reason": choice.finish_reason,
            "latency_ms": latency_ms,
        }
        if call_type:
            metadata["call_type"] = call_type

        usage = getattr(response, "usage", None)
        if usage:
            metadata["prompt_tokens"] = getattr(usage, "prompt_tokens", None)
            metadata["completion_tokens"] = getattr(usage, "completion_tokens", None)
            metadata["total_tokens"] = getattr(usage, "total_tokens", None)
            details = getattr(usage, "prompt_tokens_details", None)
            if details:
                metadata["cached_tokens"] = getattr(details, "cached_tokens", 0) or 0

        trace_logger.log(input_text, output_text, **metadata)
        return response

    litellm.completion = traced_completion
    return lambda: setattr(litellm, "completion", original_completion)


def collect(domain: str = "telecom", num_tasks: int = DEFAULT_NUM_TASKS) -> str:
    """Run tau2-bench trace collection for a given domain.

    Args:
        domain: tau2 domain name (airline, retail, telecom).
        num_tasks: Number of tasks (conversations) to simulate.

    Returns:
        Path to the output JSONL trace file.
    """
    if domain not in TAU2_DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'. Choose from: {TAU2_DOMAINS}")

    output_path = TRACES_DIR / f"tau2_{domain}" / f"tau2_{domain}_session.jsonl"

    with TraceLogger(output_path, session_id=f"tau2_{domain}") as trace_logger:
        restore = _patch_litellm(trace_logger)

        try:
            # Disable litellm cache to get clean traces
            litellm.disable_cache()

            # Import tau2 after patching litellm
            from tau2.data_model.simulation import RunConfig
            from tau2.run import run_domain

            config = RunConfig(
                domain=domain,
                task_set_name=domain,
                task_split_name="base",
                num_tasks=num_tasks,
                num_trials=1,
                agent="llm_agent",
                llm_agent=LLM_MODEL,
                llm_args_agent={"temperature": 0.0},
                user="user_simulator",
                llm_user=LLM_MODEL,
                llm_args_user={"temperature": 0.0},
                max_steps=200,
                max_errors=10,
                max_concurrency=1,  # Sequential for cleaner trace ordering
                seed=300,
                log_level="WARNING",
                enforce_communication_protocol=False,
            )

            print(f"[tau2-{domain}] Collecting traces for {num_tasks} tasks using {LLM_MODEL}...")
            run_domain(config)
        finally:
            restore()

    print(f"[tau2-{domain}] Traces written to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect tau2-bench LLM traces")
    parser.add_argument(
        "--domain",
        choices=TAU2_DOMAINS,
        default="telecom",
        help="tau2 domain to simulate (default: telecom)",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=DEFAULT_NUM_TASKS,
        help=f"Number of tasks to run (default: {DEFAULT_NUM_TASKS})",
    )
    args = parser.parse_args()
    collect(domain=args.domain, num_tasks=args.num_tasks)
