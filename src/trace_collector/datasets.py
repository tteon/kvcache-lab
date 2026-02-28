"""Dataset loaders for matrix-style trace experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .common import PROJECT_ROOT, TEST_CORPUS

DATASET_CHOICES = [
    "corpus50",
    "tau2_airline",
    "tau2_retail",
    "tau2_telecom",
    "taubench_legacy",
]


def _limit(items: list[str], num_items: int | None) -> list[str]:
    if num_items is None or num_items < 0:
        return items
    return items[:num_items]


def _task_to_text(task) -> str:
    parts = [f"[task_id]\n{task.id}"]
    if getattr(task, "description", None):
        parts.append(f"[description]\n{task.description}")
    if getattr(task, "user_scenario", None):
        parts.append(f"[user_scenario]\n{task.user_scenario}")
    if getattr(task, "ticket", None):
        parts.append(f"[ticket]\n{task.ticket}")
    return "\n\n".join(parts)


def _load_tau2_tasks(domain: str, num_items: int | None) -> list[str]:
    from tau2.run import get_tasks

    tasks = get_tasks(task_set_name=domain, task_split_name="base", num_tasks=num_items)
    return [_task_to_text(task) for task in tasks]


def _iter_legacy_taubench_inputs(taubench_dir: Path) -> Iterable[str]:
    for jsonl_path in sorted(taubench_dir.glob("*.jsonl")):
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                input_text = payload.get("input", "")
                if input_text:
                    yield input_text


def load_dataset(dataset: str, num_items: int | None = None) -> list[str]:
    """Load dataset rows as plain text prompts for collectors."""
    if dataset == "corpus50":
        return _limit(list(TEST_CORPUS), num_items)
    if dataset == "tau2_airline":
        return _load_tau2_tasks("airline", num_items)
    if dataset == "tau2_retail":
        return _load_tau2_tasks("retail", num_items)
    if dataset == "tau2_telecom":
        return _load_tau2_tasks("telecom", num_items)
    if dataset == "taubench_legacy":
        taubench_dir = PROJECT_ROOT / "lmcache-agent-trace" / "taubench"
        rows = list(_iter_legacy_taubench_inputs(taubench_dir))
        return _limit(rows, num_items)
    raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {DATASET_CHOICES}")


def dataset_description(dataset: str) -> str:
    if dataset == "corpus50":
        return "Shared 50-item factual corpus from trace_collector.common.TEST_CORPUS"
    if dataset == "tau2_airline":
        return "tau2 airline domain tasks (base split)"
    if dataset == "tau2_retail":
        return "tau2 retail domain tasks (base split)"
    if dataset == "tau2_telecom":
        return "tau2 telecom domain tasks (base split)"
    if dataset == "taubench_legacy":
        return "Legacy taubench trace inputs from lmcache-agent-trace/taubench/*.jsonl"
    return dataset
