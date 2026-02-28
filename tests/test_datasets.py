from pathlib import Path

from trace_collector.datasets import DATASET_CHOICES, load_dataset


def test_dataset_choices_include_matrix_targets():
    assert "corpus50" in DATASET_CHOICES
    assert "taubench_legacy" in DATASET_CHOICES
    assert "tau2_airline" in DATASET_CHOICES
    assert "tau2_retail" in DATASET_CHOICES
    assert "tau2_telecom" in DATASET_CHOICES


def test_load_corpus50_default_size():
    rows = load_dataset("corpus50")
    assert len(rows) == 50
    assert all(isinstance(row, str) and row for row in rows)


def test_load_corpus50_with_limit():
    rows = load_dataset("corpus50", num_items=7)
    assert len(rows) == 7


def test_load_taubench_legacy_if_available():
    taubench_dir = Path("lmcache-agent-trace") / "taubench"
    if not taubench_dir.exists():
        return
    rows = load_dataset("taubench_legacy", num_items=5)
    assert len(rows) <= 5
    assert all(isinstance(row, str) for row in rows)
