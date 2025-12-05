# Antigravity Agent Context

This document is designed to help **Antigravity** (and other AI agents) quickly understand the purpose, structure, and operational logic of this repository.

## 🧠 Repository Intent
**Goal**: Benchmark and profile **vLLM** inference performance with **LMCache** KV cache offloading.
**Key Metrics**: Time-To-First-Token (TTFT), End-to-End Latency (E2E), GPU PCIe Bandwidth, Disk I/O.
**Tiers**: Baseline (GPU), CPU Offload, Disk Offload, Scalability (Redis).

## 🗺️ Codebase Map

| File/Dir | Role | Agent Action |
|---|---|---|
| `Makefile` | **Command Interface** | EXECUTE `make baseline`, `make analyze`, etc. for quick operations. |
| `run_experiments.sh` | **Entry Point** | EXECUTE this directly for custom flags not covered by Makefile. |
| `benchmark.py` | **Load Generator** | READ/EDIT this to change request patterns (NVTX, metrics). |
| `analysis/plot_results.py` | **Analysis** | EXECUTE this to generate visualizations from `results/`. |
| `configs/*.yaml` | **Backend Config** | EDIT this to tune LMCache parameters. |
| `results/` | **Output** | READ this to analyze outcomes. |

## 🕹️ Operational Workflows

### 1. Running an Experiment
**Pattern**: `./run_experiments.sh --tier [TIER] [OPTIONS]`
**Tiers**: `baseline`, `cpu`, `disk`, `scalability`.
**Common Options**:
- `--model`: HuggingFace model ID (requires `HF_TOKEN` in `.env`).
- `--profile`: Enables **Nsight Systems** profiling (output in `profiles/`).
- `--label`: Custom tag for directory naming.

**Example Command (Agentic Workload)**:
```bash
./run_experiments.sh --tier cpu --prompt-len 1000 --gen-len 200 --num-requests 20 --label agent_simulation
```

### 2. Analyzing Results
**Pattern**: `python3 plot_results.py --input "[PATTERN]" --output-prefix "[PREFIX]"`
**Input**: Usually `results/archive/metrics_*.csv` or specific run directories.
**Outputs**:
- `*_ttft.png`, `*_e2e.png`: Latency plots.
- `*_pcie.png`: Hardware utilization.
- `*_report.md`: Textual summary table.

### 3. Modifying Infrastructure
- **To add a new offload backend**: Create a new YAML in `configs/` and update `scripts/utils.sh` to mount it.
- **To change monitoring**: Edit `start_metrics_collection` in `scripts/utils.sh`.

## ⚠️ Critical Constraints
- **Docker**: The lab runs heavily on Docker. Ensure `nvidia-container-toolkit` is active.
- **VRAM**: The `baseline` tier is sensitive to VRAM limits. If OOM occurs, reduce `--gpu-memory-utilization`.
- **Privileges**: `sudo` might be needed for `dstat` or `nsys` in some environments (handled inside container usually).
