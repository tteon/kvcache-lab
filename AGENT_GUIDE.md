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
| `run_experiments.sh` | **Host Wrapper** | OBSERVE/EXECUTE this to launch experiments. |
| `scripts/docker_entrypoint.sh` | **Container Entrypoint** | EDIT this to change experiment logic (internal). |
| `benchmark.py` | **Load Generator** | READ/EDIT this to change request patterns (NVTX, metrics). |
| `analysis/plot_results.py` | **Analysis** | EXECUTE this to generate visualizations from `results/`. |
| `configs/*.yaml` | **Backend Config** | EDIT this to tune LMCache parameters. |
| `results/` | **Output** | READ this to analyze outcomes. |

## 🕹️ Operational Workflows

### 1. Running an Experiment
**Pattern**: `./run_experiments.sh [WORKLOAD] [TIER]`
**Workloads**: `agent`, `rag`.
**Tiers**: `baseline`, `cpu`, `disk`.

**Example Command**:
```bash
./run_experiments.sh agent cpu
```

**Advanced Configuration**:
Agents can tune parameters via flags:
```bash
./run_experiments.sh \
    --workload rag \
    --tier disk \
    --gpu-mem-util 0.95 \
    --max-model-len 8000 \
    --chunk-size 512
```

### 2. Analyzing Results
**Location**: `results/<workload>_<tier>_<timestamp>/`
**Structure**:
- `metrics/`: Benchmark CSVs (`metrics_*.csv`) and logs.
- `profiles/`: Nsight Systems reports (`report.nsys-rep`).
- `plots/`: Visualizations (`comparison_ttft.png`, etc.).
- `analysis/`: Theoretical bottleneck analysis (`bottleneck_analysis.txt`).

**Key File to Check**: `analysis/bottleneck_analysis.txt` - Contains the "Compute-Bound" vs "I/O-Bound" verdict.

### 3. Modifying Infrastructure
- **To add a new offload backend**: Create a new YAML in `configs/` and update `scripts/utils.sh` to mount it.
- **To change monitoring**: Edit `start_metrics_collection` in `scripts/utils.sh`.

## ⚠️ Critical Constraints
- **Docker**: The lab runs heavily on Docker. Ensure `nvidia-container-toolkit` is active.
- **VRAM**: The `baseline` tier is sensitive to VRAM limits. If OOM occurs, reduce `--gpu-memory-utilization`.
- **Privileges**: `sudo` might be needed for `dstat` or `nsys` in some environments (handled inside container usually).
