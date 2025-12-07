# vLLM + LMCache Experiment Lab

This repository contains a reproducible experiment lab designed to benchmark and analyze the performance of **vLLM** integrated with **LMCache**. It focuses on quantifying the effects of KV cache offloading across different storage tiers (GPU, CPU, Disk) and profiling system bottlenecks for various LLM workloads.

## 🚀 Features

*   **Multi-Tier Offloading**: Configurations for GPU-only (Baseline), CPU RAM offloading, and Local Disk offloading.
*   **Workload Simulation**: Flexible benchmark script to simulate **Long Context** (RAG) and **Agentic** workloads.
*   **Detailed Profiling**:
    *   **Latency Metrics**: Time-to-First-Token (TTFT), Inter-Token Latency (ITL), End-to-End (E2E) Latency.
    *   **System Metrics**: GPU PCIe bandwidth monitoring, CPU/Disk I/O tracking.
    *   **NVTX Annotation**: Automatic tagging of "Prefill" and "Decode" stages in Nsight timelines.
    *   **KV Cache Analysis**: Impact of offloading on memory efficiency and throughput.
    *   **Bottleneck Calculator**: Theoretical speedup analysis based on compute vs I/O bandwidth.
*   **Unified Pipeline**: Seamless execution of benchmark followed by automatic analysis (Plotting + Speedup Calc).
*   **Visualization**: Automated plotting tools to compare performance across tiers.

## 📂 Repository Structure

```
.
├── Makefile                # Unified command interface (make baseline, make analyze)
├── benchmark.py            # Async OpenAI-compatible benchmark client
├── local_experiment_runner.sh # Unified Experiment + Analysis Runner
├── run_experiments.sh      # Legacy Orchestration script
├── analysis/               # Analysis tools
│   ├── plot_results.py     # Plotting script
│   └── bottleneck_calculator.py # Theoretical speedup/bottleneck calculator
├── scripts/                # Helper scripts
│   └── utils.sh            # Common functions
├── configs/                # LMCache configurations (yaml)
├── requirements.txt        # Python dependencies
└── README.md               # Documentation
```

## 🛠️ Prerequisites

*   **Docker** (with NVIDIA Container Toolkit support)
*   **NVIDIA GPU** (Tested on A100/H100, min 20GB VRAM for Llama-3-8B)
*   **Python 3.8+**

## ⚡ Quick Start

1.  **Setup Environment**:
    Create a `.env` file with your Hugging Face token:
    ```bash
    echo "HF_TOKEN=your_token_here" > .env
    ```

2.  **Install Client Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run Experiments**:
    The repository uses a unified runner script that handles Docker overrides and execution.

    *   **Syntax**:
        ```bash
        ./run_experiments.sh [OPTIONS]
        ```

    *   **Options**:
        *   `--workload [agent|rag]`: Workload type (Default: agent)
        *   `--tier [baseline|cpu|disk]`: Offload tier (Default: cpu)
        *   `--gpu-mem-util [0.0-1.0]`: vLLM GPU memory utilization (Default: 0.90)
        *   `--max-model-len [INT]`: Maximum context length (Default: 5000)
        *   `--chunk-size [INT]`: LMCache chunk size (Default: 256)
        *   `--dtype [half|float16|bfloat16]`: Model precision (Default: half)

    *   **Examples**:
        ```bash
        # Agent Workload, CPU Offload, High GPU Mem
        ./run_experiments.sh --workload agent --tier cpu --gpu-mem-util 0.95

        # RAG Workload, Disk Offload, Large Context
        ./run_experiments.sh --workload rag --tier disk --max-model-len 8000
        ```
    
    *   **Using Makefile**:
        ```bash
        make baseline   # Runs agent workload on GPU
        make disk       # Runs rag workload on Disk
        ```

## 📊 Workload Analysis

This lab allows you to simulate specific workloads to identify bottlenecks.

### 1. Long Context (RAG / Document Analysis)
*   **Characteristics**: Long input prompts, short to medium generation.
*   **Bottleneck**: Prefill Compute (TTFT) and VRAM Capacity (KV Cache).
*   **Simulation**:
**Simulation**:
    ```bash
    ./run_experiments.sh rag disk
    ```
*   **Analysis**: Check `ttft` in the results. LMCache Disk offload allows handling contexts larger than GPU RAM, though with a latency penalty during retrieval.

### 2. Agentic Workloads
*   **Characteristics**: Moderate context, multi-turn.
*   **Bottleneck**: Latency (TTFT) and Throughput.
*   **Simulation**:
    ```bash
    ./run_experiments.sh agent cpu
    ```

## 📈 Analyzing Results

The **Unified Pipeline** automatically generates a structured results directory (`results/<workload>_<tier>_<timestamp>/`) containing:
*   `metrics/`: Raw CSV metrics and vLLM server logs.
*   `profiles/`: Nsight Systems report (`.nsys-rep`).
*   `plots/`: Generated comparison plots (TTFT, E2E Latency).
*   `analysis/`: Theoretical bottleneck analysis (`bottleneck_analysis.txt`).

**Understanding the Bottleneck Calculator**:
The `bottleneck_calculator.py` estimates theoretical performance based on:
$T$: Compute time per token (GPU bound).
$R$: Retrieval time per token (I/O bound).
$\alpha$: Cache miss rate (portion of data retrieved from storage).
It outputs identifying whether the workload is **Compute-Bound** or **I/O-Bound** and suggests optimization strategies.

The lab generates CSV files (`metrics_*.csv`) containing per-request performance data.

**Generate Comparative Plots**:
**Architecture**:
The system follows a 2-layer architecture to ensure reproducibility:
1.  **Host Wrapper** (`run_experiments.sh`): Prepares the environment and launches the Docker container.
2.  **Container Entrypoint** (`scripts/docker_entrypoint.sh`): Sets up vLLM, runs the benchmark, and triggers analysis scripts.

## 🤝 Contributing
1.  Fork the repository.
2.  Add new offloading configurations in `configs/`.
3.  Add new analysis modules in `analysis/`.
4.  Submit a Pull Request.

## ⚙️ Configuration
This will output:
*   `comparison_ttft.png`: Time to First Token vs Sequence Length.
*   `comparison_e2e.png`: End-to-End Latency vs Sequence Length.

## 🔍 Profiling Details

*   **GPU Utilization**: The script automatically captures `nvidia-smi dmon` output to `pcie_stats_*.csv`. Use this to correlate PCIe bandwidth spikes with cache transfer events.
*   **Disk I/O**: For the `disk` tier, monitor `disk_io_stats_*.csv` to see the read/write throughput impact of LMCache.

## ⚙️ Configuration

Modify files in `configs/` to tune LMCache behavior:
*   `chunk_size`: Controls the granularity of cache transfer (default: 256).
*   `max_local_cache_size`: Limit for CPU/Disk usage.
*   `remote_url`: For Redis/Network offloading.
