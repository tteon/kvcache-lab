# vLLM Serving with LMCache

This directory contains scripts to deploy a Dockerized vLLM server integrated with **LMCache** for KV cache offloading, along with comprehensive monitoring and profiling tools.

## 🚀 Getting Started

### 1. Prerequisites
*   **NVIDIA GPU** (Tested on A100/H100)
*   **Docker** with NVIDIA Container Toolkit
*   **Environment Variables**: Ensure a `.env` file exists in the project root with:
    ```bash
    OPENAI_API_KEY=sk-...
    HF_TOKEN=hf_...
    ```

### 2. Start the Server
Run the `serve_llama.sh` script. It handles Docker setup, mounts, and metric collection automatically.

```bash
cd serving
./serve_llama.sh [options]
```

**Available Options:**
*   `--model <name>`: HuggingFace model hub ID (Default: `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`)
*   `--port <port>`: Serving port (Default: `8000`)
*   `--max-len <int>`: Max model context length (Default: `8192`)
*   `--gpu-util <float>`: GPU memory utilization (Default: `0.75`)
*   `--quant <str>`: Quantization method (Default: `awq`)
*   `--dtype <str>`: Data type (Default: `half`)

**Example:**
```bash
./serve_llama.sh --model "meta-llama/Meta-Llama-3-8B-Instruct" --port 9000
```

### 3. Profiling & Monitoring
The script automatically starts background monitoring processes. Logs are saved to `serving_logs/` in this directory.

*   **`lmcache_stats.csv`**: LMCache specific metrics (Cache Hits, Queries, Usage).
*   **`pcie_stats.csv`**: GPU PCIe bandwidth utilization (`nvidia-smi dmon`).
*   **`disk_io_stats.csv`**: System disk I/O (requires `dstat` on host).
*   **`vllm_metrics.log`**: Raw Prometheus metrics dump.

**Nsight Systems Profiling**:
To enable deep profiling with Nsight Systems, set the environment variable:
```bash
ENABLE_PROFILE=true ./serve_llama.sh
```
Traces will be saved to `profiles/` in the project root (or relative to script execution).

### 4. Run Inference Client
A Python client is provided to test the server and demonstrate multi-turn chat.

```bash
# Install dependencies
pip install openai python-dotenv

# Run client
python client.py
```
