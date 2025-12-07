#!/bin/bash
# set -e
set -x

# Auto-install dependencies for analysis
pip install --break-system-packages pandas matplotlib pyparsing > /dev/null 2>&1
echo "Analysis dependencies installed."

# Configuration
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
PORT=8000
API_BASE="http://localhost:$PORT/v1"

# vLLM Configuration (User configurable)
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-5000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-5000}
DTYPE=${DTYPE:-half}
CHUNK_SIZE=${CHUNK_SIZE:-256}

# Arguments
WORKLOAD=$1 # "agent" or "rag"
TIER=$2     # "baseline", "cpu", "disk"

echo "=== Starting Experiment: $WORKLOAD | Tier: $TIER ==="

# Create Results Directory
RESULTS_DIR="/app/results/${WORKLOAD}_${TIER}_$(date +%s)"
PROFILES_DIR="${RESULTS_DIR}/profiles"
PLOTS_DIR="${RESULTS_DIR}/plots"
METRICS_DIR="${RESULTS_DIR}/metrics"
ANALYSIS_DIR="${RESULTS_DIR}/analysis"

mkdir -p "$RESULTS_DIR" "$PROFILES_DIR" "$PLOTS_DIR" "$METRICS_DIR" "$ANALYSIS_DIR"
echo "Results Directory: $RESULTS_DIR"

# Define Nsys Command
NSYS_CMD="nsys profile --trace=cuda,nvtx,osrt,cublas,cudnn --output=${PROFILES_DIR}/report --force-overwrite=true --delay=10 --duration=600"

# LMCache Config Generation
LMCACHE_CONFIG="lmcache_config.yaml"
if [ "$TIER" == "cpu" ]; then
    echo "chunk_size: $CHUNK_SIZE" > $LMCACHE_CONFIG
    echo "local_device: cpu" >> $LMCACHE_CONFIG
    echo "remote_url: null" >> $LMCACHE_CONFIG
    echo "remote_serde: null" >> $LMCACHE_CONFIG
    # Add other params if needed
elif [ "$TIER" == "disk" ]; then
    echo "chunk_size: $CHUNK_SIZE" > $LMCACHE_CONFIG
    echo "local_device: file://local_disk/" >> $LMCACHE_CONFIG
    echo "remote_url: null" >> $LMCACHE_CONFIG
    mkdir -p local_disk
else
    # Baseline or fallback
    echo "chunk_size: $CHUNK_SIZE" > $LMCACHE_CONFIG
    echo "local_device: null" >> $LMCACHE_CONFIG # Disable? Or just don't load?
fi

# Start Monitoring
echo "Starting System Monitoring..."
nvidia-smi dmon -s p -d 1 -c 120 > "${METRICS_DIR}/pcie_stats.csv" &
MONITOR_PID=$!

# Start vLLM with Nsys
echo "Starting vLLM with NVTX profiling..."
# Note: LMCache enablement depends on vLLM integration. Assuming standard vLLM but with LMCache installed/env var? 
# The repo implies using `LMCACHE_CONFIG_FILE` env var.
export LMCACHE_CONFIG_FILE=$LMCACHE_CONFIG
export VLLM_ATTENTION_BACKEND=FLASH_ATTN 
# Need to check actual vllm command args from original script

CMD="python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --max-model-len $MAX_MODEL_LEN \
    --dtype $DTYPE \
    --enforce-eager \
    --port $PORT"

if [ "$TIER" == "baseline" ]; then
    # Unset config for baseline just in case
    unset LMCACHE_CONFIG_FILE
fi

$NSYS_CMD $CMD > "${METRICS_DIR}/vllm_server.log" 2>&1 &
VLLM_PID=$!

echo "vLLM PID: $VLLM_PID"

# Metrics Collection Loop
echo "Starting Metrics Collection..."
while kill -0 $VLLM_PID 2>/dev/null; do
    curl -s "http://localhost:$PORT/metrics" >> "${METRICS_DIR}/vllm_metrics.log"
    sleep 1
done &
METRICS_LOOP_PID=$!

# Wait for readiness
echo "Waiting for vLLM to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:$PORT/health | grep -q "ok"; then
        echo "vLLM is ready!"
        break
    fi
    sleep 5
done

echo "Checking API connectivity..."
curl -v http://localhost:$PORT/v1/models
echo "--------------------------------"

# Run Benchmark
if [ "$WORKLOAD" == "agent" ]; then
    # Agent: moderate context, multi-turn
    echo "Running Agent Workload..."
    python3 benchmark.py \
        --model $MODEL_NAME \
        --prompt-len 1000 \
        --gen-len 200 \
        --num-requests 20 \
        --api-base $API_BASE \
        --label agent_workload_$TIER \
        --device-name "A100" \
        --output-dir "$METRICS_DIR" > "${METRICS_DIR}/benchmark.log" 2>&1
elif [ "$WORKLOAD" == "rag" ]; then
    # RAG: Long context, short gen
    echo "Running RAG Workload..."
    python3 benchmark.py \
        --model $MODEL_NAME \
        --prompt-len 4000 \
        --gen-len 100 \
        --num-requests 10 \
        --api-base $API_BASE \
        --label rag_workload_$TIER \
        --output-dir "$METRICS_DIR" > "${METRICS_DIR}/benchmark.log" 2>&1
fi

echo "Benchmark complete."

# Stop vLLM and cleanup
kill $VLLM_PID || true
kill $MONITOR_PID || true
kill $METRICS_LOOP_PID || true
wait $VLLM_PID || true

echo "Experiment Finished. Reports generated."

# --- Automatic Analysis ---
echo "=== Starting Automatic Analysis ==="

# Fix dependency path issues (pip installs to system, python is venv)
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.12/dist-packages:/usr/lib/python3/dist-packages

# 1. Plot Results
echo "Generating plots..."
python3 /app/analysis/plot_results.py \
    --input "${METRICS_DIR}/metrics_*.csv" \
    --output-prefix "${PLOTS_DIR}/comparison"

# 2. Bottleneck Calculation (Theoretical Speedup)
echo "Calculating theoretical bottlenecks..."
# Assuming some defaults or deriving from args. 
# For RAG (Prompt=4000), we use N=4000. For Agent (Prompt=1000), N=1000.
if [ "$WORKLOAD" == "rag" ]; then
    ANALYSIS_N=4000
else
    ANALYSIS_N=1000
fi

python3 /app/analysis/bottleneck_calculator.py \
    --mode decode \
    --L $ANALYSIS_N \
    --R 50 \
    --T 500 \
    --alpha 0.1 \
    --baseline-dir "$METRICS_DIR" > "${ANALYSIS_DIR}/bottleneck_analysis.txt"

echo "Analysis complete."
echo "Results:"
echo "- Profiles: $PROFILES_DIR"
echo "- Plots:    $PLOTS_DIR"
echo "- Metrics:  $METRICS_DIR"
echo "- Analysis: $ANALYSIS_DIR"
