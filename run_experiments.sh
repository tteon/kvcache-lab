#!/bin/bash
set -e


# Default Configuration
WORKLOAD="agent"
TIER="cpu"
IMAGE_NAME="kvcache-experiment"
GPU_MEM_UTIL=0.90
MAX_MODEL_LEN=5000
DTYPE="half"
CHUNK_SIZE=256

# Argument Parsing
while [[ $# -gt 0 ]]; do
  case $1 in
    --workload)
      WORKLOAD="$2"
      shift 2
      ;;
    --tier)
      TIER="$2"
      shift 2
      ;;
    --gpu-mem-util)
      GPU_MEM_UTIL="$2"
      shift 2
      ;;
    --max-model-len)
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --dtype)
      DTYPE="$2"
      shift 2
      ;;
    --chunk-size)
      CHUNK_SIZE="$2"
      shift 2
      ;;
    *)
      # Backwards compatibility for positional args
      if [ -z "$WORKLOAD_ARG_SET" ]; then
        WORKLOAD="$1"
        WORKLOAD_ARG_SET=true
      elif [ -z "$TIER_ARG_SET" ]; then
        TIER="$1"
        TIER_ARG_SET=true
      else
        echo "Unknown argument: $1"
        exit 1
      fi
      shift 1
      ;;
  esac
done

echo "=== KV Cache Offloading Experiment Runner ==="
echo "Workload:      $WORKLOAD"
echo "Tier:          $TIER"
echo "GPU Mem Util:  $GPU_MEM_UTIL"
echo "Max Model Len: $MAX_MODEL_LEN"
echo "Dtype:         $DTYPE"
echo "Chunk Size:    $CHUNK_SIZE"
echo "Image:         $IMAGE_NAME"
echo "---------------------------------------------"

# Check if image exists, otherwise build it
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  echo "Image '$IMAGE_NAME' not found. Building..."
  docker build -t $IMAGE_NAME .
fi

# Run the experiment container
echo "Launching container..."
docker run --rm --gpus all --ipc=host --cap-add=SYS_ADMIN \
    -v $(pwd):/app \
    -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
    -e MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    -e DTYPE="$DTYPE" \
    -e CHUNK_SIZE="$CHUNK_SIZE" \
    $IMAGE_NAME \
    /app/scripts/docker_entrypoint.sh "$WORKLOAD" "$TIER"

echo "---------------------------------------------"
echo "Experiment complete. Check results/ directory."
