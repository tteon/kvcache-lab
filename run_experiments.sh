#!/bin/bash
set -e

# Default Arguments
WORKLOAD=${1:-agent}
TIER=${2:-cpu}
IMAGE_NAME="kvcache-experiment"

echo "=== KV Cache Offloading Experiment Runner ==="
echo "Workload: $WORKLOAD"
echo "Tier:     $TIER"
echo "Image:    $IMAGE_NAME"
echo "---------------------------------------------"

# Check if image exists, otherwise build it
if [[ "$(docker images -q $IMAGE_NAME 2> /dev/null)" == "" ]]; then
  echo "Image '$IMAGE_NAME' not found. Building..."
  docker build -t $IMAGE_NAME .
fi

# Run the experiment container
# We mount the current directory to /app so results are saved locally
echo "Launching container..."
docker run --rm --gpus all --ipc=host --cap-add=SYS_ADMIN \
    -v $(pwd):/app \
    $IMAGE_NAME \
    /app/scripts/docker_entrypoint.sh $WORKLOAD $TIER

echo "---------------------------------------------"
echo "Experiment complete. Check results/ directory."
