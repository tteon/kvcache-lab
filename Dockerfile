FROM lmcache/vllm-openai:latest

WORKDIR /app

# Install system dependencies for monitoring (dstat, iotop, etc. if needed)
RUN apt-get update && apt-get install -y \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy local code
COPY . /app

# Install python dependencies
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt
RUN pip install --no-cache-dir --break-system-packages pandas matplotlib vllm lmcache

# Expose vLLM port
EXPOSE 8000

# Default command (can be overridden)
CMD ["/bin/bash"]
