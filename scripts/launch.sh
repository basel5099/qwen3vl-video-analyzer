#!/bin/bash
# vLLM launcher for Qwen3-VL-8B video analysis.
# VRAM-adaptive: picks context length / memory split / pixel budget by card size.
# Env: LAB_HOME (default /root/lab), PORT (default 8101). Arg 1 overrides max-model-len.
LAB="${LAB_HOME:-/root/lab}"
cd "$LAB"
export PATH="$LAB/.venv/bin:/usr/local/cuda/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Blackwell (sm120): flashinfer's CUDA-version check is broken -> pin arch, skip sampler JIT
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [ "${CAP%%.*}" -ge 12 ]; then
  export FLASHINFER_CUDA_ARCH_LIST="12.0a"
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi

# Small cards (e.g. RTX 5090 32GB): weights 17G + vision-encoder headroom leave
# room for ~45K ctx only; pixel budget kept low so the profiling pass fits too.
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_MB" -lt 40000 ]; then
  DEF_MAXLEN=45000; UTIL=0.65; BUDGET=45000000
else
  DEF_MAXLEN=150000; UTIL=0.60; BUDGET=150000000
fi
MAXLEN=${1:-$DEF_MAXLEN}

exec .venv/bin/vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port "${PORT:-8101}" --max-model-len "$MAXLEN" --kv-cache-dtype fp8 \
  --gpu-memory-utilization "$UTIL" \
  --allowed-local-media-path "$LAB/videos" \
  --limit-mm-per-prompt "{\"video\": 1}" \
  --mm-processor-kwargs "{\"fps\": 1, \"size\": {\"longest_edge\": $BUDGET, \"shortest_edge\": 4096}}" \
  --media-io-kwargs "{\"video\": {\"num_frames\": -1}}"
