#!/bin/bash
# AMD (ROCm) vLLM launcher for Qwen3-VL-8B — big-card profile (MI300X 192GB).
# No nvidia-smi / flashinfer here; vLLM comes preinstalled in the ROCm image.
LAB="${LAB_HOME:-/root/lab}"
cd "$LAB"
export MIOPEN_USER_DB_PATH=/tmp/miopen
export MIOPEN_FIND_MODE=FAST
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
# MI300X-class (192GB): full native 256K ctx, low util (huge encoder headroom),
# 500M pixel budget -> single chunks up to ~30 min @360p / ~8 min @720p.
MAXLEN=${1:-262144}

exec vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port "${PORT:-8101}" --max-model-len "$MAXLEN" \
  --gpu-memory-utilization 0.45 \
  --allowed-local-media-path "$LAB/videos" \
  --limit-mm-per-prompt "{\"video\": 1}" \
  --mm-processor-kwargs "{\"fps\": 1, \"size\": {\"longest_edge\": 500000000, \"shortest_edge\": 4096}}" \
  --media-io-kwargs "{\"video\": {\"num_frames\": -1}}"
