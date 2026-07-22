#!/bin/bash
# AMD (ROCm) vLLM launcher for Qwen3-VL-8B — big-card profile (MI300X 192GB).
# No nvidia-smi / flashinfer here; vLLM comes preinstalled in the ROCm image.
LAB="${LAB_HOME:-/root/lab}"
cd "$LAB"
export MIOPEN_USER_DB_PATH=/tmp/miopen
export MIOPEN_FIND_MODE=FAST
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
# MI300X-class (192GB): full native 256K ctx + fp8 KV.
# Pixel budget 280M — 500M reliably kills the GPU with a memory access fault
# during large-request processing (measured July 21+22 2026); 280M passes.
MAXLEN=${1:-262144}

exec vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --port "${PORT:-8101}" --max-model-len "$MAXLEN" --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.50 \
  --allowed-local-media-path "$LAB/videos" \
  --limit-mm-per-prompt "{\"video\": 1}" \
  --mm-processor-kwargs "{\"fps\": 1, \"size\": {\"longest_edge\": 280000000, \"shortest_edge\": 4096}}" \
  --media-io-kwargs "{\"video\": {\"num_frames\": -1}}"
