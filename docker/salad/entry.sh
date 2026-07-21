#!/bin/bash
# Entrypoint for the SaladCloud image: async API first, then vLLM.
# Salad's Container Gateway requires the app to listen on IPv6 -> uvicorn
# binds "::" (dual-stack). The API comes up BEFORE the model download so the
# startup probe (max ~15 min budget on Salad) passes within seconds; jobs
# submitted before vLLM finishes loading fail cleanly and should be retried
# (same boot-order behavior as the RunPod template).
set -x
export LAB_HOME=/root/lab

mkdir -p $LAB_HOME/videos
# venv shim so analyzer's/launcher's .venv paths work with the system python
mkdir -p $LAB_HOME/.venv/bin
ln -sf "$(command -v python3)" $LAB_HOME/.venv/bin/python
ln -sf "$(command -v vllm)" $LAB_HOME/.venv/bin/vllm 2>/dev/null || true

cd $LAB_HOME && python3 -m uvicorn analyzer_api:app \
  --host '::' --port 8100 > $LAB_HOME/api.log 2>&1 &

# model download (~17GB from HF; Salad nodes are residential — first boot on a
# fresh node takes a while)
hf download Qwen/Qwen3-VL-8B-Instruct || \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct')"

bash $LAB_HOME/launch.sh > $LAB_HOME/serve-8b.log 2>&1 &
sleep 5
tail -f $LAB_HOME/serve-8b.log $LAB_HOME/api.log
