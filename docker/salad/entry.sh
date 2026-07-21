#!/bin/bash
# Entrypoint for the SaladCloud image: async API first, then vLLM.
# Salad's Container Gateway requires the app to listen on IPv6 -> uvicorn
# binds "::" (dual-stack). The API comes up BEFORE the model download so the
# startup probe (max ~15 min budget on Salad) passes within seconds; jobs
# submitted before vLLM finishes loading fail cleanly and should be retried
# (same boot-order behavior as the RunPod template).
set -x
export LAB_HOME=/root/lab
# Salad's sandboxed runtime has no CUDA UVA -> vLLM's V2 model runner
# crashes with "UVA is not available"; force the V1 runner.
export VLLM_USE_V2_MODEL_RUNNER=0
# WSL host: no CUDA VMM (expandable_segments is fatal) and VRAM is taxed —
# the proven Salad profile is util 0.85 with the 45M pixel budget.
export LAB_ALLOC_CONF=""
export LAB_GPU_UTIL=0.85
export LAB_PX_BUDGET=45000000

# Salad's SSH bridge reads the container's authorized_keys — plant the lab
# public key (plus any PUBLIC_KEY env) so every node is reachable for debugging.
mkdir -p /root/.ssh
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAY16+u4TOd5Xb9/EIw+nWJ+JtFtMkOwsO2n5AOjXFjR qwen3vl-lab' >> /root/.ssh/authorized_keys
if [ -n "$PUBLIC_KEY" ]; then echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; fi
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys

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
