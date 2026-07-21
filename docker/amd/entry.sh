#!/bin/bash
# Entrypoint for the AMD fast image: sshd + vLLM (official fast build) + async API.
set -x
export LAB_HOME=/root/lab

# SSH (key from PUBLIC_KEY env, RunPod-style)
mkdir -p /root/.ssh /run/sshd
if [ -n "$PUBLIC_KEY" ]; then echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys; fi
/usr/sbin/sshd || true

mkdir -p $LAB_HOME/videos
# venv shim so analyzer's subprocess path works with the system python
mkdir -p $LAB_HOME/.venv/bin
ln -sf "$(command -v python3)" $LAB_HOME/.venv/bin/python
ln -sf "$(command -v vllm)" $LAB_HOME/.venv/bin/vllm 2>/dev/null || true

# model (cached across restarts on the container disk)
hf download Qwen/Qwen3-VL-8B-Instruct || \
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct')"

bash $LAB_HOME/launch.sh > $LAB_HOME/serve-8b.log 2>&1 &
cd $LAB_HOME && python3 -m uvicorn analyzer_api:app \
  --host 0.0.0.0 --port 8100 > $LAB_HOME/api.log 2>&1 &
sleep 5
tail -f $LAB_HOME/serve-8b.log
