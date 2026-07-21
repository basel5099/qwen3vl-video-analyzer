#!/bin/bash
# qwen3vl-video-analyzer template bootstrap
# Runs as the container command. Idempotent: full install on first boot,
# fast path on resume. Server ends up on port 8100.
set -x
/start.sh &
export PATH=/root/.local/bin:/root/lab/.venv/bin:/usr/local/cuda/bin:$PATH
mkdir -p /root/lab/videos /root/lab/bin

echo "$LAB_LAUNCH_B64" | base64 -d > /root/lab/launch.sh
echo "$LAB_MAPREDUCE_B64" | base64 -d > /root/lab/mapreduce_prod.py
echo "$LAB_API_B64" | base64 -d > /root/lab/analyzer_api.py
chmod +x /root/lab/launch.sh

if [ ! -x /root/lab/.venv/bin/vllm ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  cd /root/lab
  /root/.local/bin/uv venv --python 3.12 .venv
  /root/.local/bin/uv pip install --python .venv/bin/python vllm
  /root/.local/bin/uv pip install --python .venv/bin/python \
    "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.25.1/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    --torch-backend=cu128 --reinstall-package torch \
    --reinstall-package torchvision --reinstall-package torchaudio
  /root/.local/bin/uv pip install --python .venv/bin/python ninja fastapi uvicorn
fi
/root/.local/bin/uv pip install --python /root/lab/.venv/bin/python fastapi uvicorn

if [ ! -x /root/lab/bin/ffprobe ]; then
  curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -o /tmp/ff.tar.xz
  tar xf /tmp/ff.tar.xz -C /root/lab/bin --strip-components=1 --wildcards '*/ffmpeg' '*/ffprobe'
  rm -f /tmp/ff.tar.xz
fi

/root/lab/.venv/bin/hf download Qwen/Qwen3-VL-8B-Instruct

bash /root/lab/launch.sh > /root/lab/serve-8b.log 2>&1 &
/root/lab/.venv/bin/uvicorn analyzer_api:app --app-dir /root/lab \
  --host 0.0.0.0 --port 8100 > /root/lab/api.log 2>&1 &
sleep 5
tail -f /root/lab/serve-8b.log
