# SaladCloud image: official vLLM CUDA build + the analyzer API.
# Target: consumer NVIDIA cards (RTX 5090 32GB primary — same adaptive
# profile as the local dual-5090 lab). No SSH: Salad only exposes the
# Container Gateway port.
FROM vllm/vllm-openai:v0.25.1

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl xz-utils && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir fastapi uvicorn

# static ffmpeg/ffprobe (normalize + chunking); BtbN GitHub build primary
# (johnvansickle.com intermittently serves error pages to CI runners)
RUN mkdir -p /root/lab/bin && \
    ( curl -fsSL --retry 3 --retry-delay 5 \
        https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz \
        -o /tmp/ff.tar.xz && \
      tar xf /tmp/ff.tar.xz -C /root/lab/bin --strip-components=2 \
        --wildcards '*/bin/ffmpeg' '*/bin/ffprobe' ) || \
    ( curl -fsSL --retry 3 --retry-delay 5 \
        https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
        -o /tmp/ff.tar.xz && \
      tar xf /tmp/ff.tar.xz -C /root/lab/bin --strip-components=1 \
        --wildcards '*/ffmpeg' '*/ffprobe' ) && \
    rm -f /tmp/ff.tar.xz && \
    /root/lab/bin/ffmpeg -version | head -1 && /root/lab/bin/ffprobe -version | head -1

COPY scripts/analyzer_api.py scripts/mapreduce_prod.py scripts/launch.sh /root/lab/
COPY docker/salad/entry.sh /root/lab/entry.sh
RUN chmod +x /root/lab/launch.sh /root/lab/entry.sh

ENV LAB_HOME=/root/lab
EXPOSE 8100
ENTRYPOINT ["bash", "/root/lab/entry.sh"]
