# AMD fast image: official vLLM ROCm build (fast kernels) + the analyzer API.
FROM vllm/vllm-openai-rocm:latest

RUN apt-get update && \
    apt-get install -y --no-install-recommends openssh-server curl xz-utils && \
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

COPY scripts/analyzer_api.py scripts/mapreduce_prod.py /root/lab/
COPY docker/amd/launch-amd.sh /root/lab/launch.sh
COPY docker/amd/entry.sh /root/lab/entry.sh
RUN chmod +x /root/lab/launch.sh /root/lab/entry.sh

ENV LAB_HOME=/root/lab
EXPOSE 22 8100
ENTRYPOINT ["bash", "/root/lab/entry.sh"]
