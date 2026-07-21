# qwen3vl-video-analyzer

Self-hosted, Gemini-style **long-video analysis API** built on
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) (Apache-2.0)
and [vLLM](https://github.com/vllm-project/vllm). Send a video of **any length and
any codec**; get back structured JSON: summary, materials/objects, actions,
timestamped notable moments, and an authenticity impression.

Runs on your own GPU (one RTX 5090 is enough) or on a RunPod pod via a
one-click self-installing template. No cloud AI APIs involved — your videos
never leave your machine.

## How it works

```
client ──POST /analyze──▶ FastAPI (:8100, async jobs, Bearer auth)
                             │ download → normalize (ffmpeg → h264/360p/1fps)
                             │ split into chunks (ffmpeg stream-copy)
                             ├──chunk──▶ vLLM #1 (:8101, GPU 0)   ┐ round-robin,
                             ├──chunk──▶ vLLM #2 (:8102, GPU 1)   ┘ auto-detected
                             │ merge + code-side global timestamps
client ◀─GET /result/{id}──  full JSON
```

- **Map-reduce beats giant context for long video**: each chunk is analyzed at
  full frame density with correct temporal grounding, then merged. A 1M-token
  context can't fit an hour of video through the vision encoder anyway (we tried,
  on an H200).
- **Any codec**: inputs are normalized first (AV1/VP9/HEVC all fine).
- **VRAM-adaptive**: `launch.sh` detects the card. ≥48 GB → 150K ctx, 10-min
  chunks. 32 GB (RTX 5090) → 45K ctx, 3-min chunks. Same scripts everywhere.
- **Async API**: `POST /analyze` returns a `job_id` instantly;
  poll `GET /result/{job_id}`. (Required behind RunPod's proxy, which kills
  HTTP requests after ~100 s.)

## Measured performance (Qwen3-VL-8B, 360p @ 1 fps)

| Hardware | 6-min video | 62-min video | 104-min video |
|---|---|---|---|
| 2× RTX 5090 (local) | **13.6 s** | **104.5 s** | **172.3 s** (35 chunks, 170 moments) |
| 1× RTX 5090 (local) | 23.3 s | 182.8 s | — |
| H200 (RunPod, $3.59/hr) | 19.5 s | — | 143.5 s |
| RTX A6000 (RunPod, $0.49/hr) | 70.6 s | — | — |

## Quickstart — local (Linux, NVIDIA GPU, driver ≥ 12.8)

```bash
export LAB_HOME=$HOME/qwen3vl-lab
mkdir -p $LAB_HOME/videos $LAB_HOME/bin && cd $LAB_HOME

# 1. env + vLLM (needs Python 3.12; uses uv)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python vllm fastapi uvicorn
#   (driver older than CUDA 13? see "CUDA wheel matching" below)

# 2. static ffmpeg (no root needed)
curl -sL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
  | tar xJ -C bin --strip-components=1 --wildcards '*/ffmpeg' '*/ffprobe'

# 3. model + scripts
.venv/bin/hf download Qwen/Qwen3-VL-8B-Instruct
cp <this-repo>/scripts/{launch.sh,mapreduce_prod.py,analyzer_api.py} $LAB_HOME/

# 4. serve — one vLLM per GPU, then the API
CUDA_VISIBLE_DEVICES=0 nohup bash launch.sh > serve-gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 PORT=8102 nohup bash launch.sh > serve-gpu1.log 2>&1 &   # optional 2nd GPU
export LAB_API_KEY=$(openssl rand -hex 24) && echo "API key: $LAB_API_KEY"
nohup .venv/bin/uvicorn analyzer_api:app --app-dir $LAB_HOME \
  --host 0.0.0.0 --port 8100 > api.log 2>&1 &
```

Wait for `curl :8101/health` (and `:8102`) to return 200 (~3-4 min), then:

```bash
curl -X POST http://localhost:8100/analyze \
  -H "Authorization: Bearer $LAB_API_KEY" -H "Content-Type: application/json" \
  -d '{"video_url": "https://example.com/video.mp4"}'
# -> {"job_id": "...", "status": "queued"}
curl http://localhost:8100/result/<job_id> -H "Authorization: Bearer $LAB_API_KEY"
```

Optional request fields: `chunk_seconds` (default adapts to the card),
`limit_seconds` (analyze only the first N seconds), `video_path` (file already
on the server, under `$LAB_HOME/videos`).

## Quickstart — RunPod (one-click pods)

```bash
export RUNPOD_API_KEY=rpa_...                   # runpod.io -> Settings
export LAB_API_KEY=$(openssl rand -hex 24)
export SSH_PUBKEY="$(cat ~/.ssh/id_ed25519.pub)"
python runpod/make_template.py
```

Then in the RunPod console: **Deploy → any GPU → template
`qwen3vl-video-analyzer`**. The pod installs everything itself (~10-15 min)
and serves the API at `https://<podId>-8100.proxy.runpod.net`. The vision
model finishes loading a few minutes after `/health` first responds — retry
the first job if it errors.

Cost reference: a 1-hour video ≈ $0.05–0.09 depending on the GPU.

## Quickstart — SaladCloud (cheapest: consumer GPUs at home)

Salad rents idle gaming PCs (RTX 5090 32GB at $0.25–0.45/hr — a quarter of
datacenter prices). Measured there: a 10-min video E2E in 41 s; a 30-min video
in **coherent** mode in 68.6 s ≈ **~1 cent per half hour**.

Deploy `docker/salad/` (built as `ghcr.io/<you>/qwen3vl-analyzer-salad`) as a
Container Group: 1 replica, GPU class RTX 5090, 8 vCPU / 16 GB RAM / 50 GB
storage, Container Gateway → port 8100 (auth off — the API has its own Bearer
key), startup probe HTTP `/health` (`failure_threshold` caps at 20, `headers`
required). The entrypoint starts the API **before** the model download so the
probe passes within Salad's ~15-min probe budget.

Salad nodes run containers under **WSL on Windows**, which changes the CUDA
rules — these cost a full night to learn:

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is fatal** — it needs
  CUDA VMM APIs that WSL doesn't provide. Every engine start dies with
  `CUDA driver error: device not ready` (plain torch matmul works, which makes
  it look like anything but an allocator setting). Unset it.
- **vLLM ≥ 0.25 needs UVA** for its V2 model runner (`UVA is not available`).
  Pin `vllm/vllm-openai:v0.11.2` (newest line with Qwen3-VL support that runs
  under WSL; Salad's own vLLM recipe pins v0.10.1 for the same reason) and set
  `VLLM_USE_V2_MODEL_RUNNER=0`.
- **WSL taxes VRAM**: the 32 GB profile that fits at `--gpu-memory-utilization
  0.65` elsewhere needs `0.85` here, with the pixel budget back at 45M
  (115M leaves no room for KV cache blocks).
- The gateway only routes to apps listening on **IPv6** — uvicorn binds `::`.
- The group snapshots the image reference at creation: pushing a new `:latest`
  does NOT reach new nodes. Update the image with **stop → PATCH → start**
  (PATCH while running returns 400/no-op).
- Nodes are a lottery (owner may be gaming on the GPU; some have broken
  gateway networking; egress varies — one node couldn't reach Google Storage
  at all). The fix is always **reallocate** — each fresh node pulls the image
  and model on residential internet (~10–40 min).

Bandwidth-friendly batch pattern: pre-normalize locally
(`ffmpeg -vf "fps=1,scale=640:360:force_original_aspect_ratio=decrease:force_divisible_by=2" -g 15 -an`
turns a 650 MB hour into ~40 MB), `scp` the small file to the node, then
`POST /analyze {"video_path": "...", "coherent": true}` — the node computes
instead of downloading.

## Hard-won notes (so you don't re-learn them)

- **CUDA wheel matching**: the default vLLM wheel is built for CUDA 13. On
  hosts with driver 12.9 install in two steps: `uv pip install vllm`, then
  `uv pip install "vllm @ <github release +cu129 wheel url>" --torch-backend=cu128
  --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio`.
- **Blackwell (sm120)**: flashinfer mis-detects the arch ("requires sm75+").
  `launch.sh` handles it (`FLASHINFER_CUDA_ARCH_LIST=12.0a`, sampler JIT off).
- **The video token budget**: Qwen3-VL's processor silently compresses ANY
  video to ~15-24K tokens via `size.longest_edge` (total-pixel budget). We
  raise it via `--mm-processor-kwargs`. `total_pixels` and per-request
  `mm_processor_kwargs` are silently ignored by vLLM 0.25.
- **Timestamps**: never let the model merge chunk timestamps — it compresses
  the timeline. `mapreduce_prod.py` globalizes them in code.
- **Don't give KV cache everything**: the vision encoder allocates outside
  vLLM's reservation; `--gpu-memory-utilization 0.9` OOMs on hour-long videos.
- **Killing vLLM**: `pkill -f "vllm serve"` leaves the `EngineCore` process
  holding all VRAM. Kill it too, and verify `nvidia-smi` shows 0 MiB before
  relaunching.

## License

Apache-2.0 (same as Qwen3-VL and vLLM). Model weights come from
`Qwen/Qwen3-VL-8B-Instruct` under their own license.
