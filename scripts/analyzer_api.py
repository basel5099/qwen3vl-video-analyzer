#!/usr/bin/env python3
"""Async public video-analysis API wrapping the local vLLM server.

The RunPod/Cloudflare proxy kills HTTP requests after ~100 s, so analysis is
asynchronous:

POST /analyze  {"video_url": "https://..."} or {"video_path": "..."}
               optional: {"chunk_seconds": 600, "limit_seconds": 0}
               -> {"job_id": "...", "status": "queued"} immediately
GET  /result/{job_id} -> {"status": "queued|running|done|error", ...result}
GET  /health
Auth on both endpoints: Authorization: Bearer $LAB_API_KEY.
Results also persisted to /root/lab/jobs/<id>.json (survive API restarts).
"""
import json
import os
import subprocess
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

LAB = Path(os.environ.get("LAB_HOME", "/root/lab"))
VIDEOS = LAB / "videos"
JOBS_DIR = LAB / "jobs"
JOBS_DIR.mkdir(exist_ok=True)
FFMPEG = LAB / "bin" / "ffmpeg"
API_KEY = os.environ.get("LAB_API_KEY", "")

try:
    _vram_mb = int(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        text=True).splitlines()[0])
except Exception:
    _vram_mb = 999999
# Small cards run a smaller vLLM context (see launch.sh) -> smaller chunks.
# Chunks are sized to nearly FILL the context (measured: ~100 tok/frame @360p,
# ~300 tok/frame @720p): richer temporal context per chunk and fewer chunks
# (each chunk pays a fixed ~9 s JSON-generation cost).
DEFAULT_CHUNK = 360 if _vram_mb < 40000 else 600
DEFAULT_CHUNK_HIGH = 90 if _vram_mb < 40000 else 150

app = FastAPI(title="qwen3vl-video-analyzer")
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
_coherent_counter = {"n": 0}  # round-robins coherent jobs across GPUs


class AnalyzeRequest(BaseModel):
    video_url: str | None = None
    video_path: str | None = None
    chunk_seconds: int | None = None
    limit_seconds: int = 0
    quality: str = "low"  # "low" = 360p (fast), "high" = 720p (fine detail)
    coherent: bool = False  # sequential chain: each chunk sees the previous
    #   chunk's summary (better narrative continuity; ~2x slower per video,
    #   but concurrent coherent jobs are pinned to different GPUs)


def check_auth(authorization: str | None):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def set_job(job_id: str, **fields):
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(fields)
        snapshot = dict(jobs[job_id])
    (JOBS_DIR / f"{job_id}.json").write_text(json.dumps(snapshot))


def normalize(src: Path, quality: str = "low") -> Path:
    scale = "1280:-2" if quality == "high" else "640:-2"
    dst = src.with_name(src.stem + "_norm.mp4")
    proc = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vf", f"fps=1,scale={scale}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-g", "15", "-an", str(dst)],
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"normalize failed: {proc.stderr[-800:]}")
    return dst


def run_job(job_id: str, req: AnalyzeRequest):
    t0 = time.time()
    video = norm = None
    cleanup = False
    try:
        if req.video_path:
            video = Path(req.video_path)
            if not video.is_file():
                raise RuntimeError(f"no such file: {video}")
        else:
            video = VIDEOS / f"api_{job_id}.mp4"
            cleanup = True
            set_job(job_id, status="running", stage="downloading")
            dl = subprocess.run(
                ["curl", "-sL", "--max-time", "1800", "-o", str(video),
                 req.video_url], capture_output=True)
            if dl.returncode != 0 or not video.is_file() or video.stat().st_size == 0:
                raise RuntimeError("video download failed")

        set_job(job_id, status="running", stage="normalizing")
        norm = normalize(video, req.quality)

        set_job(job_id, status="running", stage="analyzing")
        chunk_s = req.chunk_seconds or (
            DEFAULT_CHUNK_HIGH if req.quality == "high" else DEFAULT_CHUNK)
        env = dict(os.environ)
        if req.coherent:
            with jobs_lock:
                idx = _coherent_counter["n"]
                _coherent_counter["n"] += 1
            env["LAB_COHERENT"] = "1"
            env["LAB_BACKEND_IDX"] = str(idx)
        proc = subprocess.run(
            [str(LAB / ".venv/bin/python"), str(LAB / "mapreduce_prod.py"),
             str(norm), str(chunk_s), str(req.limit_seconds)],
            capture_output=True, text=True, timeout=7200, env=env,
        )
        if proc.returncode != 0:
            raise RuntimeError("analysis failed: "
                               + proc.stdout[-1500:] + proc.stderr[-1500:])
        result_path = LAB / f"prod_{norm.stem}.json"
        result = json.loads(result_path.read_text())
        result_path.unlink(missing_ok=True)
        result["api_wall_s"] = round(time.time() - t0, 1)
        set_job(job_id, status="done", stage="done", result=result)
    except Exception as e:
        set_job(job_id, status="error", error=str(e)[:3000],
                api_wall_s=round(time.time() - t0, 1))
    finally:
        if norm is not None:
            norm.unlink(missing_ok=True)
        if cleanup and video is not None:
            video.unlink(missing_ok=True)


@app.get("/health")
def health():
    return {"status": "ok", "model": "Qwen/Qwen3-VL-8B-Instruct", "mode": "async"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    if not req.video_url and not req.video_path:
        raise HTTPException(status_code=422, detail="provide video_url or video_path")
    job_id = uuid.uuid4().hex[:12]
    set_job(job_id, id=job_id, status="queued", created=time.time())
    threading.Thread(target=run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id, "status": "queued",
            "poll": f"/result/{job_id}"}


@app.get("/result/{job_id}")
def result(job_id: str, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        f = JOBS_DIR / f"{job_id}.json"
        if f.is_file():
            job = json.loads(f.read_text())
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job_id")
    return job
