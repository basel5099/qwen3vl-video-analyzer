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

def _detect_vram_mb() -> int:
    try:
        return int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"], text=True).splitlines()[0])
    except Exception:
        pass
    try:  # AMD: rocm-smi vram total in bytes
        out = subprocess.check_output(
            ["rocm-smi", "--showmeminfo", "vram", "--csv"], text=True)
        for line in out.splitlines():
            parts = line.split(",")
            if len(parts) >= 2 and parts[1].strip().isdigit():
                return int(int(parts[1]) / 1024 / 1024)
    except Exception:
        pass
    return 999999


_vram_mb = _detect_vram_mb()
# Small cards run a smaller vLLM context (see launch.sh) -> smaller chunks.
# Chunks are sized to nearly FILL the context (measured: ~100 tok/frame @360p,
# ~300 tok/frame @720p): richer temporal context per chunk and fewer chunks
# (each chunk pays a fixed ~9 s JSON-generation cost).
if _vram_mb < 40000:          # RTX 5090-class (45K ctx)
    # 330 not 360: keyframe drift (+15s) + coherent prefix must stay <45K
    DEFAULT_CHUNK, DEFAULT_CHUNK_HIGH = 330, 80
elif _vram_mb < 150000:       # H200/PRO6000-class (150K ctx)
    DEFAULT_CHUNK, DEFAULT_CHUNK_HIGH = 600, 140
else:                         # MI300X-class 192GB (256K ctx, 500M px budget)
    DEFAULT_CHUNK, DEFAULT_CHUNK_HIGH = 1500, 420

app = FastAPI(title="qwen3vl-video-analyzer")
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
_coherent_counter = {"n": 0}  # round-robins coherent jobs across GPUs

# Two-stage pipeline (Basel's design):
#   submit queue -> PREP workers (CPU: download+normalize, the slow part)
#   -> bounded READY queue -> GPU workers (pinned per backend, analysis only).
# GPUs never wait for ffmpeg: the next videos are pre-normalized in parallel.
import queue as _queue
_prep_queue: "_queue.Queue[tuple]" = _queue.Queue()
_ready_queue: "_queue.Queue[tuple]" = _queue.Queue(
    maxsize=int(os.environ.get("LAB_PREFETCH_DEPTH", "3")))
MAX_CONCURRENT = int(os.environ.get("LAB_MAX_CONCURRENT", "2"))
PREP_WORKERS = int(os.environ.get("LAB_PREP_WORKERS", "2"))


def _prep_worker():
    while True:
        job_id, req = _prep_queue.get()
        try:
            prepared = prep_job(job_id, req)
            if prepared is not None:
                set_job(job_id, status="queued", stage="prepared-waiting-gpu")
                _ready_queue.put((job_id, req) + prepared)  # blocks when full
        finally:
            _prep_queue.task_done()


def _gpu_worker(worker_idx: int):
    while True:
        job_id, req, video, norm, cleanup = _ready_queue.get()
        try:
            analyze_job(job_id, req, video, norm, cleanup,
                        backend_idx=worker_idx)
        finally:
            _ready_queue.task_done()


def _start_workers():
    for _ in range(PREP_WORKERS):
        threading.Thread(target=_prep_worker, daemon=True).start()
    for i in range(MAX_CONCURRENT):
        threading.Thread(target=_gpu_worker, args=(i,), daemon=True).start()


_start_workers()


class AnalyzeRequest(BaseModel):
    video_url: str | None = None
    video_path: str | None = None
    chunk_seconds: int | None = None
    limit_seconds: int = 0
    quality: str = "low"  # "low" = 360p (fast), "high" = 720p (fine detail)
    prompt: str | None = None  # optional user focus/question: steers every
    #   chunk's attention and adds a direct "user_answer" in the result
    coherent: bool = False  # sequential chain: each chunk sees the previous
    #   chunk's summary (better narrative continuity; ~2x slower per video,
    #   but concurrent coherent jobs are pinned to different GPUs)
    output_schema: dict | list | None = None  # exact JSON template the final
    #   answer must fill LITERALLY (same keys/nesting, no extras) — for
    #   callers with a strict output contract


def check_auth(authorization: str | None):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def set_job(job_id: str, **fields):
    with jobs_lock:
        jobs.setdefault(job_id, {}).update(fields)
        snapshot = dict(jobs[job_id])
    (JOBS_DIR / f"{job_id}.json").write_text(json.dumps(snapshot))


def normalize(src: Path, quality: str = "low") -> Path:
    # Fit into a WxH box preserving aspect: caps pixels/frame for ANY
    # orientation (square/vertical videos blew past the context with
    # width-only scaling — a 720x720 became 640x640 = +74% tokens/frame).
    box = "1280:720" if quality == "high" else "640:360"
    scale = f"scale={box}:force_original_aspect_ratio=decrease:force_divisible_by=2"
    dst = src.with_name(src.stem + "_norm.mp4")
    proc = subprocess.run(
        [str(FFMPEG), "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
         "-vf", f"fps=1,{scale}", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "23", "-g", "15", "-an", str(dst)],
        capture_output=True, text=True, timeout=3600,
    )
    if proc.returncode != 0 or not dst.is_file() or dst.stat().st_size == 0:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"normalize failed: {proc.stderr[-800:]}")
    return dst


def prep_job(job_id: str, req: AnalyzeRequest):
    """CPU stage: download + normalize. Returns (video, norm, cleanup) or
    None on failure (job already marked error)."""
    t0 = time.time()
    video = None
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
        return video, norm, cleanup
    except Exception as e:
        set_job(job_id, status="error", error=str(e)[:3000],
                api_wall_s=round(time.time() - t0, 1))
        if cleanup and video is not None:
            video.unlink(missing_ok=True)
        return None


def analyze_job(job_id: str, req: AnalyzeRequest, video, norm, cleanup,
                backend_idx: int | None = None):
    t0 = time.time()
    try:
        set_job(job_id, status="running", stage="analyzing")
        chunk_s = req.chunk_seconds or (
            DEFAULT_CHUNK_HIGH if req.quality == "high" else DEFAULT_CHUNK)
        env = dict(os.environ)
        if req.prompt:
            env["LAB_USER_PROMPT"] = req.prompt[:12000]
        if req.output_schema is not None:
            env["LAB_OUTPUT_SCHEMA"] = json.dumps(req.output_schema)[:12000]
        if req.coherent:
            if backend_idx is None:
                with jobs_lock:
                    backend_idx = _coherent_counter["n"]
                    _coherent_counter["n"] += 1
            env["LAB_COHERENT"] = "1"
            env["LAB_BACKEND_IDX"] = str(backend_idx)
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


def _gpu_stats():
    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,"
             "memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        for ln in out.strip().splitlines():
            i, u, mu, mt, t, p = [x.strip() for x in ln.split(",")]
            gpus.append({"index": int(i), "util": int(u), "mem_used": int(mu),
                         "mem_total": int(mt), "temp": int(t),
                         "power": float(p)})
    except Exception:
        pass
    return gpus


@app.get("/stats")
def stats():
    done = err = running = queued = 0
    video_seconds = 0.0
    total_moments = total_tokens = 0
    recent = []
    for f in sorted(JOBS_DIR.glob("*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        s = j.get("status")
        if s == "done":
            done += 1
            r = j.get("result") or {}
            video_seconds += float(r.get("duration_s") or 0)
            total_tokens += int(r.get("map_prompt_tokens") or 0)
            m = (r.get("merged_analysis") or {}).get("notable_moments") or []
            total_moments += len(m)
            if len(recent) < 6:
                recent.append({"video": r.get("video", "?"),
                               "wall_s": r.get("api_wall_s")})
        elif s == "error":
            err += 1
        elif s == "running":
            running += 1
        elif s == "queued":
            queued += 1
    try:
        load = os.getloadavg()[0]
        ncpu = os.cpu_count() or 1
        cpu_pct = min(100, round(load / ncpu * 100))
    except Exception:
        cpu_pct = -1
    mem = {}
    try:
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = int(v.strip().split()[0]) // 1024
    except Exception:
        pass
    return {
        "gpus": _gpu_stats(), "cpu_pct": cpu_pct,
        "ram_used_mb": (mem.get("MemTotal", 0) - mem.get("MemAvailable", 0)),
        "ram_total_mb": mem.get("MemTotal", 0),
        "jobs": {"done": done, "error": err, "running": running,
                 "queued": queued},
        "queues": {"prep_waiting": _prep_queue.qsize(),
                   "ready_for_gpu": _ready_queue.qsize()},
        "video_hours_done": round(video_seconds / 3600, 2),
        "total_moments": total_moments,
        "total_tokens_m": round(total_tokens / 1e6, 2),
        "ts": time.time(),
    }


DASH_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>qwen3vl analyzer</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{background:#0d1117;color:#e6edf3;font-family:system-ui,sans-serif;margin:0;padding:16px}
h1{font-size:18px;margin:0 0 14px;color:#7ee787}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px}
.k{color:#8b949e;font-size:12px}.v{font-size:26px;font-weight:700}
.bar{height:10px;background:#21262d;border-radius:5px;margin:6px 0;overflow:hidden}
.fill{height:100%;background:#238636;transition:width .5s}
.hot{background:#da3633}.warn{background:#d29922}
canvas{width:100%;height:44px}
.done{color:#7ee787}.err{color:#ff7b72}.run{color:#79c0ff}.wait{color:#d2a8ff}
small{color:#8b949e}
</style></head><body>
<h1>&#9889; qwen3vl video-analysis factory</h1>
<div class="grid" id="cards"></div>
<script>
const hist = {};
function bar(p, cls){return `<div class="bar"><div class="fill ${cls}" style="width:${p}%"></div></div>`}
function spark(id, arr){const c=document.getElementById(id); if(!c) return; const x=c.getContext('2d');
 const w=c.width=c.clientWidth, h=c.height=44; x.clearRect(0,0,w,h); x.beginPath(); x.strokeStyle='#58a6ff';
 arr.forEach((v,i)=>{const px=i/(Math.max(arr.length-1,1))*w, py=h-(v/100)*h; i?x.lineTo(px,py):x.moveTo(px,py)}); x.stroke()}
async function tick(){
 try{
  const s = await (await fetch('/stats')).json();
  let html = '';
  s.gpus.forEach(g=>{
   hist['g'+g.index] = (hist['g'+g.index]||[]).concat([g.util]).slice(-90);
   const tcls = g.temp>=80?'hot':(g.temp>=70?'warn':'');
   html += `<div class="card"><div class="k">GPU ${g.index}</div>
    <div class="v">${g.util}%</div>${bar(g.util,'')}
    <small>${g.temp}&deg;C &middot; ${Math.round(g.power)}W &middot; ${(g.mem_used/1024).toFixed(1)}/${(g.mem_total/1024).toFixed(1)} GB</small>
    ${bar(Math.round(g.temp/90*100), tcls)}
    <canvas id="sg${g.index}"></canvas></div>`;
  });
  html += `<div class="card"><div class="k">CPU</div><div class="v">${s.cpu_pct}%</div>${bar(s.cpu_pct,'')}
   <small>RAM ${(s.ram_used_mb/1024).toFixed(1)}/${(s.ram_total_mb/1024).toFixed(1)} GB</small>${bar(Math.round(s.ram_used_mb/Math.max(s.ram_total_mb,1)*100),'')}</div>`;
  html += `<div class="card"><div class="k">jobs</div>
   <div class="v"><span class="done">${s.jobs.done}&#10003;</span> <span class="err">${s.jobs.error}&#10007;</span></div>
   <small><span class="run">${s.jobs.running} running</span> &middot; <span class="wait">${s.jobs.queued} waiting</span></small><br>
   <small>prep queue: ${s.queues.prep_waiting} &middot; ready-for-GPU: ${s.queues.ready_for_gpu}</small></div>`;
  html += `<div class="card"><div class="k">processed</div><div class="v">${s.video_hours_done} h</div>
   <small>${s.total_moments} moments &middot; ${s.total_tokens_m}M visual tokens</small></div>`;
  document.getElementById('cards').innerHTML = html;
  s.gpus.forEach(g=>spark('sg'+g.index, hist['g'+g.index]));
 }catch(e){}
}
setInterval(tick, 2000); tick();
</script></body></html>"""


@app.get("/dashboard")
def dashboard():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(DASH_HTML)


@app.post("/analyze")
def analyze(req: AnalyzeRequest, authorization: str | None = Header(default=None)):
    check_auth(authorization)
    if not req.video_url and not req.video_path:
        raise HTTPException(status_code=422, detail="provide video_url or video_path")
    job_id = uuid.uuid4().hex[:12]
    set_job(job_id, id=job_id, status="queued", created=time.time(),
            queue_position=_prep_queue.qsize())
    _prep_queue.put((job_id, req))
    return {"job_id": job_id, "status": "queued",
            "queue_position": _prep_queue.qsize(),
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
