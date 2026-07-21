#!/usr/bin/env python3
"""Production long-video analysis: map-reduce at 1 fps for Basel's spec
(360p, 1 fps, ~60 min craft videos).

Chunks the video (default 600 s), analyzes each chunk against the local
vLLM server (processor samples at the server-configured fps), then:
- notable_moments timestamps are globalized IN CODE (segment offset added),
  never trusted to the model;
- a final text-only merge produces the overall summary/verdict.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Multi-GPU: one vLLM instance per port; chunks are dispatched round-robin
# across every backend that answers /health (single-GPU boxes find just one).
CANDIDATE_PORTS = [8101, 8102, 8103, 8104]
MODEL = "Qwen/Qwen3-VL-8B-Instruct"


def alive_backends():
    apis = []
    for p in CANDIDATE_PORTS:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{p}/health", timeout=3):
                apis.append(f"http://127.0.0.1:{p}/v1/chat/completions")
        except Exception:
            pass
    return apis or [f"http://127.0.0.1:{CANDIDATE_PORTS[0]}/v1/chat/completions"]
LAB = Path(os.environ.get("LAB_HOME", "/root/lab"))
FFMPEG = LAB / "bin" / "ffmpeg"
FFPROBE = LAB / "bin" / "ffprobe"
CHUNK_DIR = LAB / "videos" / "chunks"

CHUNK_PROMPT = """This is one segment of a longer craft video. Analyze it and return ONLY valid JSON:
{"segment_summary": "2-3 sentences", "materials_and_objects": [], "actions": [], "notable_moments": [{"timestamp": "mm:ss WITHIN THIS SEGMENT", "event": ""}], "craft_type": "", "authenticity_signals": ""}"""

COHERENT_PREFIX = """CONTEXT FROM EARLIER IN THIS VIDEO (already analyzed): {prev}

Analyze THIS segment as a CONTINUATION of that story — refer to established objects/steps where relevant, and note what is NEW or has progressed. """

MERGE_PROMPT = """You are given per-segment JSON analyses of ONE long craft video, in order. Write ONE merged analysis. Do NOT output timestamps (they are handled elsewhere). Return ONLY valid JSON:
{"summary": "3-4 sentences on the full arc", "materials_and_objects": [], "actions": [], "craft_type": "", "authenticity_impression": "", "narrative_arc": ""}

Segment analyses:
"""


def post(payload, timeout=1800, api=None):
    api = api or alive_backends()[0]
    req = urllib.request.Request(api, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def mmss(s):
    return f"{int(s) // 60:02d}:{int(s) % 60:02d}"


def globalize(ts_local, offset_s):
    t = ts_local.strip()
    m3 = re.match(r"(\d+):(\d+):(\d+)$", t)
    m2 = re.match(r"(\d+):(\d+)$", t)
    if m3:
        local = int(m3.group(1)) * 3600 + int(m3.group(2)) * 60 + int(m3.group(3))
    elif m2:
        local = int(m2.group(1)) * 60 + int(m2.group(2))
    else:
        return ts_local
    total = offset_s + local
    h, rem = divmod(total, 3600)
    mn, sec = divmod(rem, 60)
    return f"{h:02d}:{mn:02d}:{sec:02d}"


def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return {"raw": text}


def main():
    video = Path(sys.argv[1])
    chunk_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    limit_seconds = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # 0 = full video
    out_path = LAB / f"prod_{video.stem}.json"

    duration = float(subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", video], capture_output=True, text=True, check=True
    ).stdout.strip())
    if limit_seconds:
        duration = min(duration, limit_seconds)
    print(f"video={video.name} duration_used={duration:.0f}s chunk={chunk_seconds}s")

    CHUNK_DIR.mkdir(exist_ok=True)
    for old in CHUNK_DIR.glob("chunk_*.mp4"):
        old.unlink()
    t0 = time.time()
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", video]
    if limit_seconds:
        cmd += ["-t", str(limit_seconds)]
    cmd += ["-c", "copy", "-an", "-f", "segment",
            "-segment_time", str(chunk_seconds), "-reset_timestamps", "1",
            CHUNK_DIR / "chunk_%03d.mp4"]
    subprocess.run(cmd, check=True)
    chunks = sorted(CHUNK_DIR.glob("chunk_*.mp4"))
    print(f"split into {len(chunks)} chunks in {time.time() - t0:.1f}s")

    apis = alive_backends()
    # Coherent mode: sequential chain, pinned to ONE backend so concurrent
    # jobs (other videos) get the other GPUs. LAB_BACKEND_IDX set by the API.
    coherent = os.environ.get("LAB_COHERENT", "0") == "1"
    if coherent:
        idx = int(os.environ.get("LAB_BACKEND_IDX", "0"))
        apis = [apis[idx % len(apis)]]
    print(f"backends: {len(apis)} coherent={coherent} -> {apis}")

    def analyze_chunk(idx_chunk, prompt_text=CHUNK_PROMPT):
        i, chunk = idx_chunk
        t1 = time.time()
        resp = post({
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"file://{chunk}"}},
                {"type": "text", "text": prompt_text},
            ]}],
            "max_tokens": 2200, "temperature": 0.2,
        }, api=apis[i % len(apis)])
        wall = time.time() - t1
        u = resp["usage"]
        seg = extract_json(resp["choices"][0]["message"]["content"])
        print(f"chunk {i}: wall={wall:.1f}s in={u['prompt_tokens']} out={u['completion_tokens']}")
        return i, seg, u

    if coherent:
        results = []
        prev = ""
        for ic in enumerate(chunks):
            ptxt = (COHERENT_PREFIX.format(prev=prev) + CHUNK_PROMPT) if prev \
                else CHUNK_PROMPT
            i, seg, u = analyze_chunk(ic, ptxt)
            results.append((i, seg, u))
            prev = str(seg.get("segment_summary", ""))[:1500]
    else:
        with ThreadPoolExecutor(max_workers=len(apis)) as pool:
            results = sorted(pool.map(analyze_chunk, enumerate(chunks)))

    seg_reports, all_moments = [], []
    tot_in = tot_out = 0
    for i, seg, u in results:
        off = i * chunk_seconds
        tot_in += u["prompt_tokens"]; tot_out += u["completion_tokens"]
        for m in seg.get("notable_moments", []):
            all_moments.append({"timestamp": globalize(m.get("timestamp", ""), off),
                                "event": m.get("event", "")})
        seg_reports.append({"segment_index": i, "segment_start": mmss(off),
                            "analysis": seg})

    t2 = time.time()
    merge = post({
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": MERGE_PROMPT + json.dumps(seg_reports, indent=1)}],
        "max_tokens": 2000, "temperature": 0.2,
    })
    mu = merge["usage"]
    merged = extract_json(merge["choices"][0]["message"]["content"])
    merged["notable_moments"] = all_moments  # code-side, correct global timestamps

    result = {
        "video": video.name, "duration_s": duration, "chunks": len(chunks),
        "total_wall_s": round(time.time() - t0, 1),
        "merge_wall_s": round(time.time() - t2, 1),
        "map_prompt_tokens": tot_in, "map_gen_tokens": tot_out,
        "merge_prompt_tokens": mu["prompt_tokens"],
        "merge_gen_tokens": mu["completion_tokens"],
        "merged_analysis": merged, "segment_reports": seg_reports,
    }
    out_path.write_text(json.dumps(result, indent=1))
    print(f"TOTAL={result['total_wall_s']}s map_in={tot_in} map_out={tot_out}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
