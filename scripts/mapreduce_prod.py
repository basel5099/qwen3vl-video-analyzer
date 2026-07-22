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

# Chunks answer in PROSE (never JSON): per-chunk JSON breaks too easily
# (truncation, format drift) and the damage travels through the pipeline.
# JSON is built exactly ONCE, in the final merge step.
CHUNK_PROMPT = """This is one segment of a longer craft video. Describe it in detailed prose (plain text, NO JSON, no markdown): what happens step by step, the materials and objects used, the actions performed, what craft is being practiced, and any authenticity signals (real-time hands, cuts, overlays). Whenever you reference a specific moment, write its time WITHIN THIS SEGMENT as [mm:ss]."""

COHERENT_PREFIX = """CONTEXT FROM EARLIER IN THIS VIDEO (already analyzed): {prev}

Analyze THIS segment as a CONTINUATION of that story — refer to established objects/steps where relevant, and note what is NEW or has progressed. """

MERGE_PROMPT = """You are given sequential prose analyses of ONE long craft video. All timestamps in them are GLOBAL and look like [HH:MM:SS]. Write ONE merged analysis and return ONLY valid JSON. Copy timestamps EXACTLY as written — never invent or recompute them:
{"summary": "3-4 sentences on the full arc", "materials_and_objects": [], "actions": [], "craft_type": "", "authenticity_impression": "", "narrative_arc": "", "notable_moments": [{"timestamp": "HH:MM:SS", "event": ""}]}

Segment analyses:
"""


def post(payload, timeout=1800, api=None):
    api = api or alive_backends()[0]
    req = urllib.request.Request(api, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


# Text-only reasoning (planner + final decision) goes to Gemini Flash when a
# key is configured; the local vision model only ever SEES video segments.
GEMINI_KEY = os.environ.get("LAB_GEMINI_KEY", "").strip()
GEMINI_MODEL = os.environ.get("LAB_GEMINI_MODEL", "gemini-2.5-flash")


def text_llm(prompt_text, max_tokens=2200):
    """Returns (text, usage_dict). Gemini Flash if configured, else local."""
    if GEMINI_KEY:
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}")
        req = urllib.request.Request(url, data=json.dumps({
            "contents": [{"parts": [{"text": prompt_text}]}],
            # thinking off: with it on, thoughts eat maxOutputTokens and the
            # visible answer arrives truncated mid-JSON.
            "generationConfig": {"maxOutputTokens": max_tokens + 2000,
                                 "temperature": 0.2,
                                 "thinkingConfig": {"thinkingBudget": 0}},
        }).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.load(r)
        um = data.get("usageMetadata", {})
        return (data["candidates"][0]["content"]["parts"][0]["text"],
                {"prompt_tokens": um.get("promptTokenCount", 0),
                 "completion_tokens": um.get("candidatesTokenCount", 0)})
    resp = post({"model": MODEL,
                 "messages": [{"role": "user", "content": prompt_text}],
                 "max_tokens": max_tokens, "temperature": 0.2})
    return resp["choices"][0]["message"]["content"], resp["usage"]


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
    # strip markdown code fences (```json ... ```) some models wrap JSON in
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```\s*$", "", text.strip())
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


TS_PAT = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")


def globalize_text(text, offset_s):
    """Rewrite every [mm:ss] in chunk prose to global [HH:MM:SS] — in code,
    so the merge model only ever COPIES timestamps, never computes them."""
    return TS_PAT.sub(lambda m: "[" + globalize(m.group(1), offset_s) + "]", text)


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

    # Per-video chunk dir: concurrent jobs must never share (they race and
    # cross-contaminate each other's chunks — found the hard way).
    chunk_dir = CHUNK_DIR.parent / f"chunks_{video.stem}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for old in chunk_dir.glob("chunk_*.mp4"):
        old.unlink()
    t0 = time.time()
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", video]
    if limit_seconds:
        cmd += ["-t", str(limit_seconds)]
    cmd += ["-c", "copy", "-an", "-f", "segment",
            "-segment_time", str(chunk_seconds), "-reset_timestamps", "1",
            chunk_dir / "chunk_%03d.mp4"]
    subprocess.run(cmd, check=True)
    chunks = sorted(chunk_dir.glob("chunk_*.mp4"))
    print(f"split into {len(chunks)} chunks in {time.time() - t0:.1f}s")

    apis = alive_backends()
    # User prompt is EXCLUSIVE: when set, it fully replaces the general craft
    # prompts (no competing instructions). Without it, the defaults apply.
    user_prompt = os.environ.get("LAB_USER_PROMPT", "").strip()
    global CHUNK_PROMPT, MERGE_PROMPT
    gen_prompt = ""
    output_schema = os.environ.get("LAB_OUTPUT_SCHEMA", "").strip()
    if user_prompt:
        # PLANNER (Basel's design): first tell the model our constraint (no
        # full-video context, part-by-part analysis) and let IT write the
        # per-segment evidence-gathering instruction for the user's request.
        planner_text, _pu = text_llm((
                "A user wants the following from a long video:\n"
                f"USER REQUEST: {user_prompt}\n\n"
                "Constraint: the video cannot be analyzed in one piece. It is "
                "split into sequential segments, each analyzed in isolation "
                "by a vision model that sees ONLY that segment; the outputs "
                "are combined afterwards to answer the user.\n"
                "Write the single best instruction to give EACH segment "
                "analyst so that the combined observations fully answer the "
                "user. The instruction must ask for plain-prose observations "
                "(no JSON) with segment-local timestamps written as [mm:ss], "
                "must ask to collect ALL evidence relevant to the request, "
                "and must NOT ask the analyst to answer the global question "
                "themselves. Return ONLY the instruction text."
                + (("\nThe combined output must later fill this JSON template "
                    "— make sure the instruction collects evidence for ALL "
                    "its fields:\n" + output_schema[:3000])
                   if output_schema else "")), 500)
        gen_prompt = planner_text.strip()
        print(f"planner prompt: {gen_prompt[:200]}")
        CHUNK_PROMPT = (
            gen_prompt + "\n\nAnswer in plain prose (NO JSON). Write times "
            "WITHIN THIS SEGMENT as [mm:ss].")
        if output_schema:
            MERGE_PROMPT = (
                f"ORIGINAL USER REQUEST about ONE long video: {user_prompt}\n"
                f"Each segment was analyzed with this instruction: {gen_prompt}\n\n"
                "Below are the sequential segment observations. All timestamps "
                "in them are GLOBAL and look like [HH:MM:SS].\n"
                "Your ENTIRE output must be ONLY valid JSON that fills EXACTLY "
                "this template: identical key names and nesting, EVERY key "
                "present, NO extra keys, no task letters, no wrapper text. "
                "Where evidence is missing use the template's empty/zero "
                "defaults. Fields named *_sec take NUMERIC seconds (convert "
                "from [HH:MM:SS]); never invent timestamps not grounded in "
                "the observations.\nTEMPLATE:\n" + output_schema +
                "\n\nSegment observations:\n")
        else:
            MERGE_PROMPT = (
                f"ORIGINAL USER REQUEST about ONE long video: {user_prompt}\n"
                f"Each segment was analyzed with this instruction: {gen_prompt}\n\n"
                "Below are the sequential segment observations. All timestamps "
                "in them are GLOBAL and look like [HH:MM:SS]. Produce the final "
                "result answering the ORIGINAL USER REQUEST exactly in the form "
                "the user asked for, and return ONLY valid JSON. "
                "IF the request itself defines an explicit output JSON "
                "structure/contract, return EXACTLY that structure (verbatim "
                "key names, all fields present, no extras) INSTEAD of the "
                "default shape below. Copy timestamps EXACTLY as written. "
                "Default shape: "
                '{"user_answer": "the direct final answer in the form the user asked for", '
                '"summary": "1-2 sentences of supporting context", '
                '"notable_moments": [{"timestamp": "HH:MM:SS", "event": ""}]}'
                "\n\nSegment observations:\n")
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
        seg_text = resp["choices"][0]["message"]["content"].strip()
        print(f"chunk {i}: wall={wall:.1f}s in={u['prompt_tokens']} out={u['completion_tokens']}")
        return i, seg_text, u

    if coherent:
        results = []
        prev = ""
        for ic in enumerate(chunks):
            ptxt = (COHERENT_PREFIX.format(prev=prev) + CHUNK_PROMPT) if prev \
                else CHUNK_PROMPT
            i, seg_text, u = analyze_chunk(ic, ptxt)
            results.append((i, seg_text, u))
            prev = seg_text[:1500]
    else:
        with ThreadPoolExecutor(max_workers=len(apis)) as pool:
            results = sorted(pool.map(analyze_chunk, enumerate(chunks)))

    seg_reports = []
    tot_in = tot_out = 0
    merge_input = ""
    for i, seg_text, u in results:
        off = i * chunk_seconds
        tot_in += u["prompt_tokens"]; tot_out += u["completion_tokens"]
        gtext = globalize_text(seg_text, off)  # [mm:ss] -> [HH:MM:SS], in code
        seg_reports.append({"segment_index": i, "segment_start": mmss(off),
                            "analysis_text": gtext})
        merge_input += f"\n--- segment {i} (starts {mmss(off)}) ---\n{gtext}\n"

    t2 = time.time()
    merge_text, mu = text_llm(MERGE_PROMPT + merge_input, 2200)
    merged = extract_json(merge_text)
    # Timestamps are copy-only: drop any moment whose stamp never appears in
    # the (code-globalized) segment texts; salvage from the texts if empty.
    # (Skipped in output_schema mode — a caller contract must come back
    # EXACTLY as templated, with no injected keys.)
    if not output_schema:
        valid_ts = set(m.group(1) for m in re.finditer(r"\[(\d{2}:\d{2}:\d{2})\]", merge_input))
        moments = [m for m in merged.get("notable_moments", [])
                   if isinstance(m, dict) and m.get("timestamp") in valid_ts]
        if not moments:
            moments = [{"timestamp": m.group(1),
                        "event": merge_input[m.end():m.end() + 110].split(".")[0].strip(" -:،")}
                       for m in re.finditer(r"\[(\d{2}:\d{2}:\d{2})\]", merge_input)][:40]
        merged["notable_moments"] = moments

    result = {
        "video": video.name, "duration_s": duration, "chunks": len(chunks),
        "total_wall_s": round(time.time() - t0, 1),
        "merge_wall_s": round(time.time() - t2, 1),
        "map_prompt_tokens": tot_in, "map_gen_tokens": tot_out,
        "merge_prompt_tokens": mu["prompt_tokens"],
        "merge_gen_tokens": mu["completion_tokens"],
        "merged_analysis": merged, "segment_reports": seg_reports,
    }
    if gen_prompt:
        result["planner_prompt"] = gen_prompt
    result["text_model"] = GEMINI_MODEL if GEMINI_KEY else "local"
    out_path.write_text(json.dumps(result, indent=1))
    print(f"TOTAL={result['total_wall_s']}s map_in={tot_in} map_out={tot_out}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
