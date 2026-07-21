#!/usr/bin/env python3
"""Create (or update) the RunPod template for the video-analysis service.

Usage:
    export RUNPOD_API_KEY=rpa_...          # runpod.io -> Settings -> API Keys
    export LAB_API_KEY=$(openssl rand -hex 24)   # your service auth key
    export SSH_PUBKEY="ssh-ed25519 AAAA... you@host"   # injected into pods
    python make_template.py [existing_template_id]

Then deploy any GPU from the template in the RunPod console (or via API).
The pod self-installs everything and serves the API on port 8100
(https://<podId>-8100.proxy.runpod.net). First boot ~10-15 min.
"""
import base64
import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = HERE.parent / "scripts"

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
LAB_API_KEY = os.environ["LAB_API_KEY"]
SSH_PUBKEY = os.environ.get("SSH_PUBKEY", "")


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes().replace(b"\r\n", b"\n")).decode()


boot_b64 = b64(HERE / "bootstrap.sh")
tpl = {
    "name": "qwen3vl-video-analyzer",
    "imageName": "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04",
    "dockerArgs": f"bash -c 'echo {boot_b64} | base64 -d > /boot.sh && bash /boot.sh'",
    "containerDiskInGb": 120,
    "volumeInGb": 0,
    "ports": "22/tcp,8100/http",
    "env": [
        {"key": "LAB_LAUNCH_B64", "value": b64(SCRIPTS / "launch.sh")},
        {"key": "LAB_MAPREDUCE_B64", "value": b64(SCRIPTS / "mapreduce_prod.py")},
        {"key": "LAB_API_B64", "value": b64(SCRIPTS / "analyzer_api.py")},
        {"key": "LAB_API_KEY", "value": LAB_API_KEY},
        {"key": "PUBLIC_KEY", "value": SSH_PUBKEY},
    ],
    "readme": ("Qwen3-VL-8B async video-analysis API. POST /analyze "
               "{video_url|video_path} -> job_id; GET /result/{job_id}. "
               "Bearer LAB_API_KEY. Any codec (normalizes to h264/360p/1fps). "
               "VRAM-adaptive: 32GB cards auto-drop to 45K ctx / 3-min chunks."),
}
if len(sys.argv) > 1:
    tpl["id"] = sys.argv[1]

payload = {
    "query": "mutation SaveTemplate($input: SaveTemplateInput) "
             "{ saveTemplate(input: $input) { id name } }",
    "variables": {"input": tpl},
}
req = urllib.request.Request(
    "https://api.runpod.io/graphql",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json",
             "Authorization": f"Bearer {RUNPOD_API_KEY}"},
)
with urllib.request.urlopen(req, timeout=30) as r:
    print(json.load(r))
