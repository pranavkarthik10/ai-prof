"""Modal backend for AI Prof — VoxCPM2 TTS + distil-Whisper STT.

Two endpoints in one Modal app:
  serve()      → /v1/audio/speech        (TTS — VoxCPM2 via vLLM-Omni)
  transcribe() → /v1/audio/transcriptions (STT — distil-whisper-large-v3)

OpenAI-compatible /v1/audio/speech endpoint; point TTS_BASE_URL at the
printed *.modal.run URL (no /v1 suffix — the client appends it).

VoxCPM2 (2B, MiniCPM-4 backbone) outputs 48 kHz WAV and supports Voice
Design via a natural-language prefix prepended to the synthesis input — used
in ai_prof/rtc.py to give AI Prof a consistent professor voice.

vLLM-Omni layers omni-modal (TTS/STT) support on top of vLLM's scheduler
and exposes a drop-in /v1/audio/speech endpoint with PagedAttention and
continuous batching.

Bring-up:
    modal run   modal_app_vox.py::download_model   # 1. pull VoxCPM2 weights (CPU, cheap)
    modal run   modal_app_vox.py::download_stt     # 2. pull Whisper weights  (CPU, cheap)
    modal run   modal_app_vox.py::warm             # 3. ONE GPU run: compile CUDA kernels
    modal deploy modal_app_vox.py                  # 4. serve both endpoints
    # .env: TTS_BASE_URL=<serve URL>   TTS_MODEL=voxcpm2
    #       STT_BASE_URL=<transcribe URL>  STT_MODEL=distil-whisper-large-v3
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

import modal

# --- TTS: VoxCPM2 ------------------------------------------------------------
MODEL_NAME  = "openbmb/VoxCPM2"
SERVED_NAME = "voxcpm2"   # must match TTS_MODEL in the app's .env
GPU         = "A10G"      # 24 GB VRAM; VoxCPM2 needs ~8 GB
MAX_MODEL_LEN = 2048      # generous for any TTS input chunk
VLLM_PORT   = 8000
MINUTES     = 60

# --- STT: distil-Whisper -----------------------------------------------------
# Use the full CTranslate2 HF path; faster-whisper loads these natively.
STT_MODEL = "Systran/faster-distil-whisper-large-v3"
STT_PORT  = 8001

# Minimal OpenAI-compatible /v1/audio/transcriptions endpoint.
# Written to /tmp/stt_app.py inside the container at startup.
_STT_SERVER = """\
import io, os, tempfile
from fastapi import FastAPI, File, Form, UploadFile
from faster_whisper import WhisperModel

_MODEL_ID = os.environ.get("WHISPER_MODEL", "Systran/faster-distil-whisper-large-v3")
_model = WhisperModel(_MODEL_ID, device="cuda", compute_type="float16")

app = FastAPI()

@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default=_MODEL_ID),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
):
    data = await file.read()
    with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file.filename or ".wav")[1] or ".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        segments, _ = _model.transcribe(tmp_path, language=language or None)
        text = " ".join(s.text for s in segments).strip()
    finally:
        os.unlink(tmp_path)
    return {"text": text}
"""

app = modal.App("ai-prof-vox")

# Persistent caches — same pattern as modal_app.py (brain).
# FlashInfer cache is kept even though VoxCPM2 is transformer-only:
# vLLM itself may emit FlashInfer kernels depending on the backend.
hf_cache         = modal.Volume.from_name("ai-prof-vox-hf-cache",      create_if_missing=True)
vllm_cache       = modal.Volume.from_name("ai-prof-vox-vllm-cache",    create_if_missing=True)
flashinfer_cache = modal.Volume.from_name("ai-prof-vox-fi-cache",      create_if_missing=True)
triton_cache     = modal.Volume.from_name("ai-prof-vox-triton-cache",  create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/vllm":        vllm_cache,
    "/root/.cache/flashinfer":  flashinfer_cache,
    "/root/.triton":            triton_cache,
}

# vLLM-Omni installation recipe from the official README:
#   1. pip install vllm==0.19.0
#   2. git clone vllm-omni && pip install -e .
# We use uv for faster resolution and to avoid pip's slower solver on large graphs.
vox_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .run_commands(
        "pip install uv 'huggingface_hub[hf_transfer]>=0.27'",
        "uv pip install --system 'vllm==0.22.1'",
        "pip install hf_transfer",  # vllm's uv install drops hf_transfer; reinstall after
        "git clone --depth 1 https://github.com/vllm-project/vllm-omni.git /opt/vllm-omni",
        "cd /opt/vllm-omni && uv pip install --system -e .",
        "pip install 'voxcpm>=2.0'",  # vllm-omni requires this for VoxCPM2; it's a separate PyPI package
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _vllm_omni_cmd() -> list[str]:
    # Use `vllm-omni` CLI (installed by pip install -e vllm-omni).
    # With vllm 0.22.1 the plugin also patches `vllm serve` to accept --omni,
    # but the explicit CLI is the documented/safe path.
    return [
        "vllm-omni", "serve", MODEL_NAME,
        "--omni",
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", "4",
        "--tensor-parallel-size", "1",
        "--trust-remote-code",
    ]


def _wait_healthy(timeout_s: int = 20 * MINUTES) -> None:
    base = f"http://127.0.0.1:{VLLM_PORT}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base}/health", timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError("vLLM-Omni did not become healthy in time")


@app.function(
    image=vox_image,
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=60 * MINUTES,
)
def download_model(model_name: str = MODEL_NAME) -> None:
    """Pull VoxCPM2 weights to the Volume on CPU (no GPU billed)."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_name} ...")
    snapshot_download(model_name, ignore_patterns=["*.pt", "*.pth"])
    hf_cache.commit()
    print("Done.")


@app.function(image=vox_image, gpu=GPU, volumes=VOLUMES, timeout=60 * MINUTES)
def warm() -> None:
    """One controlled GPU run: boot vLLM-Omni, fire a warm-up TTS request to
    trigger every kernel compile, then commit the caches."""
    proc = subprocess.Popen(_vllm_omni_cmd())
    try:
        print("Waiting for vLLM-Omni to compile + become healthy (first run is slow)...")
        _wait_healthy()
        req_data = json.dumps({
            "model": SERVED_NAME,
            "input": "Warm-up synthesis complete.",
            "voice": "default",
            "response_format": "wav",
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{VLLM_PORT}/v1/audio/speech",
            data=req_data,
            headers={"Content-Type": "application/json"},
        )
        wav_bytes = urllib.request.urlopen(req, timeout=120).read()
        print(f"Warm-up TTS response: {len(wav_bytes):,} bytes of WAV audio.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    vllm_cache.commit()
    flashinfer_cache.commit()
    triton_cache.commit()
    print("Warm complete — caches committed.")


@app.function(
    image=vox_image,
    gpu=GPU,
    volumes=VOLUMES,
    scaledown_window=10 * MINUTES,
    timeout=60 * MINUTES,
    max_containers=1,
)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=VLLM_PORT, startup_timeout=25 * MINUTES)
def serve() -> None:
    print("Launching:", " ".join(_vllm_omni_cmd()))
    subprocess.Popen(_vllm_omni_cmd())


# =============================================================================
# STT — distil-whisper-large-v3 via faster-whisper-server
# Exposes OpenAI-compatible POST /v1/audio/transcriptions.
# Point STT_BASE_URL at the printed *.modal.run URL; set STT_MODEL=distil-whisper-large-v3.
# =============================================================================

stt_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .pip_install(
        "faster-whisper",
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",
        "hf_transfer",
        "huggingface_hub",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


@app.function(
    image=stt_image,
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=30 * MINUTES,
)
def download_stt(model_name: str = STT_MODEL) -> None:
    """Pull distil-Whisper weights to the shared HF cache Volume (CPU, no GPU billed)."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_name} ...")
    snapshot_download(model_name)
    hf_cache.commit()
    print("Done.")


@app.function(
    image=stt_image,
    gpu="T4",
    volumes={"/root/.cache/huggingface": hf_cache},
    scaledown_window=5 * MINUTES,
    timeout=30 * MINUTES,
    max_containers=1,
)
@modal.concurrent(max_inputs=4)
@modal.web_server(port=STT_PORT, startup_timeout=5 * MINUTES)
def transcribe() -> None:
    with open("/tmp/stt_app.py", "w") as f:
        f.write(_STT_SERVER)
    cmd = ["uvicorn", "stt_app:app", "--host", "0.0.0.0", "--port", str(STT_PORT)]
    print("Launching STT server (model:", STT_MODEL, ")")
    subprocess.Popen(cmd, cwd="/tmp", env={**os.environ, "WHISPER_MODEL": STT_MODEL})
