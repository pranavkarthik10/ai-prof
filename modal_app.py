"""Modal backend for AI Prof — the brain (Nemotron 3 Nano) served over vLLM.

OpenAI-compatible endpoint that the Gradio app points ``BRAIN_BASE_URL`` at. The
GPU container scales to zero when idle.

Nemotron 3 Nano is a hybrid Mamba-2 + MoE *reasoning* model. On first run vLLM
JIT-compiles CUDA kernels — notably FlashInfer's CUTLASS fused-MoE kernel (slow,
minutes). We persist those compile caches to Volumes and warm ONCE, so every
later cold start reuses them (~tens of seconds) instead of recompiling on an
expensive GPU. Cost discipline: ``max_containers=1`` so a request burst can never
fan out into multiple GPUs, and warming is a single controlled ``modal run`` —
never a curl storm against the live endpoint.

Bring-up:
    modal run   modal_app.py::download_model   # 1. pull weights to a Volume (CPU, cheap)
    modal run   modal_app.py::warm             # 2. ONE GPU run: compile + cache kernels
    modal deploy modal_app.py                  # 3. serve; cold starts now reuse the cache
    # the printed *.modal.run URL + "/v1" is BRAIN_BASE_URL
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.request

import modal

# --- what to serve -----------------------------------------------------------
MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
SERVED_NAME = "nemotron-3-nano"  # must match BRAIN_MODEL in the app's .env
GPU = "H100"
MAX_MODEL_LEN = 16384  # ample for slide reading + outline + history; small KV cache
VLLM_PORT = 8000
MINUTES = 60

app = modal.App("ai-prof-brain")

# Persistent caches: weights, vLLM torch.compile artifacts, and FlashInfer's JIT
# kernels. Mounting the FlashInfer cache is what stops the CUTLASS MoE kernel from
# recompiling on every cold start.
hf_cache = modal.Volume.from_name("ai-prof-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("ai-prof-vllm-cache", create_if_missing=True)
flashinfer_cache = modal.Volume.from_name("ai-prof-flashinfer-cache", create_if_missing=True)

triton_cache = modal.Volume.from_name("ai-prof-triton-cache", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/root/.cache/vllm": vllm_cache,
    "/root/.cache/flashinfer": flashinfer_cache,
    "/root/.triton": triton_cache,  # Mamba2 SSD Triton kernel cache (~11 min compile)
}

# CUDA *devel* base ships nvcc — Nemotron-H's Mamba-2 kernels JIT-compile at init.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm>=0.12.0", "huggingface_hub[hf_transfer]>=0.27")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)


def _vllm_cmd() -> list[str]:
    # No --enforce-eager: we WANT torch.compile + CUDA-graph capture so the
    # artifacts get cached to the vLLM volume (FAST_BOOT=False pattern).
    return [
        "vllm", "serve", MODEL_NAME,
        "--served-model-name", SERVED_NAME,
        "--host", "0.0.0.0", "--port", str(VLLM_PORT),
        "--max-model-len", str(MAX_MODEL_LEN),
        "--max-num-seqs", "8",
        "--tensor-parallel-size", "1",
        "--trust-remote-code",
        "--reasoning-parser", "nemotron_v3",
    ]


def _wait_healthy(timeout_s: int = 25 * MINUTES) -> None:
    base = f"http://127.0.0.1:{VLLM_PORT}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{base}/health", timeout=5)
            return
        except Exception:
            time.sleep(5)
    raise TimeoutError("vLLM did not become healthy in time")


@app.function(
    image=vllm_image,
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=60 * MINUTES,
)
def download_model(model_name: str = MODEL_NAME) -> None:
    """Pull weights to the Volume on CPU (no GPU billed during the big download)."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {model_name} -> /root/.cache/huggingface ...")
    snapshot_download(model_name, ignore_patterns=["*.pt", "*.pth"])
    hf_cache.commit()
    print("Done.")


@app.function(image=vllm_image, gpu=GPU, volumes=VOLUMES, timeout=60 * MINUTES)
def warm() -> None:
    """One controlled GPU run: boot vLLM, fire a single request to trigger every
    kernel compile, then commit the compile caches. Run with ``modal run``."""
    proc = subprocess.Popen(_vllm_cmd())
    try:
        print("Waiting for vLLM to compile + become healthy (first time is slow)...")
        _wait_healthy()
        req = urllib.request.Request(
            f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions",
            data=json.dumps(
                {
                    "model": SERVED_NAME,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 32,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        print("Response:", urllib.request.urlopen(req, timeout=120).read().decode()[:400])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
    vllm_cache.commit()
    flashinfer_cache.commit()
    print("Warm complete — compile caches committed. Cold starts will now be fast.")


@app.function(
    image=vllm_image,
    gpu=GPU,
    volumes=VOLUMES,
    scaledown_window=20 * MINUTES,  # keep warm between requests; cold start is ~2 min
    timeout=60 * MINUTES,
    max_containers=1,  # cost guard: a request burst can never fan out into many GPUs
)
@modal.concurrent(max_inputs=8)  # vLLM batches concurrent requests on the one replica
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve() -> None:
    print("Launching:", " ".join(_vllm_cmd()))
    subprocess.Popen(_vllm_cmd())
