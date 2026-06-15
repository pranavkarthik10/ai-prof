"""Modal backend for AI Prof's eyes: MiniCPM-V-4 served by llama.cpp.

The endpoint is OpenAI-compatible and accepts the image_url content produced by
``ai_prof/vision.py``. Point ``VISION_BASE_URL`` at the deployed URL; the app
adds ``/v1`` when needed.

The GGUF route is intentional: it gives this small, bursty vision service quick
cold starts and keeps the project on llama.cpp for the hackathon's Llama
Champion track. An L4 comfortably holds the Q4_K_M language model, F16 vision
projector, and an 8K context.

Bring-up:
    modal run   modal_app_vision.py::download_model
    modal run   modal_app_vision.py::warm
    modal deploy modal_app_vision.py
    # .env: VISION_BASE_URL=<serve URL>  VISION_MODEL=minicpm-v
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import struct
import subprocess
import time
import urllib.request
import zlib

import modal

MODEL_REPO = "openbmb/MiniCPM-V-4-gguf"
MODEL_REVISION = "a5a17436782fb15dff8df61ab0ec3126c3564695"
MODEL_FILE = "ggml-model-Q4_K_M.gguf"
MMPROJ_FILE = "mmproj-model-f16.gguf"
SERVED_NAME = "minicpm-v"

# Official llama.cpp CUDA server image b9570, pinned to its amd64 manifest so an
# upstream image update cannot silently alter multimodal behavior during judging.
LLAMA_CPP_IMAGE = (
    "ghcr.io/ggml-org/llama.cpp:server-cuda"
    "@sha256:3f167c81f5f281f642be62d1d9750b609fa38e7aa7be9b9ea2017a7f43a0d5eb"
)
LLAMA_SERVER = "/app/llama-server"
MODEL_DIR = "/models/minicpm-v-4"

GPU = "L4"
LLAMA_PORT = 8081
MAX_MODEL_LEN = 8192
MINUTES = 60

app = modal.App("ai-prof-vision")
model_volume = modal.Volume.from_name("ai-prof-vision-models", create_if_missing=True)

llama_image = (
    modal.Image.from_registry(LLAMA_CPP_IMAGE, add_python="3.12")
    .entrypoint([])
    .pip_install("fastapi[standard]", "httpx", "huggingface_hub>=1.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)


def _server_cmd() -> list[str]:
    return [
        LLAMA_SERVER,
        "--model",
        f"{MODEL_DIR}/{MODEL_FILE}",
        "--mmproj",
        f"{MODEL_DIR}/{MMPROJ_FILE}",
        "--alias",
        SERVED_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(LLAMA_PORT),
        "--ctx-size",
        str(MAX_MODEL_LEN),
        "--n-gpu-layers",
        "99",
        "--parallel",
        "1",
    ]


def _wait_healthy(timeout_s: int = 10 * MINUTES) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{LLAMA_PORT}/health", timeout=5
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(2)
    raise TimeoutError("llama-server did not become healthy in time")


def _png_data_uri(width: int = 64, height: int = 64) -> str:
    """Create a tiny valid RGB test image without adding Pillow to the image."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", binascii.crc32(body))

    # A white image with one blue horizontal stripe gives the vision encoder
    # real, non-uniform pixels while keeping the warm-up payload tiny.
    rows = []
    for y in range(height):
        pixel = b"\x20\x70\xd0" if height // 3 <= y < 2 * height // 3 else b"\xf8\xf8\xf8"
        rows.append(b"\x00" + pixel * width)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


@app.function(
    image=llama_image,
    volumes={"/models": model_volume},
    timeout=30 * MINUTES,
)
def download_model() -> None:
    """Download only the two GGUF files required by llama-server on CPU."""
    from huggingface_hub import hf_hub_download

    for filename in (MODEL_FILE, MMPROJ_FILE):
        print(f"Downloading {MODEL_REPO}/{filename} ...")
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=filename,
            revision=MODEL_REVISION,
            local_dir=MODEL_DIR,
        )
    model_volume.commit()
    print("MiniCPM-V model and projector downloaded.")


@app.function(
    image=llama_image,
    gpu=GPU,
    volumes={"/models": model_volume},
    timeout=20 * MINUTES,
)
def warm() -> None:
    """Smoke-test model loading and the complete multimodal request path."""
    proc = subprocess.Popen(_server_cmd())
    try:
        print("Waiting for llama-server to load MiniCPM-V...")
        _wait_healthy()
        payload = {
            "model": SERVED_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is the stripe? Answer briefly."},
                        {"type": "image_url", "image_url": {"url": _png_data_uri()}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 16,
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{LLAMA_PORT}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        response = urllib.request.urlopen(request, timeout=180).read().decode()
        print("Multimodal warm-up response:", response[:800])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


@app.function(
    image=llama_image,
    gpu=GPU,
    volumes={"/models": model_volume},
    scaledown_window=5 * MINUTES,
    timeout=30 * MINUTES,
    max_containers=1,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def serve():
    """Expose llama.cpp only after the model has finished loading."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    # ``from __future__ import annotations`` makes the route annotation a
    # string; FastAPI resolves it through module globals.
    globals()["_ProxyRequest"] = Request

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        print("Launching:", " ".join(_server_cmd()))
        proc = subprocess.Popen(_server_cmd())
        try:
            # Modal does not route traffic to the ASGI app until startup exits.
            # This avoids llama-server's temporary 503 while weights are loading.
            await __import__("asyncio").to_thread(_wait_healthy)
            print("MiniCPM-V is ready.")
            yield
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    proxy = FastAPI(lifespan=lifespan)
    upstream = f"http://127.0.0.1:{LLAMA_PORT}"

    @proxy.api_route(
        "/{path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    )
    async def forward(path: str, request: _ProxyRequest):
        client = httpx.AsyncClient(timeout=None)
        upstream_request = client.build_request(
            request.method,
            f"{upstream}/{path}",
            params=request.query_params,
            headers={
                key: value
                for key, value in request.headers.items()
                if key.lower() not in {"host", "content-length"}
            },
            content=await request.body(),
        )
        response = await client.send(upstream_request, stream=True)

        async def body():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()

        return StreamingResponse(
            body(),
            status_code=response.status_code,
            headers={
                key: value
                for key, value in response.headers.items()
                if key.lower() not in {"content-length", "transfer-encoding", "connection"}
            },
        )

    return proxy
