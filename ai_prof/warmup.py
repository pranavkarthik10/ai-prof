"""Wake configured inference services without blocking the Gradio frontend."""

from __future__ import annotations

import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import CONFIG, ModelConfig

_started = False
_start_lock = threading.Lock()


def _service_url(config: ModelConfig, path: str) -> str | None:
    if not config.base_url:
        return None
    base = config.base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}{path}"


def _wake(name: str, url: str, api_key: str) -> None:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=30 * 60) as response:
            print(f"[warmup] {name} ready ({response.status})")
    except Exception as exc:
        print(f"[warmup] {name} failed: {exc}")


def _wake_all() -> None:
    services = [
        ("vision", _service_url(CONFIG.vision, "/health"), CONFIG.vision.api_key),
        ("brain", _service_url(CONFIG.brain, "/health"), CONFIG.brain.api_key),
        ("tts", _service_url(CONFIG.tts, "/health"), CONFIG.tts.api_key),
        # Whisper constructs its model while importing the FastAPI app, so this
        # route is available only after the GPU container is actually ready.
        ("stt", _service_url(CONFIG.stt, "/openapi.json"), CONFIG.stt.api_key),
    ]
    configured = [(name, url, key) for name, url, key in services if url]
    if not configured:
        return

    print("[warmup] waking configured inference services...")
    with ThreadPoolExecutor(max_workers=len(configured)) as pool:
        futures = {
            pool.submit(_wake, name, url, key): name
            for name, url, key in configured
        }
        for future in as_completed(futures):
            future.result()


def start_service_warmup() -> None:
    """Start one background warmup pass for this Gradio process."""
    global _started
    with _start_lock:
        if _started:
            return
        _started = True
    threading.Thread(target=_wake_all, name="service-warmup", daemon=True).start()
