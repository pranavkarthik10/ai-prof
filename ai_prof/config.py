"""Runtime configuration for AI Prof.

Both models are reached over OpenAI-compatible HTTP, which is what the common
self-hosted runtimes expose:
  - MiniCPM-V (eyes)  -> llama.cpp ``llama-server`` with ``--mmproj``
  - Nemotron (brain)  -> vLLM / llama.cpp chat endpoint

If a base URL is left blank, that model falls back to MOCK mode so the whole
app still runs end-to-end with no weights — useful for UI work and demos.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _clean(value: str | None) -> str | None:
    """Treat blank / placeholder env values as unset."""
    if value is None:
        return None
    value = value.strip()
    return value or None


@dataclass(frozen=True)
class ModelConfig:
    base_url: str | None
    api_key: str
    model: str

    @property
    def is_live(self) -> bool:
        return self.base_url is not None


@dataclass(frozen=True)
class Config:
    vision: ModelConfig
    brain: ModelConfig
    slide_dpi: int

    @property
    def fully_mocked(self) -> bool:
        return not self.vision.is_live and not self.brain.is_live


def load_config() -> Config:
    return Config(
        vision=ModelConfig(
            base_url=_clean(os.getenv("VISION_BASE_URL")),
            api_key=_clean(os.getenv("VISION_API_KEY")) or "sk-no-key-required",
            model=_clean(os.getenv("VISION_MODEL")) or "minicpm-v",
        ),
        brain=ModelConfig(
            base_url=_clean(os.getenv("BRAIN_BASE_URL")),
            api_key=_clean(os.getenv("BRAIN_API_KEY")) or "sk-no-key-required",
            model=_clean(os.getenv("BRAIN_MODEL")) or "nemotron-3-nano",
        ),
        slide_dpi=int(_clean(os.getenv("SLIDE_DPI")) or "150"),
    )


CONFIG = load_config()
