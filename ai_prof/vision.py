"""The eyes: MiniCPM-V reads a slide image into a structured 'slide reading'.

Computed once per slide and cached — interjections reuse the reading and never
re-run the (heavy) vision pass.
"""

from __future__ import annotations

import base64
import mimetypes
from functools import lru_cache

from openai import OpenAI

from .config import CONFIG

_READING_PROMPT = (
    "You are reading a single lecture slide shown as an image. Produce a compact, "
    "structured reading another model will use to explain the slide aloud. Use these sections, "
    "omitting any that don't apply:\n"
    "TITLE: <slide title>\n"
    "BULLETS: <key points, one per line>\n"
    "EQUATIONS: <any formulas, written out>\n"
    "DIAGRAM: <what any figure/diagram/chart shows>\n"
    "CONCEPTS: <the core concepts this slide is about>\n"
    "Be faithful to what's actually on the slide. Do not invent content."
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(base_url=CONFIG.vision.openai_base_url, api_key=CONFIG.vision.api_key)


def _data_uri(image_path: str) -> str:
    mime = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _mock_reading(text: str, question: str | None) -> str:
    head = next((ln.strip() for ln in text.splitlines() if ln.strip()), "Untitled slide")
    body = " ".join(text.split())[:400]
    if question:
        return f"[mock vision] Looking closely for: {question}\nVisible text: {body or '(none)'}"
    return (
        f"TITLE: {head}\n"
        f"BULLETS:\n{text or '(no extractable text)'}\n"
        "CONCEPTS: (mock reading — set VISION_BASE_URL for a real MiniCPM-V pass)"
    )


def read_slide(image_path: str, text_layer: str = "", question: str | None = None) -> str:
    """Return a structured reading of the slide image.

    ``question`` switches to a targeted 'look closer' read for a specific ask.
    Falls back to a mock reading derived from the PDF text layer if no endpoint
    is configured.
    """
    if not CONFIG.vision.is_live:
        return _mock_reading(text_layer, question)

    instruction = (
        f"Look closely at this slide and answer: {question}"
        if question
        else _READING_PROMPT
    )
    resp = _client().chat.completions.create(
        model=CONFIG.vision.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": _data_uri(image_path)}},
                ],
            }
        ],
        temperature=0.1,
        max_tokens=512,
    )
    return (resp.choices[0].message.content or "").strip()
