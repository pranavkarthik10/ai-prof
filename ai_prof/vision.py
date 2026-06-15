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
    "You are reading a single lecture slide shown as an image. "
    "Respond in English only. "
    "Produce a compact, structured reading another model will use to explain the slide aloud. "
    "Use exactly these plain-text sections, omitting any that don't apply:\n"
    "TITLE: <slide title>\n"
    "BULLETS: <key points, one per line>\n"
    "EQUATIONS: <any formulas, written out>\n"
    "DIAGRAM: <what any figure/diagram/chart shows>\n"
    "CONCEPTS: <the core concepts this slide is about>\n"
    "Be faithful to what's actually on the slide. Do not invent content. "
    "Do not use XML or JSON — plain text only."
)

_RETRY_PROMPT = (
    "Read this lecture slide image and respond in English only. "
    "Use plain text with these sections (omit any that don't apply):\n"
    "TITLE: ...\nBULLETS: ...\nEQUATIONS: ...\nDIAGRAM: ...\nCONCEPTS: ...\n"
    "No XML, no JSON, no repetition."
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


def _is_degenerate(text: str) -> bool:
    """Detect infinite-loop or language-drift outputs."""
    if not text:
        return True
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return True
    # Flag if >40% of non-empty lines are duplicates (repetition loop)
    if len(set(lines)) / len(lines) < 0.6:
        return True
    # Flag if majority of characters are non-ASCII (language drift)
    non_ascii = sum(1 for c in text if ord(c) > 127)
    if non_ascii / max(len(text), 1) > 0.3:
        return True
    return False


def _call_vision(instruction: str, image_path: str) -> str:
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


def read_slide(
    image_path: str,
    text_layer: str = "",
    question: str | None = None,
    prior_reading: str | None = None,
) -> str:
    """Return a structured reading of the slide image.

    ``question`` switches to a targeted 'look closer' read for a specific ask.
    ``prior_reading`` is the reading of the previous slide; passed when slides
    are part of an animation sequence so the model has context on what changed.
    Falls back to a mock reading derived from the PDF text layer if no endpoint
    is configured.
    """
    if not CONFIG.vision.is_live:
        return _mock_reading(text_layer, question)

    if question:
        instruction = f"Look closely at this slide and answer in English: {question}"
    else:
        instruction = _READING_PROMPT
        if prior_reading:
            instruction = (
                f"The previous slide reading was:\n{prior_reading}\n\n"
                + instruction
            )

    result = _call_vision(instruction, image_path)
    if _is_degenerate(result):
        result = _call_vision(_RETRY_PROMPT, image_path)
    if _is_degenerate(result):
        return _mock_reading(text_layer, question)
    return result
