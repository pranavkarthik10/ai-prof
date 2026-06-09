"""The brain: Nemotron turns a slide reading into a streamed, TA-style explanation
and answers interjected questions using the cached reading as context.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import lru_cache

from openai import OpenAI

from .config import CONFIG

_SYSTEM = (
    "You are AI Prof, a warm, concise teaching assistant walking a student through "
    "their lecture slides one at a time. Explain like a good TA: plain language, an "
    "analogy when it helps, and a quick check for understanding. No preamble, no "
    "'this slide shows' filler — just teach. Keep it to a few short paragraphs."
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(base_url=CONFIG.brain.base_url, api_key=CONFIG.brain.api_key)


def _stream_chat(messages: list[dict]) -> Iterator[str]:
    if not CONFIG.brain.is_live:
        yield from _mock_stream(messages)
        return
    stream = _client().chat.completions.create(
        model=CONFIG.brain.model,
        messages=messages,
        temperature=0.6,
        max_tokens=700,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _mock_stream(messages: list[dict]) -> Iterator[str]:
    """Token-by-token mock so the streaming UX is visible without a brain endpoint."""
    user = messages[-1]["content"]
    text = (
        "(mock brain — set BRAIN_BASE_URL for real Nemotron output) "
        "Here's the gist of this slide: "
        + " ".join(user.split())[:300]
        + " ... The key idea to hold onto is how these pieces connect. "
        "Does that part make sense so far?"
    )
    for tok in text.split(" "):
        yield tok + " "
        time.sleep(0.02)


def explain_slide(
    reading: str,
    *,
    slide_no: int,
    total: int,
    outline: str,
    history: list[dict] | None = None,
) -> Iterator[str]:
    """Stream a TA-style explanation of the current slide."""
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Lecture deck outline:\n{outline}\n\n"
                f"You are now on slide {slide_no} of {total}. Its reading:\n{reading}\n\n"
                "Explain this slide to the student."
            ),
        },
    ]
    yield from _stream_chat(messages)


def answer_question(
    question: str,
    *,
    reading: str,
    slide_no: int,
    history: list[dict] | None = None,
) -> Iterator[str]:
    """Stream an answer to an interjection, grounded in the current slide reading."""
    messages = [{"role": "system", "content": _SYSTEM}]
    for turn in (history or [])[-6:]:
        messages.append(turn)
    messages.append(
        {
            "role": "user",
            "content": (
                f"We're on slide {slide_no}. Its reading:\n{reading}\n\n"
                f"The student asks: {question}\n\n"
                "Answer concisely, then offer to continue the lecture."
            ),
        }
    )
    yield from _stream_chat(messages)
