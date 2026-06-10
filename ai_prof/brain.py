"""The brain: Nemotron turns a slide reading into a streamed, TA-style explanation
and answers interjected questions using the cached reading as context.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from functools import lru_cache

from openai import OpenAI

from .config import CONFIG

_SYSTEM = """You are AI Prof, teaching one student through a lecture deck in real time.

Sound like a professor speaking naturally, not a study guide generating notes.
- Open conversationally and get to the point. On the first slide, a simple opening
  like "Hey, welcome back. Today we're looking at image filtering" is ideal.
- Teach one or two ideas at a time in short spoken paragraphs. Connect them to the
  previous slide when useful.
- Explain jargon in plain language and use a compact example or analogy only when it
  genuinely makes the idea easier to understand.
- Guide the student toward the idea instead of exhaustively reciting every item.
- Treat course codes, lecture numbers, citations, and credits as background metadata
  unless they matter to the concept being taught.
- End with a brief transition or a useful content question when one fits naturally.

Never use canned headings such as "Key ideas on this slide" or "Quick check". Do not
announce "Slide 1", narrate the slide layout, repeat all visible text, or ask whether
the student is ready to begin or move on. Do not say "this slide shows". Keep the
response concise enough to sound good aloud."""


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(base_url=CONFIG.brain.openai_base_url, api_key=CONFIG.brain.api_key)


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
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content


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
    messages = [{"role": "system", "content": _SYSTEM}]
    for turn in (history or [])[-6:]:
        if turn.get("content"):
            messages.append(turn)
    transition = (
        "This is the opening slide. Welcome the student briefly and introduce today's topic."
        if slide_no == 1
        else "Continue naturally from the previous explanation without greeting the student again."
    )
    messages.append(
        {
            "role": "user",
            "content": (
                f"Lecture deck outline:\n{outline}\n\n"
                f"Current position: slide {slide_no} of {total}.\n"
                f"Grounded slide reading:\n{reading}\n\n"
                f"{transition} Teach the important idea conversationally. Do not turn the "
                "response into a structured slide summary."
            ),
        }
    )
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
                "Answer directly and conversationally. Relate the answer to the current "
                "topic, then smoothly hand control back to the lecture without a canned offer."
            ),
        }
    )
    yield from _stream_chat(messages)
