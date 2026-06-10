"""Professor agent planning.

The model plans one short teaching beat at a time. The Gradio orchestrator owns
all state mutation and validates every requested tool before executing it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from openai import OpenAI

from .config import CONFIG

Trigger = Literal["continue", "question"]
ActionName = Literal[
    "goto_slide",
    "next_slide",
    "prev_slide",
    "write_note",
    "write_latex",
    "clear_whiteboard",
]

_PLANNER_SYSTEM = """You are the planning brain of AI Prof, a live professor.
Plan exactly one short teaching beat at a time.

You receive a compact index of the entire lecture, the full reading of the current
slide, recent conversation, and the current whiteboard. Decide what the student
should hear next and whether a slide or whiteboard action would help.

Return JSON only:
{
  "narration": "Natural spoken explanation, usually 1-3 short paragraphs.",
  "actions": [
    {"tool": "goto_slide", "args": {"index": 4}},
    {"tool": "write_note", "args": {"title": "Kernel", "body": "A small matrix of weights."}},
    {"tool": "write_latex", "args": {"expression": "g(x,y)=sum_i sum_j h(i,j)f(x-i,y-j)"}},
    {"tool": "clear_whiteboard", "args": {}}
  ],
  "continue_lecture": true
}

Rules:
- Slide indices are 1-based and must exist in the supplied deck index.
- Use at most one navigation action and at most two whiteboard actions.
- Stay on the current slide unless another indexed slide is clearly more useful.
- For a student question, answer it directly. Navigate only when another slide
  materially improves the answer.
- Use the whiteboard only when a compact note or equation improves understanding.
- Clear it when old material would be confusing.
- Do not call a tool just to demonstrate agency.
- Keep narration conversational and suitable for text-to-speech.
- Use no markdown headings and do not mention tool names."""


@dataclass(frozen=True)
class AgentAction:
    tool: ActionName
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TeachingBeat:
    narration: str
    actions: tuple[AgentAction, ...] = ()
    continue_lecture: bool = True


def _client() -> OpenAI:
    return OpenAI(
        base_url=CONFIG.brain.openai_base_url,
        api_key=CONFIG.brain.api_key,
    )


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("agent response must be a JSON object")
    return value


def _repair_json(text: str) -> str | None:
    """Ask the brain to repair malformed planner JSON without re-planning."""
    if not CONFIG.brain.is_live:
        return None
    try:
        response = _client().chat.completions.create(
            model=CONFIG.brain.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Repair the supplied malformed JSON. Preserve its intended values, "
                        "return one valid JSON object only, and add no commentary."
                    ),
                },
                {"role": "user", "content": text},
            ],
            temperature=0,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content
    except Exception as exc:
        print(f"[agent] JSON repair error: {exc}")
        return None


def _validate_actions(raw_actions: Any, total_slides: int) -> tuple[AgentAction, ...]:
    if not isinstance(raw_actions, list):
        return ()

    valid: list[AgentAction] = []
    navigation_count = 0
    whiteboard_count = 0
    for raw in raw_actions:
        if not isinstance(raw, dict):
            continue
        tool = raw.get("tool")
        args = raw.get("args") if isinstance(raw.get("args"), dict) else {}

        if tool in {"goto_slide", "next_slide", "prev_slide"}:
            if navigation_count:
                continue
            if tool == "goto_slide":
                index = args.get("index")
                if not isinstance(index, int) or not 1 <= index <= total_slides:
                    continue
                args = {"index": index}
            else:
                args = {}
            navigation_count += 1
        elif tool == "write_note":
            if whiteboard_count >= 2:
                continue
            title = str(args.get("title", "")).strip()[:80]
            body = str(args.get("body", "")).strip()[:500]
            if not title and not body:
                continue
            args = {"title": title, "body": body}
            whiteboard_count += 1
        elif tool == "write_latex":
            if whiteboard_count >= 2:
                continue
            expression = str(args.get("expression", "")).strip()[:500]
            if not expression:
                continue
            args = {"expression": expression}
            whiteboard_count += 1
        elif tool == "clear_whiteboard":
            if whiteboard_count >= 2:
                continue
            args = {}
            whiteboard_count += 1
        else:
            continue
        valid.append(AgentAction(tool=tool, args=args))
    return tuple(valid)


def _fallback_beat(
    *,
    trigger: Trigger,
    current_slide: int,
    total_slides: int,
    current_reading: str,
    question: str | None,
) -> TeachingBeat:
    title = next(
        (
            line.split(":", 1)[1].strip()
            for line in current_reading.splitlines()
            if line.upper().startswith("TITLE:")
        ),
        f"slide {current_slide}",
    )
    if trigger == "question":
        narration = (
            f"Let’s connect that question to {title}. {question or ''} "
            "The important idea is how the details on this slide support the concept."
        )
        return TeachingBeat(narration=narration.strip(), continue_lecture=False)
    return TeachingBeat(
        narration=f"Let’s work through {title} and focus on the main idea.",
        continue_lecture=current_slide < total_slides,
    )


def plan_teaching_beat(
    *,
    trigger: Trigger,
    deck_index: str,
    current_slide: int,
    total_slides: int,
    current_reading: str,
    whiteboard_state: list[dict[str, str]] | None = None,
    history: list[dict] | None = None,
    question: str | None = None,
) -> TeachingBeat:
    """Return one validated teaching beat from the professor agent."""
    if not CONFIG.brain.is_live:
        return _fallback_beat(
            trigger=trigger,
            current_slide=current_slide,
            total_slides=total_slides,
            current_reading=current_reading,
            question=question,
        )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _PLANNER_SYSTEM}
    ]
    for turn in (history or [])[-6:]:
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content:
            messages.append({"role": role, "content": content})
    messages.append(
        {
            "role": "user",
            "content": (
                f"Trigger: {trigger}\n"
                f"Student question: {question or '(none)'}\n"
                f"Current slide: {current_slide} of {total_slides}\n\n"
                f"Complete deck index:\n{deck_index}\n\n"
                f"Current slide reading:\n{current_reading}\n\n"
                "Current whiteboard JSON:\n"
                f"{json.dumps(whiteboard_state or [], ensure_ascii=True)}"
            ),
        }
    )

    try:
        response = _client().chat.completions.create(
            model=CONFIG.brain.model,
            messages=messages,
            temperature=0.25,
            max_tokens=700,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or ""
        try:
            data = _extract_json(raw)
        except (json.JSONDecodeError, ValueError):
            repaired = _repair_json(raw)
            if not repaired:
                raise
            data = _extract_json(repaired)
        narration = str(data.get("narration", "")).strip()
        if not narration:
            raise ValueError("agent returned empty narration")
        return TeachingBeat(
            narration=narration,
            actions=_validate_actions(data.get("actions"), total_slides),
            continue_lecture=bool(data.get("continue_lecture", trigger == "continue")),
        )
    except Exception as exc:
        print(f"[agent] planning error: {exc}")
        return _fallback_beat(
            trigger=trigger,
            current_slide=current_slide,
            total_slides=total_slides,
            current_reading=current_reading,
            question=question,
        )
