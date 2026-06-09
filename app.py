"""AI Prof — Gradio app.

Vertical slice #1 + text interjection: upload a lecture PDF, AI Prof reads each
slide as an image (MiniCPM-V) and explains it like a TA (Nemotron, streamed).
Ask a question at any time; it answers using the cached slide reading, then you
continue the walkthrough.
"""

from __future__ import annotations

import gradio as gr

from ai_prof.brain import answer_question, explain_slide
from ai_prof.config import CONFIG
from ai_prof.pdf_utils import Deck, render_pdf
from ai_prof.vision import read_slide


def _new_state() -> dict:
    return {"deck": None, "index": 0, "readings": {}}


def _ensure_reading(state: dict, idx: int) -> str:
    """Read + cache the slide once; reused by both explanation and Q&A."""
    cache = state["readings"]
    if idx not in cache:
        slide = state["deck"].slides[idx]
        cache[idx] = read_slide(slide.image_path, text_layer=slide.text)
    return cache[idx]


def _slide_view(state: dict):
    deck: Deck | None = state["deck"]
    if not deck:
        return None, "No deck loaded."
    idx = state["index"]
    slide = deck.slides[idx]
    return slide.image_path, f"Slide {idx + 1} / {len(deck)}"


# ----------------------------------------------------------------------------- handlers


def on_upload(pdf_file, state):
    state = _new_state()
    if pdf_file is None:
        img, caption = _slide_view(state)
        return state, img, caption, []
    state["deck"] = render_pdf(pdf_file, dpi=CONFIG.slide_dpi)
    img, caption = _slide_view(state)
    return state, img, caption, []


def on_explain(state, chat):
    deck: Deck | None = state["deck"]
    if not deck:
        gr.Warning("Upload a lecture PDF first.")
        yield chat
        return
    idx = state["index"]
    reading = _ensure_reading(state, idx)
    chat = chat + [{"role": "assistant", "content": ""}]
    acc = ""
    for tok in explain_slide(
        reading,
        slide_no=idx + 1,
        total=len(deck),
        outline=deck.outline(),
        history=chat,
    ):
        acc += tok
        chat[-1]["content"] = acc
        yield chat


def on_ask(question, state, chat):
    deck: Deck | None = state["deck"]
    if not deck:
        gr.Warning("Upload a lecture PDF first.")
        yield chat, ""
        return
    question = (question or "").strip()
    if not question:
        yield chat, ""
        return
    idx = state["index"]
    reading = _ensure_reading(state, idx)
    chat = chat + [{"role": "user", "content": question}, {"role": "assistant", "content": ""}]
    acc = ""
    for tok in answer_question(question, reading=reading, slide_no=idx + 1, history=chat):
        acc += tok
        chat[-1]["content"] = acc
        yield chat, ""


def on_nav(delta, state):
    deck: Deck | None = state["deck"]
    if deck:
        state["index"] = max(0, min(len(deck) - 1, state["index"] + delta))
    img, caption = _slide_view(state)
    return state, img, caption


# ----------------------------------------------------------------------------- UI

_BANNER = (
    "⚠️ Running in **mock mode** — set `VISION_BASE_URL` / `BRAIN_BASE_URL` (see `.env.example`) "
    "to plug in real MiniCPM-V + Nemotron."
    if CONFIG.fully_mocked
    else None
)

with gr.Blocks(title="AI Prof") as demo:
    state = gr.State(_new_state())

    gr.Markdown("# 🎓 AI Prof\nUpload your lecture slides and get walked through them, one by one.")
    if _BANNER:
        gr.Markdown(_BANNER)

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            slide_img = gr.Image(label="Slide", show_label=False, height=460)
            caption = gr.Markdown("No deck loaded.")
            with gr.Row():
                prev_btn = gr.Button("◀ Prev")
                explain_btn = gr.Button("▶ Explain this slide", variant="primary")
                next_btn = gr.Button("Next ▶")
            pdf = gr.File(label="Lecture PDF", file_types=[".pdf"], type="filepath")

        with gr.Column(scale=2):
            chat = gr.Chatbot(label="AI Prof", height=520)
            question = gr.Textbox(
                placeholder="Interject — ask a question about this slide…",
                show_label=False,
                submit_btn=True,
            )

    # Upload -> render deck, then auto-explain slide 1 for the live-lecture feel.
    pdf.change(on_upload, [pdf, state], [state, slide_img, caption, chat]).then(
        on_explain, [state, chat], [chat]
    )

    explain_btn.click(on_explain, [state, chat], [chat])
    question.submit(on_ask, [question, state, chat], [chat, question])
    prev_btn.click(on_nav, [gr.State(-1), state], [state, slide_img, caption])
    next_btn.click(on_nav, [gr.State(1), state], [state, slide_img, caption])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
