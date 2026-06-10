"""AI Prof — Gradio app.

Vertical slice #1 + text interjection: upload a lecture PDF, AI Prof reads each
slide as an image (MiniCPM-V) and explains it like a TA (Nemotron, streamed).
Ask a question at any time; it answers using the cached slide reading, then you
continue the walkthrough.
"""

from __future__ import annotations

import io
import threading
import uuid
import wave

import numpy as np
import gradio as gr
from openai import OpenAI

from ai_prof.brain import answer_question, explain_slide
from ai_prof.config import CONFIG
from ai_prof.pdf_utils import Deck, render_pdf
from ai_prof.vision import read_slide

# ---------------------------------------------------------------------------
# Optional WebRTC real-time voice layer (requires: pip install "fastrtc[vad]")
# When fastrtc is installed, replace the push-to-talk gr.Audio mic below with
# a gr.WebRTC component and wire it through build_rtc_handler — see
# ai_prof/rtc.py for full wiring instructions.
# ---------------------------------------------------------------------------
try:
    from ai_prof.rtc import build_rtc_handler as _build_rtc_handler
    _RTC_AVAILABLE = True
except (ImportError, RuntimeError):
    _RTC_AVAILABLE = False

# module-level pre-read cache: {session_id: {slide_idx: reading_str}}
_preread: dict[str, dict[int, str]] = {}
_preread_lock = threading.Lock()


def _new_state() -> dict:
    return {"deck": None, "index": 0, "readings": {}, "session_id": str(uuid.uuid4())}


def _ensure_reading(state: dict, idx: int) -> str:
    """Read + cache the slide once; reused by both explanation and Q&A."""
    sid = state["session_id"]
    with _preread_lock:
        if sid in _preread and idx in _preread[sid]:
            result = _preread[sid][idx]
            state["readings"][idx] = result
            return result
    cache = state["readings"]
    if idx not in cache:
        slide = state["deck"].slides[idx]
        cache[idx] = read_slide(slide.image_path, text_layer=slide.text)
    return cache[idx]


def _preload_background(deck: Deck, session_id: str) -> None:
    for idx in range(1, len(deck)):
        with _preread_lock:
            if session_id not in _preread:
                return  # session invalidated by a new upload
        slide = deck.slides[idx]
        reading = read_slide(slide.image_path, text_layer=slide.text)
        with _preread_lock:
            if session_id not in _preread:
                return
            _preread[session_id][idx] = reading


def _slide_view(state: dict):
    deck: Deck | None = state["deck"]
    if not deck:
        return None, "No deck loaded."
    idx = state["index"]
    slide = deck.slides[idx]
    return slide.image_path, f"Slide {idx + 1} / {len(deck)}"


# ----------------------------------------------------------------------------- handlers


def on_upload(pdf_file, state):
    old_sid = state.get("session_id")
    state = _new_state()
    sid = state["session_id"]

    if old_sid:
        with _preread_lock:
            _preread.pop(old_sid, None)

    if pdf_file is None:
        img, caption = _slide_view(state)
        return state, img, caption, []

    deck = render_pdf(pdf_file, dpi=CONFIG.slide_dpi)
    state["deck"] = deck

    with _preread_lock:
        _preread[sid] = {}

    slide0 = deck.slides[0]
    reading0 = read_slide(slide0.image_path, text_layer=slide0.text)
    state["readings"][0] = reading0
    with _preread_lock:
        _preread[sid][0] = reading0

    if len(deck) > 1:
        t = threading.Thread(target=_preload_background, args=(deck, sid), daemon=True)
        t.start()

    img, caption = _slide_view(state)
    return state, img, caption, []


_STATUS_READING = (
    '<div style="background:#f0f9ff;border-left:3px solid #3b82f6;padding:6px 12px;'
    'font-size:0.85rem;color:#1e40af;border-radius:0 4px 4px 0">📖 Reading slide…</div>'
)
_STATUS_EXPLAINING = (
    '<div style="background:#f0fdf4;border-left:3px solid #22c55e;padding:6px 12px;'
    'font-size:0.85rem;color:#166534;border-radius:0 4px 4px 0">💬 Explaining…</div>'
)
_STATUS_IDLE = ""


def on_explain(state, chat):
    deck: Deck | None = state["deck"]
    if not deck:
        gr.Warning("Upload a lecture PDF first.")
        yield chat, _STATUS_IDLE
        return
    idx = state["index"]
    yield chat, _STATUS_READING
    reading = _ensure_reading(state, idx)
    yield chat, _STATUS_EXPLAINING
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
        yield chat, _STATUS_EXPLAINING
    yield chat, _STATUS_IDLE


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


def on_transcribe(audio):
    if audio is None:
        return ""
    if not CONFIG.brain.is_live:
        return "[voice input]"
    sr, data = audio
    if data is None or len(data) == 0:
        return ""
    buf = io.BytesIO()
    if data.dtype != np.int16:
        data = (data * 32767).astype(np.int16)
    if data.ndim > 1:
        data = data[:, 0]
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    buf.seek(0)
    buf.name = "audio.wav"
    base = CONFIG.brain.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    client = OpenAI(base_url=base, api_key=CONFIG.brain.api_key)
    transcript = client.audio.transcriptions.create(model="whisper-1", file=buf)
    return transcript.text


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
            status_strip = gr.HTML(value=_STATUS_IDLE)
            chat = gr.Chatbot(label="AI Prof", height=500)
            with gr.Row():
                question = gr.Textbox(
                    placeholder="Ask a question… (or use the mic 🎙️)",
                    show_label=False,
                    submit_btn=True,
                    scale=5,
                )
                mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=False,
                    show_label=False,
                    scale=1,
                )
            # TODO: wire fastrtc when installed — replace `mic` above with:
            #
            #   if _RTC_AVAILABLE:
            #       _rtc_handler = _build_rtc_handler(state_getter=lambda: state.value)
            #       webrtc = gr.WebRTC(
            #           label="Live voice (real-time)",
            #           rtc_configuration=_rtc_handler.rtc_configuration,
            #           mode="send-receive",
            #       )
            #       webrtc.stream(
            #           _rtc_handler,
            #           inputs=[webrtc, state],
            #           outputs=[webrtc],
            #           time_limit=120,
            #       )
            #
            # See ai_prof/rtc.py for the full pipeline:
            #   student mic → STT (/v1/audio/transcriptions)
            #               → brain.answer_question (streamed text)
            #               → TTS (/v1/audio/speech, PCM chunks)
            #               → student speaker (sub-second latency)

    pdf.change(on_upload, [pdf, state], [state, slide_img, caption, chat]).then(
        on_explain, [state, chat], [chat, status_strip]
    )

    explain_btn.click(on_explain, [state, chat], [chat, status_strip])
    question.submit(on_ask, [question, state, chat], [chat, question])
    prev_btn.click(on_nav, [gr.State(-1), state], [state, slide_img, caption])
    next_btn.click(on_nav, [gr.State(1), state], [state, slide_img, caption])
    mic.stop_recording(on_transcribe, inputs=[mic], outputs=[question])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
