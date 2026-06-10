"""AI Prof — Gradio app.

Vertical slice #1 + text interjection: upload a lecture PDF, AI Prof reads each
slide as an image (MiniCPM-V) and explains it like a TA (Nemotron, streamed).
Ask a question at any time; it answers using the cached slide reading, then you
continue the walkthrough.
"""

from __future__ import annotations

import io
import html
import threading
import time
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
    from ai_prof.rtc import build_rtc_handler as _build_rtc_handler, tts_speak_full
    _RTC_AVAILABLE = True
except Exception:
    try:
        from ai_prof.rtc import tts_speak_full  # TTS works without fastrtc
    except Exception:
        tts_speak_full = lambda _text: None  # type: ignore
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


def _reading_field(reading: str, name: str) -> str:
    prefix = f"{name}:"
    for line in reading.splitlines():
        if line.upper().startswith(prefix):
            return line[len(prefix):].strip()
    return ""


def _whiteboard_view(state: dict, reading: str | None = None) -> str:
    deck: Deck | None = state["deck"]
    if not deck:
        return (
            '<div class="whiteboard-empty">'
            "<strong>Professor's whiteboard</strong>"
            "<span>Key ideas and worked notes will appear here.</span>"
            "</div>"
        )
    idx = state["index"]
    reading = reading or state["readings"].get(idx, "")
    title = _reading_field(reading, "TITLE") or f"Slide {idx + 1}"
    concepts = _reading_field(reading, "CONCEPTS")
    if not concepts:
        concepts = "Listening for the central idea..."
    return (
        '<div class="whiteboard-sheet">'
        f'<div class="whiteboard-kicker">Working notes · {idx + 1}/{len(deck)}</div>'
        f"<h3>{html.escape(title)}</h3>"
        f"<p>{html.escape(concepts)}</p>"
        '<div class="whiteboard-line"></div>'
        '<span class="whiteboard-hint">The professor can draw here as the lecture develops.</span>'
        "</div>"
    )


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
        return state, img, caption, [], _whiteboard_view(state)

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
    return state, img, caption, [], _whiteboard_view(state, reading0)


_STATUS_READING = (
    '<div style="background:#f0f9ff;border-left:3px solid #3b82f6;padding:6px 12px;'
    'font-size:0.85rem;color:#1e40af;border-radius:0 4px 4px 0">📖 Reading slide…</div>'
)
_STATUS_EXPLAINING = (
    '<div style="background:#f0fdf4;border-left:3px solid #22c55e;padding:6px 12px;'
    'font-size:0.85rem;color:#166534;border-radius:0 4px 4px 0">💬 Explaining…</div>'
)
_STATUS_SPEAKING = (
    '<div style="background:#fdf4ff;border-left:3px solid #a855f7;padding:6px 12px;'
    'font-size:0.85rem;color:#6b21a8;border-radius:0 4px 4px 0">🔊 Professor speaking…</div>'
)
_STATUS_IDLE = ""


def on_explain(state, chat):
    deck: Deck | None = state["deck"]
    if not deck:
        gr.Warning("Upload a lecture PDF first.")
        yield chat, _STATUS_IDLE, None
        return
    idx = state["index"]
    yield chat, _STATUS_READING, None
    reading = _ensure_reading(state, idx)
    yield chat, _STATUS_EXPLAINING, None
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
        yield chat, _STATUS_EXPLAINING, None
    audio = tts_speak_full(acc)
    if audio is not None:
        yield chat, _STATUS_SPEAKING, audio
        time.sleep(len(audio[1]) / audio[0])
    yield chat, _STATUS_IDLE, None


def on_teach_deck(state, chat):
    """Stream explanation for each slide, speak it aloud, then advance."""
    deck: Deck | None = state["deck"]
    if not deck:
        gr.Warning("Upload a lecture PDF first.")
        img, caption = _slide_view(state)
        yield state, img, caption, chat, _STATUS_IDLE, _whiteboard_view(state), None
        return

    start_idx = state["index"]
    for idx in range(start_idx, len(deck)):
        state["index"] = idx
        img, caption = _slide_view(state)
        yield state, img, caption, chat, _STATUS_READING, _whiteboard_view(state), None

        reading = _ensure_reading(state, idx)
        board = _whiteboard_view(state, reading)
        yield state, img, caption, chat, _STATUS_EXPLAINING, board, None

        history = chat
        chat = chat + [{"role": "assistant", "content": ""}]
        acc = ""
        for tok in explain_slide(
            reading,
            slide_no=idx + 1,
            total=len(deck),
            outline=deck.outline(),
            history=history,
        ):
            acc += tok
            chat[-1]["content"] = acc
            yield state, img, caption, chat, _STATUS_EXPLAINING, board, None

        # Speak the full explanation; hold on this slide until audio finishes.
        audio = tts_speak_full(acc)
        if audio is not None:
            sr, pcm = audio
            yield state, img, caption, chat, _STATUS_SPEAKING, board, audio
            time.sleep(len(pcm) / sr)

        if idx < len(deck) - 1:
            state["index"] = idx + 1
            next_img, next_caption = _slide_view(state)
            yield (
                state,
                next_img,
                next_caption,
                chat,
                _STATUS_READING,
                _whiteboard_view(state),
                None,
            )

    yield state, *_slide_view(state), chat, _STATUS_IDLE, _whiteboard_view(state), None


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
    return state, img, caption, _whiteboard_view(state)


def on_transcribe(audio):
    if audio is None:
        return ""
    if not CONFIG.stt.is_live:
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
    client = OpenAI(base_url=CONFIG.stt.base_url, api_key=CONFIG.stt.api_key)
    transcript = client.audio.transcriptions.create(model=CONFIG.stt.model, file=buf)
    return transcript.text


# ----------------------------------------------------------------------------- UI

_BANNER = (
    "⚠️ Running in **mock mode** — set `VISION_BASE_URL` / `BRAIN_BASE_URL` (see `.env.example`) "
    "to plug in real MiniCPM-V + Nemotron."
    if CONFIG.fully_mocked
    else None
)

_CSS = """
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding-inline: 24px !important;
}
.app-title {
    max-width: 1280px;
    margin: 0 auto 18px;
    padding: 4px 2px 8px;
}
.app-title h1 {
    margin: 0 0 4px !important;
    color: #24283b;
    font-size: 2.15rem !important;
    font-weight: 800 !important;
    letter-spacing: -.035em;
}
.app-title p {
    margin: 0 !important;
    color: #697386;
    font-size: 1rem;
}
.workspace-row {
    width: 100%;
    max-width: 1280px;
    margin-inline: auto;
    align-items: stretch !important;
}
.panel-card {
    min-width: 0 !important;
    overflow: hidden;
    gap: 0 !important;
    border: 1px solid #e1e5eb;
    border-radius: 14px;
    background: #fff;
}
.panel-title {
    flex: 0 0 46px !important;
    min-height: 46px !important;
    height: 46px !important;
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 14px !important;
    background: #ecebff;
    border-bottom: 1px solid #dedcff;
    color: #5b55c7;
}
.panel-title p {
    margin: 0 !important;
    font-size: .86rem;
    font-weight: 700;
    line-height: 1.35;
}
.panel-body {
    padding: 12px !important;
}
.teaching-panel { min-height: 655px; }
.slide-frame {
    flex: 0 0 550px !important;
    height: 550px !important;
    min-height: 550px !important;
    border: 0 !important;
    border-radius: 0 !important;
}
.slide-frame img {
    height: 550px !important;
    object-fit: contain !important;
    background: #f8f9fb;
}
.slide-footer {
    padding: 0 12px 12px !important;
}
.slide-caption {
    min-height: 24px;
    margin: 0 !important;
    color: #667085;
}
.slide-controls {
    gap: 10px !important;
}
.slide-controls button {
    min-height: 42px !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}
.nav-button button {
    color: #5b55c7 !important;
    border: 1px solid #d8d6ff !important;
    background: #f7f6ff !important;
}
.explain-button button {
    color: #fff !important;
    border-color: #625ce7 !important;
    background: #625ce7 !important;
    box-shadow: 0 5px 14px rgb(98 92 231 / 20%) !important;
}
.whiteboard {
    flex: 1 1 auto !important;
    min-height: 550px;
    border: 0;
    border-radius: 0;
    overflow: hidden;
    background:
        linear-gradient(#e8edf3 1px, transparent 1px),
        linear-gradient(90deg, #e8edf3 1px, transparent 1px),
        #fbfcfe;
    background-size: 28px 28px;
}
.whiteboard-empty, .whiteboard-sheet {
    min-height: 550px;
    padding: 28px 32px;
    color: #172033;
}
.whiteboard-empty {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    color: #667085;
}
.whiteboard-sheet h3 { font-size: 1.8rem; margin: 24px 0 18px; }
.whiteboard-sheet p { font-size: 1.2rem; line-height: 1.7; max-width: 90%; }
.whiteboard-kicker {
    color: #3157a4;
    font-size: .78rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
}
.whiteboard-line { width: 72px; height: 4px; margin: 30px 0 14px; background: #f4b740; }
.whiteboard-hint { color: #7a8496; font-size: .82rem; }
.bottom-panel {
    min-height: 382px;
}
.transcript-panel {
    flex: 1 1 auto !important;
    height: 334px !important;
    min-height: 334px !important;
    border: 0 !important;
    border-radius: 0 !important;
}
.transcript-panel .placeholder {
    height: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 24px !important;
    color: #8a94a6 !important;
    text-align: center !important;
}
.question-body, .upload-body {
    flex: 1 1 auto !important;
    min-height: 334px;
    padding: 18px !important;
}
.question-body {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
}
.question-row {
    align-items: stretch !important;
    gap: 10px !important;
}
.question-input textarea {
    min-height: 58px !important;
}
.question-input {
    flex: 1 1 auto !important;
}
.ask-button {
    min-width: 96px !important;
    max-width: 110px !important;
}
.ask-button button {
    height: 100% !important;
    min-height: 58px !important;
    font-weight: 750 !important;
}
.mic-label {
    margin: 2px 0 -4px !important;
    color: #697386;
    font-size: .82rem;
    font-weight: 650;
}
.mic-control {
    min-height: 108px !important;
    max-height: 122px !important;
    overflow: hidden !important;
}
.mic-control button[aria-label="Record"],
.mic-control button.record {
    min-width: 150px !important;
    min-height: 58px !important;
    border-radius: 999px !important;
    color: #fff !important;
    background: #625ce7 !important;
    border-color: #625ce7 !important;
    font-size: 1rem !important;
    font-weight: 750 !important;
}
.teach-button {
    margin-top: auto !important;
}
.upload-body {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
}
.upload-control {
    min-height: 210px !important;
}
.upload-copy {
    margin: 0 !important;
    color: #667085;
    font-size: .88rem;
}
@media (max-width: 900px) {
    .gradio-container { padding-inline: 12px !important; }
    .teaching-panel, .bottom-panel { min-width: 100% !important; }
}
"""

with gr.Blocks(title="AI Prof", theme=gr.themes.Soft(), css=_CSS) as demo:
    state = gr.State(_new_state())

    gr.Markdown(
        "# AI Prof\nA live, guided walkthrough of your lecture.",
        elem_classes=["app-title"],
    )
    if _BANNER:
        gr.Markdown(_BANNER)

    with gr.Row(equal_height=True, elem_classes=["workspace-row"]):
        with gr.Column(scale=1, elem_classes=["panel-card", "teaching-panel"]):
            gr.Markdown("Lecture slides", elem_classes=["panel-title"])
            slide_img = gr.Image(
                show_label=False,
                height=470,
                elem_classes=["slide-frame"],
            )
            with gr.Column(elem_classes=["slide-footer"]):
                caption = gr.Markdown("No deck loaded.", elem_classes=["slide-caption"])
                with gr.Row(elem_classes=["slide-controls"]):
                    prev_btn = gr.Button("Previous", elem_classes=["nav-button"])
                    explain_btn = gr.Button(
                        "Explain slide",
                        variant="primary",
                        elem_classes=["explain-button"],
                    )
                    next_btn = gr.Button("Next", elem_classes=["nav-button"])

        with gr.Column(scale=1, elem_classes=["panel-card", "teaching-panel"]):
            gr.Markdown("Whiteboard", elem_classes=["panel-title"])
            whiteboard = gr.HTML(
                value=_whiteboard_view(_new_state()),
                elem_classes=["whiteboard"],
            )

    with gr.Row(equal_height=True, elem_classes=["workspace-row"]):
        with gr.Column(scale=5, elem_classes=["panel-card", "bottom-panel"]):
            gr.Markdown("Lecture transcript", elem_classes=["panel-title"])
            status_strip = gr.HTML(value=_STATUS_IDLE)
            prof_audio = gr.Audio(
                autoplay=True,
                show_label=False,
                visible=False,  # hidden; plays automatically via autoplay
                interactive=False,
            )
            chat = gr.Chatbot(
                show_label=False,
                height=320,
                type="messages",
                layout="panel",
                placeholder=(
                    "Upload a lecture to begin. The professor's explanation "
                    "will appear here as it is spoken."
                ),
                elem_classes=["transcript-panel"],
            )

        with gr.Column(scale=3, elem_classes=["panel-card", "bottom-panel"]):
            gr.Markdown("Ask a question", elem_classes=["panel-title"])
            with gr.Column(elem_classes=["question-body"]):
                with gr.Row(equal_height=True, elem_classes=["question-row"]):
                    question = gr.Textbox(
                        placeholder="Type a question...",
                        show_label=False,
                        lines=1,
                        elem_classes=["question-input"],
                        scale=5,
                    )
                    ask_btn = gr.Button(
                        "Ask",
                        variant="primary",
                        elem_classes=["ask-button"],
                        scale=1,
                    )
                gr.Markdown("Or ask out loud", elem_classes=["mic-label"])
                mic = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    streaming=False,
                    show_label=False,
                    elem_classes=["mic-control"],
                )
                teach_btn = gr.Button(
                    "Teach from current slide",
                    variant="secondary",
                    elem_classes=["teach-button"],
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

        with gr.Column(scale=2, elem_classes=["panel-card", "bottom-panel"]):
            gr.Markdown("Lecture upload", elem_classes=["panel-title"])
            with gr.Column(elem_classes=["upload-body"]):
                pdf = gr.File(
                    label="Drop a PDF to begin",
                    file_types=[".pdf"],
                    type="filepath",
                    height=210,
                    elem_classes=["upload-control"],
                )
                gr.Markdown(
                    "The professor starts at slide 1 and advances automatically. "
                    "Use the slide controls to revisit anything.",
                    elem_classes=["upload-copy"],
                )

    lecture_outputs = [state, slide_img, caption, chat, status_strip, whiteboard, prof_audio]
    pdf.change(
        on_upload,
        [pdf, state],
        [state, slide_img, caption, chat, whiteboard],
    ).then(
        on_teach_deck,
        [state, chat],
        lecture_outputs,
    )

    explain_btn.click(on_explain, [state, chat], [chat, status_strip, prof_audio])
    teach_btn.click(on_teach_deck, [state, chat], lecture_outputs)
    question.submit(on_ask, [question, state, chat], [chat, question])
    ask_btn.click(on_ask, [question, state, chat], [chat, question])
    prev_btn.click(
        on_nav,
        [gr.State(-1), state],
        [state, slide_img, caption, whiteboard],
    )
    next_btn.click(
        on_nav,
        [gr.State(1), state],
        [state, slide_img, caption, whiteboard],
    )
    mic.stop_recording(on_transcribe, inputs=[mic], outputs=[question])


if __name__ == "__main__":
    demo.launch()
