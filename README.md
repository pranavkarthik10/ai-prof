# AI Prof

A real-time AI teaching assistant for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon)
(**Backyard AI** track).

Upload your real lecture slides and AI Prof walks through them one by one — reading each slide *as an image*
(diagrams, equations, layout, not just scraped text) and explaining it like a TA would in class. Ask a
question at any time and it pauses, answers, and picks back up. Built for a classmate who has the slides but
misses the in-class explanation.

## How it works
- **Eyes — [MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V-4) (~4.1B, GGUF / llama.cpp):** reads each slide image.
- **Brain — [Nemotron 3 Nano](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3):** turns the slide
  reading into a streamed, real-time explanation and answers interjections.
- **Frontend:** Gradio (custom UI), hosted as a Hugging Face Space.

Both models stay under the hackathon's ≤32B-per-model cap.

## Run it
```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
.venv/bin/python app.py        # opens the Gradio UI
```
Out of the box it runs in **mock mode** (no weights) so you can drive the full UX —
upload a PDF, watch it stream explanations, ask questions. To plug in the real models,
copy `.env.example` to `.env` and point `VISION_BASE_URL` / `BRAIN_BASE_URL` at your
OpenAI-compatible endpoints (llama.cpp `llama-server` for MiniCPM-V, vLLM/llama.cpp for Nemotron).

## Status
**Initial vertical slice working:** upload a lecture PDF → each page rendered as an image →
MiniCPM-V reads the slide → Nemotron streams a TA-style explanation → text interjections
(ask a question, grounded in the cached slide reading). Slide readings are cached so
questions never re-run the vision pass.

Layout: [`app.py`](app.py) (Gradio UI) · [`ai_prof/pdf_utils.py`](ai_prof/pdf_utils.py) (PDF→slides) ·
[`ai_prof/vision.py`](ai_prof/vision.py) (eyes) · [`ai_prof/brain.py`](ai_prof/brain.py) (brain) ·
[`ai_prof/config.py`](ai_prof/config.py) (endpoints / mock fallback).

Next up (see [IDEATION.md](IDEATION.md)): pipeline prefetch for instant transitions, `look_closer`
real-time vision, whiteboard (Mermaid/Excalidraw), then voice (TTS → push-to-talk → barge-in).
