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
**Professor agent loop working:** upload a lecture PDF → render and index the complete
deck → Nemotron plans short teaching beats → the orchestrator executes validated slide
navigation and whiteboard tools → VoxCPM speaks the explanation. Student questions
cancel active narration, can send the professor to a more relevant slide, and can add
supporting notes or equations to the whiteboard.

The current tool set is `goto_slide`, `next_slide`, `prev_slide`, `write_note`,
`write_latex`, and `clear_whiteboard`. Targeted vision via `look_closer` and richer
Excalidraw diagrams remain follow-up work.

Layout: [`app.py`](app.py) (Gradio UI) · [`ai_prof/pdf_utils.py`](ai_prof/pdf_utils.py) (PDF→slides) ·
[`ai_prof/vision.py`](ai_prof/vision.py) (eyes) · [`ai_prof/brain.py`](ai_prof/brain.py) (brain) ·
[`ai_prof/config.py`](ai_prof/config.py) (endpoints / mock fallback).

The inference services run on Modal: Nemotron through vLLM, VoxCPM2 through
vLLM-Omni, and distil-Whisper through a small OpenAI-compatible FastAPI server.
The Gradio frontend remains deployable as a Hugging Face Space.

Run the focused test suite with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Next up (see [IDEATION.md](IDEATION.md)): prepared-deck storage on Hugging Face,
animated structured diagrams, stronger cancellation during browser audio playback,
and targeted `look_closer` vision.

The current target design is documented in [ARCHITECTURE.md](ARCHITECTURE.md), including complete-deck
indexing, the professor tool loop, a preprocessed demo lecture, synchronized speech and whiteboard actions,
and interruption/resume behavior.
