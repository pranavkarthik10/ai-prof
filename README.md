---
title: AI Prof
emoji: 🎓
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: "3.12"
tags:
  - track:backyard
  - sponsor:openbmb
  - sponsor:openai
  - sponsor:nvidia
  - sponsor:modal
  - achievement:offbrand
  - achievement:llama
  - achievement:fieldnotes
---

# AI Prof

A real-time AI teaching assistant for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon)
(**Backyard AI** track).

Upload your real lecture slides and AI Prof walks through them one by one — reading each slide *as an image*
(diagrams, equations, layout, not just scraped text) and explaining it like a TA would in class. Ask a
question at any time and it pauses, answers, and picks back up. Built for a classmate who has the slides but
misses the in-class explanation.

## Demo

- **Live Space:** [AI Prof on Hugging Face](https://huggingface.co/spaces/build-small-hackathon/ai-prof)
- **Demo video:** **TODO before submission — add the public video URL here**
- **Social post:** **TODO before submission — add the public post URL here**
- **Team:** [@pranavkarthik10](https://huggingface.co/pranavkarthik10)

### Judge-friendly walkthrough

1. Upload a lecture PDF.
2. Watch AI Prof index every slide before teaching, giving the agent a map of the complete lecture.
3. Start the lecture and see the professor explain slides aloud while adding notes or equations to the whiteboard.
4. Interrupt with a typed or spoken question. The professor can answer, navigate to a supporting slide, and continue.

## How it works
- **Eyes — [MiniCPM-V](https://huggingface.co/openbmb/MiniCPM-V-4) (~4.1B, GGUF / llama.cpp):** reads each slide image.
- **Brain — [Nemotron 3 Nano](https://huggingface.co/collections/nvidia/nvidia-nemotron-v3):** turns the slide
  reading into a teaching plan, selects tools, explains concepts, and answers interjections.
- **Voice:** VoxCPM2 speaks the professor's response; distil-Whisper transcribes student questions.
- **Runtime:** Modal hosts the four OpenAI-compatible inference services and scales them down when idle.
- **Frontend:** a custom Gradio classroom UI hosted as a Hugging Face Space.

Every model stays under the hackathon's ≤32B-per-model cap.

## Why it is agentic

AI Prof does more than summarize a PDF. Nemotron receives a compact index of the
entire deck, the current slide reading, recent conversation, and whiteboard state.
For every teaching beat it returns validated narration and actions. The orchestrator
can navigate slides, write a note, typeset an equation, clear the board, or continue
the lecture. Student questions cancel the active lecture turn and trigger a new plan
grounded in the full deck.

Current tools: `goto_slide`, `next_slide`, `prev_slide`, `write_note`,
`write_latex`, and `clear_whiteboard`.

## Small-model stack

| Role | Model | Size / serving |
| --- | --- | --- |
| Slide vision | MiniCPM-V-4 | ~4.1B, Q4_K_M GGUF with llama.cpp |
| Professor agent | NVIDIA Nemotron 3 Nano | 30B total / 3B active, vLLM |
| Speech synthesis | VoxCPM2 | vLLM-Omni |
| Speech recognition | distil-Whisper large-v3 | faster-whisper |

## Run it
```bash
uv venv --python 3.12 && uv pip install -r requirements.txt
.venv/bin/python app.py        # opens the Gradio UI
```
Out of the box it runs in **mock mode** (no weights) so you can drive the full UX —
upload a PDF, watch it stream explanations, ask questions. To plug in the real models,
copy `.env.example` to `.env` and point `VISION_BASE_URL` / `BRAIN_BASE_URL` at your
OpenAI-compatible endpoints (llama.cpp `llama-server` for MiniCPM-V, vLLM/llama.cpp for Nemotron).

MiniCPM-V is included as a Modal service:

```bash
modal run modal_app_vision.py::download_model
modal run modal_app_vision.py::warm
modal deploy modal_app_vision.py
```

Set `VISION_BASE_URL` to the deployed `serve` URL. The service uses MiniCPM-V-4
Q4_K_M plus its F16 vision projector on one L4 and scales down after five idle
minutes.

## Pre-indexed deck cache

AI Prof hashes each uploaded PDF and caches its rendered slide images, extracted
text, MiniCPM-V readings, and complete deck index. Re-uploading the same deck with
the same vision model and DPI skips indexing entirely.

The cache is local by default. To share a prepared demo deck across local testing
and the public Space, create a Hugging Face dataset repo and configure:

```bash
HF_DECK_CACHE_REPO=pranavkarthik10/ai-prof-decks
HF_TOKEN=hf_...
HF_DECK_CACHE_WRITE=true
```

Upload the demo PDF once while writes are enabled. The processed deck is stored
under `decks/<content-key>/` in the dataset. For the public Space, set
`HF_DECK_CACHE_REPO` but leave `HF_DECK_CACHE_WRITE=false`; judges can then load
that exact deck from the **Prepared lectures** picker without uploading the PDF
or granting the app write access.

Do not enable remote writes for arbitrary uploads to a public dataset. Processed
slides can contain the original lecture material. Use a private dataset for personal
testing or only prewarm material you have permission to publish.
