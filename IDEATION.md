# Build Small Hackathon — Ideation & Decisions

> Working notes from ideation. Project: **AI Prof** (primary submission).
> Hackathon: https://huggingface.co/build-small-hackathon · Field guide: https://build-small-hackathon-field-guide.hf.space/

## Hackathon constraints (the rules that shape everything)
- **Model size: ≤32B parameters PER MODEL** (not aggregate). You can freely combine multiple
  small models as long as each one individually stays under the cap. ("not just active params")
- **No sponsor exclusivity** — a sponsor's model just has to be *a core part of the experience*;
  you may mix in other providers' models.
- **Platform:** must be built in **Gradio**, hosted as a **Hugging Face Space**.
- **Deliverables:** working Space + **demo video** + **social media post**.
- **Timeline:** hack window June 5–15, 2026.
- **Credits:** OpenAI $100 is for **Codex (their coding agent)**, *not* API/inference. Modal $250, HF $20.

## Tracks
- **Backyard AI** — solve a genuine problem for *someone you personally know*. Judged on specificity,
  **actual user adoption**, and appropriate model fit. ← our track.
- **Thousand Token Wood (TTW)** — delightful/unconventional, AI must be load-bearing (game, story, art).

## Prize surface (it stacks — one app can hit many)
Main tracks ($18k): 1st–4th per track + Community Choice ($2k).
Sponsor awards:
- **OpenBMB $10k** — build with MiniCPM (incl. vision MiniCPM-V / omni MiniCPM-o).
- **OpenAI $10k** — won via **Codex-attributed commits** in the repo/Space (build-tool track, model-agnostic).
- **NVIDIA** — two RTX 5080 GPUs; build with **Nemotron**. One for "best space", one for community engagement (likes).
- **Modal $20k credits** — use Modal for dev/runtime, note in README.
Special awards ($8k): Bonus Quest Champion $2k · Off-Brand (best custom UI via `gr.Server`) $1.5k ·
Tiny Titan (best app on ≤4B model) $1.5k · Best Demo $1k · Best Agent $1k · Judges' Wildcard $1k.
Six merit badges: Off the Grid (no cloud APIs) · Well-Tuned (publish fine-tune) · Off-Brand (custom UI) ·
Llama Champion (llama.cpp runtime) · Sharing is Caring (publish agent trace) · Field Notes (build blog).

## Decision: build the AI Prof (Backyard AI)
**Problem (real):** a specific classmate finds that having the slides isn't enough — they're static and
lack the in-class explanation. Test with **real, anonymized lecture slides** from our class.

**Core loop:** upload lecture PDF → model reads each slide *as an image* → explains it like a TA,
streamed in real time → classmate can **interject** with a question at any moment.

### Two-model architecture (stacks OpenBMB + NVIDIA, cleanly justified)
- **MiniCPM-V (~4.1B, GGUF)** = *the eyes.* Reads each slide as an image (diagrams, equations, layout —
  not just scraped text). Core → satisfies **OpenBMB**. Runs via **llama.cpp** → Llama Champion badge.
- **Nemotron 3 Nano (9B, or 30B-A3B MoE = only ~3.6B active → fast decode)** = *the brain.* Turns the
  slide reading into the spoken explanation and answers interjections. Reasoning/agentic → **NVIDIA** fit.
- Per-model cap means no param-budget tension between the two. Division of labor (see/explain) survives
  the "core part of the experience" test → not sponsor-stuffing.

### Prize map for this one app
Backyard placement · OpenBMB $10k · NVIDIA RTX 5080 · OpenAI $10k (Codex commits) · Modal $20k (self-host) ·
Off-Brand $1.5k+badge (custom whiteboard UI) · Llama Champion · Off the Grid · Sharing is Caring (publish a
teaching-session trace) · Field Notes (blog) · Best Demo. → realistically 6+ surfaces.

### Scope discipline (vertical slice order)
1. Slide upload → MiniCPM-V reads → Nemotron explains, **streamed** in a simple custom UI. *(Submittable alone.)*
2. Text interjection (pause, ask, resume).
3. Whiteboard — model emits **Mermaid / Excalidraw JSON** (structured; small models do this reliably).
   Avoid freeform tldraw generation — that's the day-eating trap.
4. Voice / TTS — only with slack.

## Real-time architecture (the open design area — wants more thought)
Goal: feel like a *live* lecture — explanation streams as if the prof is talking through each slide,
smooth slide-to-slide, interruptible mid-sentence.
- **Token streaming:** stream Nemotron output to the UI (Gradio generators) so it appears as it's generated.
- **Pipeline overlap / prefetch:** the heavy step is the MiniCPM-V vision pass. While narrating slide N,
  run vision + Nemotron prefill for slide N+1 in the background (producer/consumer queue) → instant transitions.
- **Two-stage per slide, cached:** (a) MiniCPM-V → a structured "slide reading" (text + diagram desc + equations),
  computed once and cached; (b) Nemotron → the explanation. Interjections reuse cached (a), never re-run vision.
- **Interjection = interrupt + branch:** need a *cancellable* generation (async cancel token / threading.Event).
  On input: stop current stream, answer using the cached slide reading + history as context, then resume.
- **Streaming TTS (optional):** chunk explanation into sentences, TTS each as generated, play sequentially.
  Barge-in (interrupt-on-speech) is hard mode — defer.
- **Session state:** { slides[], current_index, cached_slide_readings{}, conversation_history }.
- **Hosting tradeoff for low latency:** free HF Space hardware likely too slow for 2 models in real time.
  Lean: **Modal GPU as inference backend** (uses the Modal track) + Gradio Space as frontend. Self-hosting
  (not an external inference API) still supports the *Off the Grid* badge.

## Voice (STT + TTS) — makes it feel like *class*, but it's the deepest rabbit hole
Two more tiny, on-HF models (free under the ≤32B-*per-model* cap; FAQ uses "a 7B speech model" as its example).
Keep both **open + self-hosted** (not ElevenLabs/Deepgram/OpenAI *API*) → protects the **Off the Grid** badge.

**Pipeline:** TTS (out) speaks Nemotron's streamed explanation; STT (in) transcribes the spoken interjection
into text for Nemotron; **VAD** is the referee that detects *when* the classmate starts talking → triggers barge-in.

- **STT (interjections) — pick: Whisper, or Moonshine for speed.**
  - `faster-whisper` / `distil-whisper` (large-v3 ~1.5B, or base/small for latency) — accurate, OpenAI open weights.
  - **Moonshine** (~tiny) — built for real-time/on-device, faster time-to-text on short clips.
  - Interjections are short → small model is fine; *time-to-text*, not accuracy, is the bottleneck. Start with
    `faster-whisper-base`, switch to Moonshine only if laggy.
- **TTS (narration) — pick: Kokoro, fallback Piper.**
  - **Kokoro-82M** (`hexgrad/Kokoro-82M`) — 82M, good quality, fast time-to-first-audio, streamable. Sweet spot.
  - **Piper** — even lower latency, CPU-friendly, slightly more robotic. Use if speech layer isn't on GPU.
  - Stream sentence-by-sentence: synthesize + play each sentence as Nemotron emits it (audio ~1 sentence behind gen).
- **Barge-in / turn-taking — use FastRTC** (`fastrtc`, the Gradio/HF WebRTC stack). Gives low-latency mic+playback
  over WebRTC, built-in **VAD turn detection** (`ReplyOnPause`), and the hook to cancel playback + generation the
  instant the user speaks. Avoids hand-rolling silence detection; reuses our cancellable-generation design.
  - Loop: narrate (TTS) → Silero VAD hears speech → kill TTS + cancel Nemotron → buffer to end-of-speech → STT →
    answer using the **cached slide reading** as context → TTS answer → resume narration.
- **Latency budget (what "feels live" needs):** minimize time-to-first-audio. STT small (~100–300ms) + Nemotron
  MoE fast prefill + Kokoro first chunk (~tens of ms). Run STT/TTS/VAD on the **same Modal GPU** as Nemotron
  (or CPU for Piper+Silero) to avoid network hops.
- **Scope ladder (don't let voice eat the week):**
  1. **TTS narration only** — Prof talks, classmate *types* to interject. Low risk, already feels real-time.
  2. **Push-to-talk interject** — hold key / tap to ask aloud → STT. No VAD/barge-in. ~90% of magic, ~30% of work.
  3. **Full-duplex barge-in** via FastRTC + VAD — only once 1–2 are solid. The 2→3 jump is where time goes.
- **Model count after voice:** MiniCPM-V (vision) + Nemotron (brain) + Whisper/Moonshine (STT) + Kokoro/Piper (TTS)
  + Silero VAD. Multi-model is explicitly blessed; more "appropriate model fit" surface, but more integration.

## Backburner ideas (TTW — only if a 2nd submission is feasible)
- **Emotion-driven TRPG / choose-your-own-adventure:** free text → LLM updates structured NPC emotional
  state (trust/fear/affection…) → state steers the story. Best Agent + Sharing is Caring (emotion-delta traces).
- **AI Garlic Phone:** message/drawing passed down a chain of model personas, mutating; needs a reveal payoff.
  If a drawing is passed, pulls in vision = OpenBMB again.
- **Sketch-to-story adventure:** you draw your action, MiniCPM-V interprets the doodle, the world reacts.
  Fuses vision + emotion mechanic. Strong demo, only-AI-can-do-this.
- **Model-vs-model battle → traces for training** (browser-brawl-style, different domain): most technically
  impressive (Best Agent + Well-Tuned + Sharing is Caring) but research-shaped; the training loop can eat the week.

## Key model references
- MiniCPM-V-4 (4.1B, multimodal): https://huggingface.co/openbmb/MiniCPM-V-4
- MiniCPM-V-4 GGUF (llama.cpp): https://huggingface.co/openbmb/MiniCPM-V-4-gguf
- Nemotron 3 Nano collection: https://huggingface.co/collections/nvidia/nvidia-nemotron-v3
- Nemotron 3 Nano 4B: https://huggingface.co/blog/nvidia/nemotron-3-nano-4b
- Nemotron 3 Nano 30B-A3B: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
