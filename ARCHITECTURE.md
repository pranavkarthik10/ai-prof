# AI Prof Architecture

This document captures the intended production and demo architecture for AI Prof.
The core experience is a professor-like agent that understands an entire lecture,
controls the visible slide and whiteboard, speaks naturally, and responds to student
interruptions without losing its place.

## Deployment

```text
Hugging Face Space
  Gradio UI
  session state
  professor orchestrator
  slide and whiteboard rendering
            |
            v
Modal inference services
  MiniCPM-V slide vision
  Nemotron professor agent
  VoxCPM text-to-speech
  Whisper or Moonshine speech-to-text
            |
            v
Hugging Face storage
  published demo deck
  processed deck manifests
  slide images and readings
```

- **Hugging Face Space:** public Gradio application and lightweight orchestration.
- **Modal:** GPU-backed, self-hosted model inference that can scale down when idle.
- **Local development:** the same Gradio app in mock mode or pointed at local model
  servers.
- **Hugging Face storage:** persistent processed lectures, including one polished
  public demo deck that works immediately for judges and visitors.

## Deck Preparation

The professor should not begin teaching after only slide 1 is processed. Navigation
and question answering depend on knowing what exists across the whole lecture.

On upload:

1. Hash the PDF to identify an existing processed deck.
2. If cached, load its manifest and slide assets.
3. Otherwise, render every page and extract the PDF text layer.
4. Run MiniCPM-V over every slide.
5. Build a compact deck index from all slide readings.
6. Persist the processed result when appropriate.
7. Enable the lecture only when the complete index is ready.

Gradio displays preparation progress while this runs. It does not need to hold the
heavy processing itself; it can call Modal jobs and stream their progress.

### Processed Deck Format

```text
decks/<pdf_sha256>/
  manifest.json
  source.pdf
  slides/
    001.png
    002.png
  readings/
    001.json
    002.json
```

Each index entry should remain compact enough to keep the entire deck map in the
agent context:

```json
{
  "slide": 7,
  "title": "Convolution",
  "summary": "Applying a kernel across an image",
  "concepts": ["kernel", "stride", "weighted sum"],
  "equations": ["g(x,y) = sum_i sum_j h(i,j)f(x-i,y-j)"],
  "visuals": ["A 3 by 3 kernel moving across a pixel grid"]
}
```

The agent receives the complete compact index, but only the full reading for the
current or specifically retrieved slides.

## Professor Agent

The model should make decisions at meaningful teaching boundaries, not once per
sentence or drawing stroke. One agent turn produces a short **teaching beat**:

```json
{
  "narration": "Imagine this grid is a small patch of the image.",
  "actions": [
    {"tool": "draw_grid", "args": {"rows": 3, "cols": 3}, "at": 0.4}
  ],
  "next": "continue"
}
```

The orchestrator executes the actions and speech. After the beat completes, it asks
the agent what to do next.

### Agent Context

Each decision receives:

- Complete compact deck index
- Current slide number and full cached slide reading
- Current whiteboard state
- Recent conversation and teaching beats
- Saved lecture position
- Trigger: continue lecture or student question

### Tools

- `goto_slide(index)` - move to the best supporting slide
- `next_slide()` and `prev_slide()` - ordinary navigation
- `look_closer(question)` - ask MiniCPM-V to inspect the current slide for a
  specific visual detail; wire this after the core loop
- `write_latex(expression, position)` - place a typeset equation
- `draw_diagram(spec)` - render structured Excalidraw-style primitives
- `clear_whiteboard()` - reset the board when the visual context changes
- `highlight_region(bbox)` - optional later enhancement

Tool calls and their results should be logged as a publishable teaching-session
trace.

## Student Interruption

The lecture is controlled by an explicit state machine:

```text
NARRATING
  -> student begins speaking
INTERRUPTING
  -> stop TTS and cancel current generation
LISTENING
  -> capture speech until push-to-talk release or VAD pause
THINKING
  -> transcribe, search deck index, choose slide and visual support
ANSWERING
  -> navigate or draw when useful, then speak the answer
RESUMING
  -> continue from the saved teaching position
```

When a student asks a question, the agent first decides whether the current slide
is sufficient. It may:

- Answer on the current slide
- Navigate to a more relevant earlier or later slide
- Inspect a slide more closely
- Draw an explanation on the whiteboard
- Combine navigation and drawing

The agent decides whether to return to the previous slide afterward. Automatically
returning every time would make the lecture feel mechanical.

The orchestrator saves the interrupted teaching beat and sentence position. For the
first implementation, resuming at the beginning of that beat is acceptable and much
simpler than resuming at an exact audio sample.

## Speech

- **TTS:** VoxCPM for professor narration.
- **STT:** faster-whisper or Moonshine for short student questions.
- **Transport and turn detection:** FastRTC, initially push-to-talk and later VAD
  barge-in.

Narration should be synthesized in short sentence or beat-sized chunks. This keeps
latency low and gives the orchestrator clean cancellation boundaries.

## Whiteboard

Avoid unrestricted free-form drawing as the primary path. It requires too many model
calls and is difficult to synchronize or reproduce.

Use structured operations:

- LaTeX for equations
- Excalidraw-style primitives for boxes, arrows, labels, grids, and highlights
- Optional Mermaid for diagrams where automatic layout is useful
- Manim only for prepared showcase animations, not the live agent loop

The model emits one structured drawing plan per teaching beat. The frontend animates
the resulting primitives locally, so drawing does not require another inference call
for every stroke.

### Speech and Drawing Synchronization

Each action may include an approximate offset relative to the narration:

```json
{
  "narration": "The center pixel is replaced using all nine neighbors.",
  "actions": [
    {"tool": "highlight_cell", "args": {"row": 1, "column": 1}, "at": 0.2},
    {"tool": "write_latex", "args": {"expression": "1/9 sum pixels"}, "at": 2.1}
  ]
}
```

The orchestrator:

1. Starts TTS for the teaching beat.
2. Executes visual actions at their approximate offsets.
3. Waits for speech and drawing to finish.
4. Requests the next teaching beat.

This produces the feeling of talking while drawing without making an agent call per
stroke.

## Demo Path

The public Space should offer two entry points:

1. **Try the prepared lecture:** loads an already processed, polished deck from
   Hugging Face storage immediately.
2. **Upload your own lecture:** preprocesses the complete deck, displays progress,
   then starts the professor.

The prepared lecture is the reliable judging and demo-video path. User uploads prove
the system generalizes without making the first experience depend on a long vision
preprocessing wait.

## Implementation Order

1. Complete-deck manifest and compact index
2. Prepared demo deck loading from Hugging Face storage
3. Professor teaching-beat schema and tool executor
4. Slide navigation tools
5. Structured whiteboard tools and local animation
6. VoxCPM narration with cancellable chunks
7. Push-to-talk interruption and STT
8. Automatic slide retrieval during questions
9. VAD barge-in
10. Targeted `look_closer` vision calls

