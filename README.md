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

## Status
Early build. See [IDEATION.md](IDEATION.md) for the design, architecture notes, and prize strategy.
