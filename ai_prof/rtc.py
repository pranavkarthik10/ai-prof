"""Real-time WebRTC voice handler for AI Prof.

Architecture
------------
Student mic  →  fastrtc ReplyOnPause  →  stt_transcribe()
                                       →  brain.answer_question()   (streamed text)
                                       →  tts_speak()               (streamed PCM)
                                       →  student speaker

The handler is exposed as ``build_rtc_handler()`` which returns a
``fastrtc.ReplyOnPause`` instance ready to be wired into a ``gr.WebRTC``
component:

    from ai_prof.rtc import build_rtc_handler
    handler = build_rtc_handler(state_getter=lambda: app_state)
    webrtc = gr.WebRTC(rtc_configuration=handler.rtc_configuration)
    webrtc.stream(handler, inputs=[webrtc, state], outputs=[webrtc])

If fastrtc is not installed the module still imports cleanly — every public
symbol is present but raises ``RuntimeError("fastrtc not installed")`` when
called, so app.py can do a safe conditional import.
"""

from __future__ import annotations

import io
import base64
import threading
import wave
from typing import Callable, Generator

import numpy as np

# ---------------------------------------------------------------------------
# Optional fastrtc import — degrade gracefully when not installed.
# ---------------------------------------------------------------------------
try:
    from fastrtc import ReplyOnPause, SileroVad  # type: ignore
    _FASTRTC_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTRTC_AVAILABLE = False
    ReplyOnPause = None  # type: ignore
    SileroVad = None  # type: ignore

from ai_prof.brain import answer_question
from ai_prof.config import CONFIG, ModelConfig

# ---------------------------------------------------------------------------
# STT helpers
# ---------------------------------------------------------------------------

_MIN_SPEECH_RMS = 0.005


def _has_speech(audio_pcm: np.ndarray) -> bool:
    if audio_pcm.size == 0:
        return False
    pcm = audio_pcm.astype(np.float32)
    if np.issubdtype(audio_pcm.dtype, np.integer):
        pcm /= float(np.iinfo(audio_pcm.dtype).max)
    return float(np.sqrt(np.mean(np.square(pcm)))) >= _MIN_SPEECH_RMS


def _stt_transcribe_live(audio_pcm: np.ndarray, sample_rate: int = 16_000) -> str:
    """Transcribe a NumPy PCM array via OpenAI-compatible /v1/audio/transcriptions.

    Returns the transcript string, or an empty string on failure / mock mode.
    """
    stt_cfg: ModelConfig = CONFIG.stt

    if not stt_cfg.is_live:
        # Mock: return a placeholder so the rest of the pipeline can be tested.
        return "[STT mock — set STT_BASE_URL to transcribe real audio]"
    if not _has_speech(audio_pcm):
        return ""

    try:
        import openai  # already in requirements

        # Encode PCM → in-memory WAV bytes
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            pcm16 = (audio_pcm * 32767).astype(np.int16)
            wf.writeframes(pcm16.tobytes())
        buf.seek(0)
        buf.name = "audio.wav"  # openai SDK reads .name for MIME sniffing

        client = openai.OpenAI(
            base_url=stt_cfg.openai_base_url,
            api_key=stt_cfg.api_key,
        )
        transcript = client.audio.transcriptions.create(
            model=stt_cfg.model,
            file=buf,
            response_format="text",
        )
        return str(transcript).strip()
    except Exception as exc:  # pragma: no cover
        print(f"[rtc] STT error: {exc}")
        return ""


# ---------------------------------------------------------------------------
# TTS helpers
# ---------------------------------------------------------------------------

# VoxCPM2 Voice Design: prepend a description in parentheses to steer the
# synthesised voice without requiring a reference audio clip.
_PROF_VOICE = "(Warm, articulate academic professor, clear and measured pace)"
_voice_anchors: dict[str, str] = {}
_voice_anchor_lock = threading.Lock()


def reset_tts_voice(voice_key: str | None) -> None:
    """Forget the generated reference voice for one lecture session."""
    if not voice_key:
        return
    with _voice_anchor_lock:
        _voice_anchors.pop(voice_key, None)


def _speech_request(text: str, voice_key: str | None = None):
    tts_cfg: ModelConfig = CONFIG.tts
    client = __import__("openai").OpenAI(
        base_url=tts_cfg.openai_base_url,
        api_key=tts_cfg.api_key,
    )
    extra_body = {}
    if voice_key:
        with _voice_anchor_lock:
            ref_audio = _voice_anchors.get(voice_key)
        if ref_audio:
            extra_body["ref_audio"] = ref_audio
    request = {
        "model": tts_cfg.model,
        "voice": CONFIG.tts_voice,
        "input": f"{_PROF_VOICE}{text}" if not extra_body else text,
        "response_format": "wav",
    }
    if extra_body:
        request["extra_body"] = extra_body
    return client.audio.speech.create(
        **request,
    )


def _remember_voice(voice_key: str | None, wav_bytes: bytes) -> None:
    """Use the first utterance as VoxCPM reference audio for later beats."""
    if not voice_key:
        return
    try:
        source = io.BytesIO(wav_bytes)
        clipped = io.BytesIO()
        with wave.open(source, "rb") as reader:
            params = reader.getparams()
            frames = reader.readframes(min(reader.getnframes(), reader.getframerate() * 8))
        with wave.open(clipped, "wb") as writer:
            writer.setparams(params)
            writer.writeframes(frames)
        wav_bytes = clipped.getvalue()
    except wave.Error:
        pass
    with _voice_anchor_lock:
        if voice_key in _voice_anchors:
            return
        _voice_anchors[voice_key] = (
            "data:audio/wav;base64," + base64.b64encode(wav_bytes).decode("ascii")
        )


def _tts_speak_stream(text: str, sample_rate: int = 48_000) -> Generator[np.ndarray, None, None]:
    """Yield a PCM chunk (float32, mono) for *text* via /v1/audio/speech.

    VoxCPM2 returns a 48 kHz WAV file; we decode it and yield one chunk.
    Falls back to a silent chunk when TTS_BASE_URL is unset.
    """
    tts_cfg: ModelConfig = CONFIG.tts

    if not tts_cfg.is_live:
        yield np.zeros(sample_rate // 2, dtype=np.float32)
        return

    try:
        response = _speech_request(text)
        buf = io.BytesIO(response.content)
        with wave.open(buf, "rb") as wf:
            raw = wf.readframes(wf.getnframes())
        pcm16 = np.frombuffer(raw, dtype=np.int16)
        yield pcm16.astype(np.float32) / 32768.0
    except Exception as exc:  # pragma: no cover
        print(f"[rtc] TTS error: {exc}")
        yield np.zeros(sample_rate // 2, dtype=np.float32)


def tts_speak_full(
    text: str,
    *,
    voice_key: str | None = None,
) -> tuple[int, np.ndarray] | None:
    """Call TTS and return (sample_rate, pcm_float32) for gr.Audio, or None in mock mode.

    This is used by the agent loop (on_teach_deck / on_explain) to speak each
    slide's explanation.  No fastrtc needed — pure HTTP to the Modal endpoint.
    Returns None when TTS_BASE_URL is unset so callers can skip gracefully.
    """
    tts_cfg: ModelConfig = CONFIG.tts
    if not tts_cfg.is_live:
        return None
    try:
        response = _speech_request(text, voice_key)
        wav_bytes = response.content
        _remember_voice(voice_key, wav_bytes)
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return sr, pcm
    except Exception as exc:
        print(f"[rtc] TTS error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Public: build the ReplyOnPause handler
# ---------------------------------------------------------------------------

def build_rtc_handler(
    state_getter: Callable[[], dict],
    sample_rate: int = 16_000,
) -> "ReplyOnPause":
    """Return a ``fastrtc.ReplyOnPause`` handler wired to STT → brain → TTS.

    Parameters
    ----------
    state_getter:
        Zero-argument callable that returns the current Gradio session state
        dict (same dict used by app.py — must contain ``deck``, ``index``,
        ``readings`` keys).
    sample_rate:
        Sample rate (Hz) of the audio chunks delivered by fastrtc (default
        16 000 Hz, which matches Silero VAD's expectation).

    Usage in app.py
    ---------------
    ::

        import gradio as gr
        try:
            from ai_prof.rtc import build_rtc_handler
            _rtc_available = True
        except RuntimeError:
            _rtc_available = False

        if _rtc_available:
            _rtc_state_ref = {}  # populated in on_upload / on_explain

            handler = build_rtc_handler(state_getter=lambda: _rtc_state_ref["state"])

            with gr.Column(scale=2):
                # ... existing chat column ...
                webrtc = gr.WebRTC(
                    label="Voice interjection (hold to speak)",
                    rtc_configuration=handler.rtc_configuration,
                    mode="send-receive",
                )
                webrtc.stream(
                    handler,
                    inputs=[webrtc, state],
                    outputs=[webrtc],
                    time_limit=120,
                )
        else:
            # TODO: wire fastrtc when installed — see ai_prof/rtc.py
            pass
    """
    if not _FASTRTC_AVAILABLE:
        raise RuntimeError(
            "fastrtc is not installed. "
            "Install it with: pip install 'fastrtc[vad]'"
        )

    def _voice_reply(
        audio: tuple[int, np.ndarray],
        gradio_state: dict | None = None,
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """Called by ReplyOnPause once the student stops speaking.

        Receives the buffered audio segment since the last pause, runs the
        full STT → brain → TTS pipeline, and yields PCM chunks back to the
        student's speaker.

        Parameters
        ----------
        audio:
            (sample_rate, pcm_array) tuple delivered by fastrtc.
        gradio_state:
            The Gradio session state dict (passed as a gr.State input).
        """
        in_sr, pcm = audio
        pcm = pcm.astype(np.float32)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)  # stereo → mono
        # Resample if fastrtc delivers a different rate than we asked for.
        if in_sr != sample_rate and in_sr > 0:
            factor = sample_rate / in_sr
            new_len = int(len(pcm) * factor)
            pcm = np.interp(
                np.linspace(0, len(pcm) - 1, new_len),
                np.arange(len(pcm)),
                pcm,
            ).astype(np.float32)

        # 1. Speech-to-text
        question = _stt_transcribe_live(pcm, sample_rate=sample_rate)
        if not question:
            return

        # 2. Brain: stream the text answer
        state = gradio_state or state_getter()
        deck = state.get("deck")
        if deck is None:
            # No deck loaded yet — skip answering.
            return
        idx = state.get("index", 0)
        reading = state.get("readings", {}).get(idx, "")

        answer_chunks: list[str] = []
        for tok in answer_question(question, reading=reading, slide_no=idx + 1, history=[]):
            answer_chunks.append(tok)
        answer_text = "".join(answer_chunks)

        if not answer_text.strip():
            return

        # 3. TTS: stream PCM back to the caller
        tts_sr = 48_000  # VoxCPM2 outputs 48 kHz WAV
        for pcm_chunk in _tts_speak_stream(answer_text, sample_rate=tts_sr):
            yield tts_sr, pcm_chunk

    return ReplyOnPause(
        _voice_reply,
        vad_parameters={"threshold": 0.5},
    )
