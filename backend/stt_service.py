import os
import tempfile
from typing import Optional

_whisper_model = None


def get_whisper_model():
    """
    Lazily loads WhisperModel to ensure clean, fast startup.
    Model: base.en, CPU, int8 quantization.
    """
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # Using base.en with int8 for fast CPU inference
        _whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_audio_file(file_bytes: bytes, filename_hint: str = "audio.webm") -> str:
    """
    Transcribes audio bytes (WebM, WAV, MP3) using faster-whisper.
    Returns cleaned transcription text.
    """
    if not file_bytes or len(file_bytes) < 100:
        return ""

    suffix = os.path.splitext(filename_hint)[-1] or ".webm"
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(file_bytes)
            temp_path = f.name
            temp_file = temp_path

        model = get_whisper_model()
        segments, info = model.transcribe(temp_path, beam_size=5, language="en")
        
        transcribed_segments = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                transcribed_segments.append(text)
        
        full_transcript = " ".join(transcribed_segments).strip()
        return full_transcript
    except Exception as e:
        print(f"[stt_service] Transcription error: {e}")
        return ""
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass
