from faster_whisper import WhisperModel
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# CPU-friendly Whisper model
model = WhisperModel("tiny", device="cpu")

def transcribe_audio(file_path: str) -> str:
    segments, _ = model.transcribe(file_path)
    return " ".join([seg.text for seg in segments])
