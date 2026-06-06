import os
import uuid
from gtts import gTTS

AUDIO_DIR = "generated_audio"


def detect_language(text: str) -> str:
    return "ar" if any("\u0600" <= c <= "\u06FF" for c in text) else "en"


def generate_tts_audio(text: str) -> str:
    if not text or not text.strip():
        raise ValueError("Text is empty")

    os.makedirs(AUDIO_DIR, exist_ok=True)

    lang = detect_language(text)

    filename = f"tts_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    tts = gTTS(
        text=text,
        lang=lang,
        slow=False
    )

    tts.save(filepath)

    return filepath