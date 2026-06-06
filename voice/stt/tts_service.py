import os
import uuid
from gtts import gTTS

AUDIO_DIR = "generated_audio"


def detect_language(text: str) -> str:
    if not text:
        return "en"

    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    english_chars = sum(1 for c in text if c.isascii() and c.isalpha())

    return "ar" if arabic_chars > english_chars else "en"


def normalize_language(language: str | None, text: str) -> str:
    if language:
        language = language.lower().strip()

        if language.startswith("ar"):
            return "ar"

        if language.startswith("en"):
            return "en"

    return detect_language(text)


def generate_tts_audio(text: str, language: str | None = None) -> str:
    if not text or not text.strip():
        raise ValueError("Text is empty")

    os.makedirs(AUDIO_DIR, exist_ok=True)

    lang = normalize_language(language, text)

    filename = f"tts_{uuid.uuid4().hex}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    tts = gTTS(
        text=text,
        lang=lang,
        slow=False
    )

    tts.save(filepath)

    return filepath