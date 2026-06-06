from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

from voice.stt.tts_service import generate_tts_audio

router = APIRouter()


class TTSRequest(BaseModel):
    text: str


@router.post("/tts")
def text_to_speech(request: TTSRequest):

    audio_path = generate_tts_audio(request.text)

    return FileResponse(
        audio_path,
        media_type="audio/mpeg",
        filename="caresense_response.mp3"
    )