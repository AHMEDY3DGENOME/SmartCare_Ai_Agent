from fastapi import APIRouter
from pydantic import BaseModel

from ai.agent.llm_agent import generate_llm_chat_response
from backend.api.routes.dashboard import build_dashboard_payload

router = APIRouter()


class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
def chat(request: ChatRequest):

    dashboard_payload = build_dashboard_payload()

    response = generate_llm_chat_response(
        question=request.question,
        dashboard_payload=dashboard_payload
    )

    return response