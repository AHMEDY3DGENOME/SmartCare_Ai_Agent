from uuid import uuid4
from typing import Optional, Dict, Any

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel, Field

from ai.agent.langgraph_agent import generate_langgraph_chat_response
from backend.api.routes.dashboard import build_dashboard_payload


router = APIRouter()


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    agent: str
    provider: str
    model: Optional[str] = None
    question: str
    intent: str
    answer: str
    context: Dict[str, Any]
    agent_trace: Dict[str, Any]
    disclaimer: str


@router.post("/chat", response_model=ChatResponse)
def chat(request_data: ChatRequest, request: Request, response: Response):
    session_id = (
        request_data.session_id
        or request.cookies.get("caresense_session_id")
        or str(uuid4())
    )

    response.set_cookie(
        key="caresense_session_id",
        value=session_id,
        httponly=False,
        samesite="lax",
    )

    dashboard_payload = build_dashboard_payload()

    result = generate_langgraph_chat_response(
        question=request_data.question,
        dashboard_payload=dashboard_payload,
        session_id=session_id,
    )

    result["session_id"] = session_id

    return result