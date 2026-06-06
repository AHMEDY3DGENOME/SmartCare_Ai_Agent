from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.routes.simulator import router as simulator_router
from backend.api.routes.dashboard import router as dashboard_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.tts import router as tts_router

app = FastAPI(
    title="CareSense AI",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Routers
# --------------------------------------------------

app.include_router(simulator_router)
app.include_router(dashboard_router)
app.include_router(chat_router)
app.include_router(tts_router)

# --------------------------------------------------
# Static Files
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="dashboard/frontend"),
    name="static"
)

# --------------------------------------------------
# Basic Endpoints
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "project": "CareSense AI",
        "status": "running",
        "version": "0.1.0",
        "modules": [
            "WiFi CSI Simulator",
            "Risk Engine",
            "Fall Detection",
            "Medical AI Agent",
            "Gemini LLM Agent",
            "TTS Service",
            "Dashboard"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.get("/dashboard")
def dashboard_page():
    return FileResponse(
        "dashboard/frontend/index.html"
    )