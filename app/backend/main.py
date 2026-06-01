from __future__ import annotations

from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from app.backend.schemas import (
    HealthResponse, 
    PredictionRequest, 
    PredictionResponse, 
    ContradictionRequest, 
    ContradictionResponse
)
from app.backend.service import PredictionService

app = FastAPI(title="Contradictory My Dear Watson API", version="0.2.0")
service = PredictionService()

# API Endpoints
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if service.is_ready:
        return HealthResponse(status="ok", model_loaded=True)
    return HealthResponse(status="degraded", model_loaded=False, detail=service.load_error)


@app.get("/model-info")
def model_info() -> dict:
    return service.model_info()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        result = service.predict(
            premise=request.premise,
            hypothesis=request.hypothesis,
            language=request.language,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return PredictionResponse(**result)


@app.post("/detect-contradictions", response_model=ContradictionResponse)
def detect_contradictions(request: ContradictionRequest) -> ContradictionResponse:
    try:
        result = service.detect_contradictions(
            text=request.text,
            language=request.language,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return ContradictionResponse(**result)

# Frontend Serving
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/ui", StaticFiles(directory=str(frontend_path), html=True), name="ui")

    @app.get("/")
    def read_index():
        return FileResponse(frontend_path / "index.html")
else:
    @app.get("/")
    def read_root():
        return {"message": "API is running. UI directory not found."}
