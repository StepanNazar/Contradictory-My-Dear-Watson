from __future__ import annotations

from fastapi import FastAPI, HTTPException, status

from app.backend.schemas import HealthResponse, PredictionRequest, PredictionResponse
from app.backend.service import PredictionService

app = FastAPI(title="Contradictory My Dear Watson API", version="0.1.0")
service = PredictionService()


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
