from fastapi import FastAPI, Request
from fastapi.responses import Response
import joblib
import pandas as pd
import time
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from app.schemas import PredictRequest, PredictResponse
from app.logging_config import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Heart Disease Risk Prediction API", version="1.0.0")

# Resolve model path robustly
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "production", "model.pkl")
model = joblib.load(model_path)

REQUEST_COUNT = Counter("api_requests_total", "Total API requests", ["endpoint", "http_status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])
PREDICTION_COUNT = Counter("predictions_total", "Total predictions made", ["predicted_class"])


@app.middleware("http")
async def log_and_time_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    # Observe and increment metrics
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(endpoint=request.url.path, http_status=response.status_code).inc()

    # Emit JSON logging
    logger.info(
        "request_completed",
        extra={
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": round(duration * 1000, 2),
        },
    )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    # Convert Pydantic request to pandas DataFrame
    df = pd.DataFrame([payload.model_dump()])

    # Predict
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    confidence = float(proba[pred])

    # Update metrics
    PREDICTION_COUNT.labels(predicted_class=str(pred)).inc()

    # Emit structured log
    logger.info("prediction_made", extra={"prediction": int(pred), "confidence": confidence})

    return PredictResponse(
        prediction=int(pred),
        label="disease" if pred == 1 else "no_disease",
        confidence=round(confidence, 4),
    )


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
