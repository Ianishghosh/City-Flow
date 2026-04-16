from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from inference import Predictor

# ── App setup ─────────────────────────────────────────────────────────
cfg  = Config()
app  = FastAPI(title=cfg.api_title, version=cfg.api_version)

predictor = Predictor(cfg)

# Load adjacency matrix once at startup
with open(cfg.adj_file, "rb") as f:
    adj_raw = pickle.load(f, encoding='latin1')
    adj_mx = adj_raw[2] if isinstance(adj_raw, (tuple, list)) and len(adj_raw) == 3 else adj_raw  # (207, 207)


# ── Schemas ───────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    speed_window: list   # (12, 207) flattened or nested list


class PredictResponse(BaseModel):
    predictions: list    # (12, 207) — next 60 min speed predictions
    message: str


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "✅ City-Flow API is running!", "version": cfg.api_version}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        speed_window = np.array(req.speed_window, dtype=np.float32)

        if speed_window.shape != (12, 207):
            raise HTTPException(
                status_code=422,
                detail=f"Expected shape (12, 207), got {speed_window.shape}"
            )

        predictions = predictor.predict(speed_window, adj_mx)

        return PredictResponse(
            predictions=predictions.tolist(),
            message="✅ Prediction successful — next 60 minutes forecasted"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Run directly ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg.api_host, port=cfg.api_port)