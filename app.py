#!/usr/bin/env python3
"""Auto-generated FastAPI prediction API."""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import joblib, pandas as pd

app = FastAPI(title="ML Prediction API")
pipeline = joblib.load("models/final_pipeline.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

class InputData(BaseModel):
    age: Optional[float] = None
    salary: Optional[float] = None
    experience: Optional[float] = None

@app.get("/health")
def health():
    return {"status": "ok", "model": "loaded"}

@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    pred = pipeline.predict(df)[0]
    label = label_encoder.inverse_transform([pred])[0]
    proba = pipeline.predict_proba(df)[0].tolist()
    return {"prediction": str(label), "probabilities": proba}

@app.post("/predict/batch")
def predict_batch(data: List[InputData]):
    df = pd.DataFrame([d.dict() for d in data])
    preds = pipeline.predict(df)
    labels = label_encoder.inverse_transform(preds).tolist()
    return {"predictions": [str(l) for l in labels]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
