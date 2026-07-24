import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Body Fat Predictor API")

# Allow the frontend (served from anywhere, including this same app) to call /predict
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("bodyfat_model.pkl")
scaler = joblib.load("bodyfat_scaler.pkl")
clip_bounds = joblib.load("bodyfat_clip_bounds.pkl")

# Exact column order the model/scaler were fit on
FEATURE_ORDER = [
    "Sex", "Age", "Weight", "Height", "Neck", "Chest", "Abdomen",
    "Hip", "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist",
]


class Measurements(BaseModel):
    sex: str = Field(..., description="'M' or 'F'")
    age: float
    weight: float = Field(..., description="kg")
    height: float = Field(..., description="meters")
    neck: float = Field(..., description="cm")
    chest: float = Field(..., description="cm")
    abdomen: float = Field(..., description="cm")
    hip: float = Field(..., description="cm")
    thigh: float = Field(..., description="cm")
    knee: float = Field(..., description="cm")
    ankle: float = Field(..., description="cm")
    biceps: float = Field(..., description="cm")
    forearm: float = Field(..., description="cm")
    wrist: float = Field(..., description="cm")


class Prediction(BaseModel):
    body_fat_percent: float


@app.post("/predict", response_model=Prediction)
def predict(m: Measurements):
    sex_code = {"M": 0, "F": 1}.get(m.sex.upper())
    if sex_code is None:
        raise HTTPException(status_code=422, detail="sex must be 'M' or 'F'")

    row = {
        "Sex": sex_code, "Age": m.age, "Weight": m.weight, "Height": m.height,
        "Neck": m.neck, "Chest": m.chest, "Abdomen": m.abdomen, "Hip": m.hip,
        "Thigh": m.thigh, "Knee": m.knee, "Ankle": m.ankle, "Biceps": m.biceps,
        "Forearm": m.forearm, "Wrist": m.wrist,
    }
    X = pd.DataFrame([row], columns=FEATURE_ORDER)

    # Same IQR clip bounds fit on the training set — must match training-time preprocessing exactly
    for col, (lower, upper) in clip_bounds.items():
        X[col] = X[col].clip(lower, upper)

    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_ORDER)
    pred = float(model.predict(X_scaled)[0])
    pred = max(0.0, pred)  # a prediction below 0% isn't physically meaningful

    return Prediction(body_fat_percent=round(pred, 1))


# Serve the frontend
app.mount("/", StaticFiles(directory="static", html=True), name="static")
