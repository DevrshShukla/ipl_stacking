from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(title="IPL Score Prediction API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = "models/pipeline.pkl"
DATA_PATH = "data/processed_data.csv"

model = None
teams = []
venues = []

@app.on_event("startup")
def load_assets():
    global model, teams, venues
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")
        
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        teams = sorted(list(set(df['batting_team'].unique()) | set(df['bowling_team'].unique())))
        venues = sorted(list(df['venue'].unique()))
    else:
        print(f"Warning: Data not found at {DATA_PATH}")

class MatchInput(BaseModel):
    batting_team: str
    bowling_team: str
    venue: str
    current_score: int = Field(..., ge=0)
    overs_completed: float = Field(..., ge=5.0, le=19.5)
    wickets: int = Field(..., ge=0, le=9)

class PredictionOutput(BaseModel):
    predicted_score: int
    
@app.get("/api/teams")
def get_teams():
    return {"teams": teams}

@app.get("/api/venues")
def get_venues():
    return {"venues": venues}

@app.post("/api/predict", response_model=PredictionOutput)
def predict_score(data: MatchInput):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    # Validate over format (e.g. 5.1, 5.2... 5.5) -> decimal part should be 0, 1, 2, 3, 4, 5
    frac = data.overs_completed - int(data.overs_completed)
    if frac >= 0.6:
        raise HTTPException(status_code=400, detail="Invalid over format. Decimal part should be between 0 and 0.5 (e.g., 5.3 for 5 overs and 3 balls)")
        
    # Standardize over value to what the model expects
    overs_standard = int(data.overs_completed) + frac / 0.6 * (6/10) # wait, data prep used over + ball/6.0
    # Let's say user inputs 5.3 (5 overs 3 balls). int(5.3)=5, frac=0.3. Ball = 3.
    # In data_prep, overs_completed = over + ball / 6.0
    ball = int(round(frac * 10))
    overs_for_model = int(data.overs_completed) + ball / 6.0
    
    input_df = pd.DataFrame([{
        "batting_team": data.batting_team,
        "bowling_team": data.bowling_team,
        "venue": data.venue,
        "current_score": data.current_score,
        "overs_completed": overs_for_model,
        "wickets": data.wickets
    }])
    
    prediction = model.predict(input_df)[0]
    
    # Realistic prediction bounds:
    # A Random Forest might under-predict for out-of-distribution high scores.
    # We enforce a realistic minimum score based on the current situation.
    remaining_overs = 20 - overs_for_model
    if remaining_overs > 0:
        crr = data.current_score / overs_for_model
        # Minimum expected run rate, scaled down by wickets lost
        min_rr = max(crr * 0.6, 6.0) * ((10 - data.wickets) / 10.0)
        min_realistic_score = data.current_score + (remaining_overs * min_rr)
        prediction = max(prediction, min_realistic_score)
    else:
        prediction = max(prediction, data.current_score)
    
    return {"predicted_score": int(round(prediction))}

# Serve static files last
if os.path.exists("public"):
    app.mount("/", StaticFiles(directory="public", html=True), name="public")
