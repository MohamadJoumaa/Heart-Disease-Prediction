from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Heart Disease Prediction API")

# Load models and preprocessor
try:
    models = {
        'Decision Tree': joblib.load('dt_model.pkl'),
        'Random Forest': joblib.load('rf_model.pkl'),
        'KNN': joblib.load('knn_model.pkl')
    }
except Exception as e:
    print(f"Warning: Models not found. Ensure notebook has been executed. Error: {e}")
    models = {}

class PatientData(BaseModel):
    Age: int
    Sex: str
    ChestPainType: str
    RestingBP: int
    Cholesterol: int
    FastingBS: int
    RestingECG: str
    MaxHR: int
    ExerciseAngina: str
    Oldpeak: float
    ST_Slope: str
    ModelType: str = 'Random Forest'

@app.get("/")
def home():
    return {"message": "Welcome to the Heart Disease Prediction API. Use /predict endpoint."}

@app.post("/predict")
def predict(data: PatientData):
    if not models:
        raise HTTPException(status_code=500, detail="Models not loaded properly on the server.")
    
    if data.ModelType not in models:
        raise HTTPException(status_code=400, detail=f"ModelType must be one of {list(models.keys())}")
    
    # Create DataFrame from input data
    input_dict = data.dict(exclude={'ModelType'})
    input_df = pd.DataFrame([input_dict])
    
    model = models[data.ModelType]
    
    try:
        prediction = model.predict(input_df)[0]
        prediction_prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
        
        return {
            "prediction": int(prediction),
            "probability": float(prediction_prob) if prediction_prob is not None else None,
            "model_used": data.ModelType,
            "message": "Heart Disease" if prediction == 1 else "Normal"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
