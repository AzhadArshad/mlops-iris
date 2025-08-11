from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import numpy as np

class PredictRequest(BaseModel):
    model_name: str  # "LogisticRegression" or "RandomForest"
    features: list[float]

app = FastAPI(title="Iris Model API", version="1.0")

def load_model(model_name: str):
    mlflow.set_tracking_uri("http://localhost:5000")
    model_uri = f"models:/{model_name}/Production"
    return mlflow.sklearn.load_model(model_uri)

@app.get("/")
def root():
    return {"message": "Iris Model API is running"}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        model = load_model(request.model_name)
        data = np.array(request.features).reshape(1, -1)
        prediction = model.predict(data)
        return {
            "model": request.model_name,
            "prediction": int(prediction[0])
        }
    except Exception as e:
        return {"error": str(e)}