from fastapi import FastAPI
import numpy as np
from pydantic import BaseModel

app = FastAPI()

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    prediction = np.mean(data.features)  # Dummy prediction
    return {"prediction": prediction}
