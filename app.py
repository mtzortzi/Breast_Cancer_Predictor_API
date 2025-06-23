"""
    app.py: REST API for predictions

    Starts an API server that uses the trained model for predictions

    Purpose: Expose the trained model so that users or applications can send new tumor measurements and get back a prediction: benign or malignant

    @author: mtzortzi

"""

import mlflow.sklearn
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow.tracking import MlflowClient

# Define input schema
# This tells FastAPI what kind of input it should expect in JSON (Only 5 features here for simplicity)

class CancerInput(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float


# Load the model 


path = "mlruns/0/models/m-a19a297fa7714971a0218702b02ba5e6/artifacts/"

model = mlflow.sklearn.load_model(path)


app = FastAPI()


@app.get("/")
def read_root():
    return{"message": "Mlflow API is running"}


@app.post("/predict")
def predict(input_data: CancerInput):
    data = [[
        input_data.mean_radius,
        input_data.mean_texture,
        input_data.mean_perimeter,
        input_data.mean_area,
        input_data.mean_smoothness
    ]]

    pred = model.predict(data)
    return {"prediction": int(pred[0])}

