# Breast Cancer Predictor API

This microservice trains multiple classification models to predict breast cancer diagnoses, logs them using MLflow, and exposes the best one as a REST API via FastAPI.

- **MLflow** for model tracking and artifact logging
- **FastAPI** for serving predictions as a REST API


# Dataset 

The dataset comes built-in with scikit-learn, and it’s officially called the Breast Cancer Wisconsin Diagnostic Dataset.


# Binary classification.

We want to predict whether a tumor is benign (non-cancerous) or malignant (cancerous) based on some measurements from a digitized image of a fine needle aspirate (FNA) of a breast mass.


## Target Variable
target: 0 means malignant, 1 means benign


# Features (input variables)

There are 30 numerical features, like:

mean radius: average size of the tumor

mean texture: variation in texture

mean perimeter, mean area

mean smoothness: smoothness of cell edges

Each is a measurement calculated over the shape or texture of the tumor in an image.


# Create Environment & Install Requirements
``` bash
conda create -n mlflow-demo python=3.10
conda activate mlflow-demo
pip install -r requirements.txt
```

# Train the Model
This will train the model, Log it to mlruns/, print the local model path.
``` bash
python train.py
```

# Serve with FastAPI
``` bash
uvicorn app:app --reload
```
