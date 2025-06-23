# Breast Cancer Predictor API (MLflow + FastAPI)

This project trains a **Random Forest** model to classify breast cancer tumors as benign or malignant. It uses:

- **MLflow** for model tracking and artifact logging
- **FastAPI** for serving predictions as a REST API


- Create Environment & Install Requirements
  ``` bash
  conda create -n mlflow-demo python=3.10
  conda activate mlflow-demo
  pip install -r requirements.txt
  ```
  
- Train the Model
  This will train the model, log it to mlruns/, and print the local model path

  ``` bash
  python train.py
  ```
  
- Serve with FastAPI
  ``` bash
  uvicorn app:app --reload
  ```
