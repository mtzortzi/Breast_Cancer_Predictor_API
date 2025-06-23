"""
    model training 

    @author mtzortzi
"""

import os
import mlflow 
import pandas as pd
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("breast-cancer-model-comparison")

def train_and_log(model, model_name, params):
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        # Log model under a subfolder
        model_uri = mlflow.sklearn.log_model(model, artifact_path="model")

        print(f'Model logged at:')
        print(f'{model_uri}')
        print("Logged model to:", mlflow.get_artifact_uri("model"))

        print(f'Model {model_name} is trained and logged. Accuracy: {acc:.4f}')
        
        print(f'RUN ID: {run.info.run_id}')

    print(f'Working directory:', os.getcwd())


if __name__ == "__main__":

    #### Load Data ####
    data = load_breast_cancer()
    print(f'features - input variables {data.feature_names}')
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    print(f'y_train {y_train.shape}')
    print(f'y_train {y_train}')


    #### Define Models ####
    models = [
            (
                RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
                "RandomForest",
                {"n_estimators":100, "max_depth":5}

            ),
            (
                GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
                "GradientBoosting",
                {"n_estimators":100, "learning_rate":0.1}
            ),
            (
                LogisticRegression(max_iter=1000, solver="liblinear"),
                "LogisticRegression",
                {"max_iter":1000, "solver":"liblinear"}
            )
    ]

    #### Tain and Log All Models
    for model, name, params in models:
        train_and_log(model, name, params)

