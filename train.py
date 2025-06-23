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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


mlflow.set_tracking_uri("file:./mlruns")


def train_model():

    data = load_breast_cancer()
    print(f'features - input variables {data.feature_names}')
    # Loads the dataset
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    with mlflow.start_run(nested=False) as run:
       
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)


        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("accuracy", acc)
        model_uri = mlflow.sklearn.log_model(clf, name="model")

        print(f'Model logged at: {model_uri}')
        print("Logged model to:", mlflow.get_artifact_uri("model"))

        print(f'Model trained and logged. Accuracy: {acc:.4f}')
        
        print(f'RUN ID: {run.info.run_id}')

    print(f'Working directory:', os.getcwd())

    print(f'y_train {y_train.shape}')
    print(f'{pd.DataFrame(y_train)}')
    print(f'Y_train {y_train}')



if __name__ == "__main__":
    train_model()
