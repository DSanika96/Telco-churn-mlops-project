import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import joblib
from  data_ingestion import load_data   
from preprocessing import preprocess
from xgboost import XGBClassifier


def train_model(run_id,X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_id=run_id, nested=True) as run:
        print("Model training...")
        C = 1.0
        solver = "lbfgs"
        max_iter = 500

        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)
        mlflow.log_param("max_iter", max_iter)

        model = LogisticRegression(C=C, solver=solver, max_iter=max_iter)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)

        mlflow.log_metric("val_accuracy", float(acc))
        mlflow.log_metric("val_auc", float(auc)) 

        mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="telco_churn_model"
        )

        local_model_path = "model_joblib.pkl"
        joblib.dump(model, local_model_path)
        mlflow.log_artifact(local_model_path)

        print("Run id:", run.info.run_id)
        print("Logged metrics: acc=%.4f auc=%.4f" % (acc, auc))



def train_model_XGB(run_id,X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_id=run_id, nested=True) as run:
        print("Model training...")
 
        objective="binary:logistic"
        n_estimators=200
        max_depth=4
        learning_rate=0.05
        subsample=0.8
        colsample_bytree=0.8
        eval_metric="logloss"
        random_state=42

        mlflow.log_param("objective", "binary:logistic")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("learning_rate", 0.05)
        mlflow.log_param("subsample", 0.8)
        mlflow.log_param("colsample_bytree", 0.8)
        mlflow.log_param("eval_metric", "logloss")

        model = XGBClassifier(  objective="binary:logistic",
                                n_estimators=200,
                                max_depth=4,
                                learning_rate=0.05,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                eval_metric="logloss",
                                random_state=42
                                )
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        probs = model.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, preds)
        auc = roc_auc_score(y_val, probs)

        mlflow.log_metric("val_accuracy", float(acc))
        mlflow.log_metric("val_auc", float(auc)) 

        mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        registered_model_name="telco_churn_model_xgb"
        )

        local_model_path = "model_joblib.pkl"
        joblib.dump(model, local_model_path)
        mlflow.log_artifact(local_model_path)

        print("Run id:", run.info.run_id)
        print("Logged metrics: acc=%.4f auc=%.4f" % (acc, auc))



def main(data_path, mlflow_tracking_uri, experiment_name, register_model):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Starting run with ID: {run_id}")

        X,y = load_data(run_id)
        X_train, X_val, y_train, y_val  = preprocess(run_id,X,y,1)
        print(X_train)
        train_model(run_id,X_train, X_val, y_train, y_val)
        train_model_XGB(run_id,X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--mlflow_uri", default="./mlruns")
    parser.add_argument("--experiment_name", default="telco_churn_experiment")
    parser.add_argument("--register", action="store_true")
    args = parser.parse_args()

    main(args.data_path, args.mlflow_uri, args.experiment_name, args.register)
