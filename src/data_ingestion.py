import pandas as pd # pyright: ignore[reportMissingModuleSource]
import os
import mlflow # type: ignore

def load_data(run_id):
    with mlflow.start_run(run_id=run_id, nested=True):
        df = pd.read_csv("./data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
        if df.shape[0] == 0 :
            print("Row count zero")
            print("data shape ",df.shape)
            return None
        if "Churn" not in df.columns:
            print("Target variable not present")
            return

        mlflow.log_param("data shape",df.shape)
        print("data shape ",df.shape)
        mlflow.log_param("nulls",df.isna().sum())
        print("Missing values ",df.isna().sum().sum())

        # Setting customer id as index
        df.set_index('customerID',inplace=True) 
        X = pd.DataFrame(df.drop("Churn",axis = 1))
        y = pd.DataFrame(df["Churn"])

        return X,y
