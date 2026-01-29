import mlflow
import pandas as pd
from .preprocessing import preprocess

MODEL_NAME = "telco_churn_model_xgb"
MODEL_ALIAS= "production"


def load_model():
    print("Loading model.....")
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.xgboost.load_model(model_uri)
    return model
 
# load_model()

def preprocess_input(input_dict:dict):

    print(input_dict)
    data_df = pd.DataFrame.from_dict(input_dict, orient="columns")
    df_new = pd.DataFrame([data_df.set_index(0)[1].to_dict()])

    print(df_new.head())
    data_df = preprocess(run_id=None,X=df_new,y=None,train_flg=0)
    return data_df

def predict_proba(data_df,model):
    proba = model.predict_proba(data_df)
    print("Scored data ......",proba)
    return proba[:,1][0]