from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from ..utils import Customer,Response
from typing import List
from ..serving import load_model,preprocess_input,predict_proba


app = FastAPI(title="Telco Churn Project")


class PredictRequest(BaseModel):
    data :List[Customer]

class PredictResponse(BaseModel):
    data :List[Response] 


@app.post("/predict",response_model=PredictResponse)
def predict(req: PredictRequest):
    print("Inside predict..............................................................")
    print("-------DATA RECEIVED ----------------------------------------")
    threshold = 0.5
    res = []
    for reqdata in req.data:
        model = load_model()
        data_df = preprocess_input(reqdata)
        print("Scoring data....." )
        proba = predict_proba(data_df,model)
        print("Proba :",proba)
        result = Response(
        churn_probability = proba,
        churn = (proba>= threshold) ,
        threshold = threshold
        )
        res.append(result) 
        print("Scoring completed...." )
        print(result)
    print(PredictResponse(data = res))
    return PredictResponse(data = res)

    

@app.get("/health")
def health():
    print("Inside HEALTH..............................................................")

    return {"status": "OK"}