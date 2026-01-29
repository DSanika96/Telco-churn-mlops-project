from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Customer(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService : str
    MultipleLines : str
    InternetService: str
    OnlineSecurity : str
    OnlineBackup : str
    DeviceProtection : str
    TechSupport : str
    StreamingTV : str
    StreamingMovies : str
    Contract : str
    PaperlessBilling : str
    PaymentMethod : str
    MonthlyCharges : float 
    TotalCharges : float

class Response(BaseModel):
    churn_probability: float
    churn:bool
    threshold:float