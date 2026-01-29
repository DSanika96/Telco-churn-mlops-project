
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
import numpy as np # pyright: ignore[reportMissingImports, reportMissingModuleSource]
import mlflow # pyright: ignore[reportMissingModuleSource,reportMissingImports]
from sklearn.impute import SimpleImputer # pyright: ignore[reportMissingModuleSource]
from sklearn.pipeline import Pipeline # pyright: ignore[reportMissingModuleSource]
from sklearn.model_selection import train_test_split # pyright: ignore[reportMissingModuleSource]
from sklearn.compose import ColumnTransformer # pyright: ignore[reportMissingModuleSource]
import joblib
from sklearn.preprocessing import OneHotEncoder




def fit_transformer(X,numeric_features,categorical_features):

    # Create a preprocessor with imputation steps
    numeric_transformer = Pipeline(steps=[
        # Impute missing numerical values with the mean of the training data
        ('imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_transformer = Pipeline(steps=[
        # Impute missing categorical values with the most frequent value (mode) of the training data
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder',OneHotEncoder(handle_unknown='ignore',sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    preprocessor = preprocessor.fit(X)
    print(X.columns)
    mlflow.sklearn.log_model(
    sk_model=preprocessor,
    artifact_path="transformer",
    registered_model_name="telco_churn_preprocessor"
    )
    print("Transformer model logged")
    return None


def transform_data(run_id,X,numeric_features,categorical_features,train_flg):
    # Missing value imputation
    MODEL_NAME = "telco_churn_preprocessor"
    MODEL_ALIAS= "production"
    print(X.columns)
    if train_flg == 1 :
         
        transformer = mlflow.sklearn.load_model(model_uri=f"runs:/{run_id}/transformer")

    else:
        model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        transformer = mlflow.sklearn.load_model(model_uri)
    transformer.set_output(transform = "pandas")
    X = pd.DataFrame(transformer.transform(X),
                    index=X.index
                     )
    print(X.columns)
    
    #Convertin the variables in a binary numeric variable
    # X['Partner']=X['Partner'].map({'Yes':1, 'No':0})
    # X['Dependents']=X['Dependents'].map({'Yes':1, 'No':0})
    # X['PhoneService']=X['PhoneService'].map({'Yes':1, 'No':0})
    # X['PaperlessBilling']=X['PaperlessBilling'].map({'Yes':1, 'No':0})

    # # convert all the categorical variables into dummy variables
    # X_cat = pd.get_dummies(X[categorical_features ], drop_first=True)
    # print(X_cat.columns)
    # X = X[numeric_features].merge(X_cat, left_index=True, right_index=True, how='inner')
    return X

    # Dropping columns which could introduce multicollinearity
    # X.drop(['OnlineBackup_No internet service','OnlineSecurity_No internet service','TechSupport_No internet service','DeviceProtection_No internet service'
    #                 ,'StreamingTV_No internet service','StreamingMovies_No internet service'],axis=1,inplace=True)        

def preprocess(run_id,X,y,train_flg):
    # with mlflow.start_run(run_id=run_id, nested=True):
        print("Preprocessing data ...")


        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = ['SeniorCitizen','gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                            'PaperlessBilling', 'PaymentMethod']

        for i in numeric_features:
        # Changing the dtype of TotalCharges from object to int
            X[i] = pd.to_numeric(X[i],errors='coerce') 


        if train_flg == 1:
            y['Churn']=y['Churn'].map({'Yes':1, 'No':0})

            fit_transformer(X,numeric_features,categorical_features)
            X_transformed = transform_data(run_id,X,numeric_features,categorical_features,train_flg)
             
            X_train, X_val, y_train, y_val = train_test_split(X_transformed, y, test_size=0.2, random_state=42, stratify=y)
            return X_train, X_val, y_train, y_val 
        else :
            X_transformed = transform_data(run_id,X,numeric_features,categorical_features,train_flg)
            # for i in numeric_features:
            # # Changing the dtype of TotalCharges from object to int
            #     X_transformed[i] = pd.to_numeric(X_transformed[i],errors='coerce')       
            return X_transformed  

           



        