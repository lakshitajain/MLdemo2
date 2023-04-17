import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn import metrics
# Load the dataset
d=pd.read_csv('data/cardio_train.csv',delimiter=';')

feature_cols1 = ['age', 'gender',"height", 'weight','ap_lo','ap_hi','cholesterol',"gluc","smoke","active"]
logistic_regression = LogisticRegression()

d["weight"]=d["weight"]/74.205690
d["age"]=d["age"]/29
d["height"]=d["height"]/164.359229

Q1 = d["ap_lo"].quantile(0.25)
Q3 = d["ap_lo"].quantile(0.75)
IQR = Q3 - Q1

d["ap_lo"] = d["ap_lo"][~((d["ap_lo"] < (Q1 - 1.5 * IQR)) |(d["ap_lo"] > (Q3 + 1.5 * IQR)))]

d["ap_lo"]=d["ap_lo"].fillna(d["ap_lo"].mean())
d["ap_hi"]=d["ap_hi"].fillna(d["ap_hi"].mean())
X = d[feature_cols1] # Features
y = d.cardio 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=4)
artifact_path = "data"
import mlflow
from sklearn.metrics import mean_squared_error, r2_score
with mlflow.start_run():
    data_version = len(list(mlflow.search_runs(filter_string="tags.mlflow.source.name='dataset.py'"))) + 1
    #data_version = mlflow.search_runs(filter_string="tags.mlflow.source.name='dataset.py'").count() + 1
    artifact_path = f"data_version_{data_version}"
    np.savetxt(f"{artifact_path.replace('/', '-')}.csv", np.concatenate((X_train, np.array(y_train).reshape(-1, 1)), axis=1), delimiter=",")
    mlflow.log_artifact(f"{artifact_path}.csv")
    #logistic_regression
    logistic_regression= LogisticRegression()
    logistic_regression.fit(X_train,y_train)
    y_pred=logistic_regression.predict(X_test)
    mlflow.log_metric('Accuracy',metrics.accuracy_score(y_test, y_pred))
    mlflow.log_metric('MSE',mean_squared_error(y_test, y_pred))
    mlflow.log_metric('r2',r2_score(y_test, y_pred))
    # Add the dataset artifact as a run tag
    mlflow.set_tag("dataset_version", data_version)
    mlflow.sklearn.log_model(logistic_regression, "model", registered_model_name="logistic_regression-model-8")
mlflow.end_run()