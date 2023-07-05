import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.utils import read_data
from src.preprocessing import preprocess

app = FastAPI(title='Predicting Machine Failure')

# Represents a particular machine (or datapoint)
class Machine(BaseModel):
    ProductID: object
    Type: object
    AirTemperature: float
    ProcessTemperature: float
    RotationalSpeed: int
    Torque: float
    ToolWear: int
    TWF: int
    HDF: int
    PWF: int
    OSF: int
    RNF: int

@app.on_event("startup")
def load_clf():
    # Load classifier from pickle file
    with open("models/xgboost.pickle", "rb") as file:
        global clf
        clf = pickle.load(file)

@app.get("/")
def home():
    return {"name": "Predict Machine Failure",
            "version": "v1.0.0"}

@app.post("/predict")
def predict(machine: Machine):

    # store parameters in an array
    data_point = np.array(
        [
            [
                machine.ProductID,
                machine.Type,
                machine.AirTemperature,
                machine.ProcessTemperature,
                machine.RotationalSpeed,
                machine.Torque,
                machine.ToolWear,
                machine.TWF,
                machine.HDF,
                machine.PWF,
                machine.OSF,
                machine.RNF
            ]
        ]
    )

    df_train = read_data("train_data")
    X = df_train.drop("MachineFailure", axis=1)

    print("Processed...")
    X_train_processed, X_test_processed = preprocess(X, data_point, feature_selection=True)

    print("Predict..")
    pred = clf.predict(X_test_processed).tolist()
    pred = pred[0]
    print(pred)
    return {"Prediction": pred}
