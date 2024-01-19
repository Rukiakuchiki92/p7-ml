"""
Main file of the API
"""

import json
import pickle as pk

from fastapi import FastAPI

import pandas as pd

# from pydantic import BaseModel


# app
app = FastAPI()

# data
df = pd.read_csv("app_train_sample_clean.csv")
df.to_pickle("hello_world.pkl")
# TODO : load shap values


# TODO : delete this
# class item(BaseModel):
#     ID_CLIENT: int


@app.get("/")
def root():
    """Home page"""

    return {"Hello": "World"}


@app.get("/list_ids")
def get_list_ids():
    """Return list of ids"""

    # load the df
    # find the list of ids
    # return the list of ids

    list_ids = [1, 2, 3, 4, 5]

    return {"list_ids": list_ids}


@app.get("/get_client_info/{client_id}")
def get_client_info(client_id):
    """Return data dict for a client"""

    # load the df
    # find the client with his id
    # build dict with all variables
    # return the dict

    client_info = {
        "ID_CLIENT": 1,
        "AGE": 30,
        "SALARY": 1000,
        # etc etc.
    }

    return {"client_info": client_info}


@app.get("/get_prediction/{client_id}")
def get_prediction(client_id):
    """Return prediction for a client"""

    # load the df
    # find the client with his id
    # transform if needed the vector client
    # perform the .predict of this client
    # return the prediction

    client_predit = {
        "0": 0.55,
        "1": 0.45,
    }

    return {"client_predit": client_predit}


@app.get("/get_population_info/{client_id}")
def get_shap(client_id):
    """Return shap values for a client"""

    # load the df
    # find the client with his id
    # transform if needed the vector client
    # perform the shap values computation of this client
    # return the values

    client_shap = {
        "AGE": 0.001,
        "SALARY": 0.122,
        "FLAG_OWN_CAR": -0.34,
    }

    return {"client_shap": client_shap}


@app.get("/get_population_summary/")
def get_population_summary():
    """Return population summary"""

    # load the df
    # build the describe of the df
    # round 2 values if needed
    # return the describe

    population_summary = {
        "AGE": {"mean": 30, "std": 5},
        "SALARY": {"mean": 1000, "std": 200},
        "FLAG_OWN_CAR": {"mean": 0.5, "std": 0.5},
    }
    return {"population_summary": population_summary}
