"""
Main file of the API
"""

import json
import pickle as pk
from typing import Any, Hashable
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import requests
from fastapi import FastAPI
import shap
# data
df = pd.read_csv("app_train_sample_clean.csv")


###################################
# TODO : load shap values object
###################################


###################################
# TODO : load model ML
###################################


def manual_fillna(value):
    """Remplace les valeurs manquantes (NaN) dans des données."""

    # Vérifie si la valeur est une liste est qu'elle n'est pas vide
    if isinstance(value, list) and len(value) > 0:

        # Récupère le premier élément de la liste
        first = value[0]
        # vérifie si le premier élément de la liste est de type int, float ou str
        if isinstance(first, (int, float, str)):

            # si c'est un float, arrondit chaque élément de la liste à 4 décimal
            if isinstance(first, float):
                return [round(v, 4) for v in value]
            
            # si c'est pas un float renvoie la liste telle quelle
            return value
    # si la valeur n'est pas une liste valide, renvoie un message indiquant que la valeur est manquante
    return f"NAN : {str(type(value))}"


def convert_dictionary(original_dict: dict[Hashable, Any]) -> dict[str, Any]:
    """fonction permettant de convertir un dictionnaire imbriqué -> dictionnaire simple

    :param  original_dict: dictionnaire d'entrée de la fonction

    returns dict[str, Any]
    """
    dict_entry = {}  # Crée un nouveau dictionnaire vide pour stocker les entrées converties.

    for attr, val in original_dict.items():
        val_temp = [v for v in val.values()]  # Crée une liste des valeurs associées à l'attribut.
        dict_entry[attr] = val_temp  # Stocke l'attribut et sa liste de valeurs dans le nouveau dictionnaire.

    return dict_entry  # Renvoie le nouveau dictionnaire créé.





def convert_dict_to_list(input_dict: dict[Hashable, Any]) -> list[tuple[str, Any]]:
    """fonction permettant de convertir un dictionnaire de type dict[hashable,any] -> list[tuple[str, Any]]

    :param  original_dict: dictionnaire d'entrée de la fonction

    returns list[tuple[str, Any]]
    """

    result_list = [(str(key), value) for key, value in input_dict.items()]

    return result_list


# app
app = FastAPI()


@app.get("/")
def root():
    """Home page"""

    return {"Hello": "World"}


@app.get("/get_list_ids")
def get_list_ids():
    """Return list of ids"""

    return {"list_ids": df["ID_CLIENT"].tolist()}


@app.get("/get_population_summary/")
def get_population_summary():
    """Return population summary"""

    # select_columns = ["AGE", "REVENU_TOTAL", "CNT_FAM_MEMBERS"]

    select_data = df.iloc[:, :20].describe().round(2).to_dict()
    list_population = convert_dictionary(select_data)

    return {"population_summary": list_population}


@app.get("/get_client_info/{client_id}")
def get_client_info(client_id: int):
    """Return data dict for a client"""

    select_row = df.loc[df["ID_CLIENT"] == client_id].to_dict()
    client_info = convert_dictionary(select_row)

    client_info = {k: manual_fillna(v) for k, v in client_info.items()}

    return {"client_info": client_info}


@app.get("/get_prediction/{client_id}")
def get_prediction(client_id):
    """Return prediction for a client"""

    # load the df
    # find the client with his id
    select_row = df.loc[df["ID_CLIENT"] == client_id]
    # transform if needed the vector client
    # perform the .predict of this client
    # return the prediction
    return {"client_predit": prediction}


@app.get("/get_shap/{client_id}")
def get_shap(client_id):
    """Return shap values for a client"""

    # load the df
    select_row = df.loc[df["ID_CLIENT"] == client_id].to_dict()
    # find the client with his id
    client_info = convert_dictionary(select_row)
    # transform if needed the vector client
    # perform the shap values computation of this client
    
    ###################
    # TODO : code this
    ###################

    client_shap = {
        "AGE": 0.001,
        "SALARY": 0.122,
        "FLAG_OWN_CAR": -0.34,
    }

    return {"client_shap": client_shap}
