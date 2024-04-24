from joblib import load
from typing import Any, Hashable
import hashlib
import pandas as pd
import traceback
from fastapi import FastAPI,Form, File, UploadFile
import numpy as np
import shap
import pickle as pk

# Créer une instance de l'application FastAPI
app = FastAPI()


########################
# Lecture des fichiers #
########################


def lecture_x_test_original ():
    x_test_original = pd.read_csv("app_train_sample.csv")
    x_test_original = x_test_original.rename(columns=str.lower)
    return x_test_original


def lecture_x_test_original_clean():
    x_test_clean = pd.read_csv("app_train_sample_clean.csv")
    return x_test_clean

# Charger le dictionnaire des valeurs SHAP
shap_dict = load("shap_dict.joblib")


#################################################
# Lecture du modèle de prédiction et des scores #
#################################################
model_rf = load("model_rf.joblib")


y_pred_rf = model_rf.predict(lecture_x_test_original_clean().drop(labels="ID_CLIENT", axis=1))    # Prédiction de la classe 0 ou 1
y_pred_rf_proba = model_rf.predict_proba(lecture_x_test_original_clean().drop(labels="ID_CLIENT", axis=1)) # Prédiction du % de risque

# Récupération du score du client
y_pred_proba_df = pd.DataFrame(y_pred_rf_proba, columns=['proba_classe_0', 'proba_classe_1'])
y_pred_proba_df = pd.concat([y_pred_proba_df['proba_classe_1'],lecture_x_test_original_clean()['ID_CLIENT']], axis=1)

# Récupération de la décision
y_pred_rf_df = pd.DataFrame(y_pred_rf, columns=['prediction'])
y_pred_rf_df = pd.concat([y_pred_rf_df, lecture_x_test_original_clean()['ID_CLIENT']], axis=1)
y_pred_rf_df['client'] = np.where(y_pred_rf_df.prediction == 1, "Le client n'est pas solvable 💸🚫  ", "Le client est solvable 💰🥳 ")
y_pred_rf_df['decision'] = np.where(y_pred_rf_df.prediction == 1, "CRÉDIT NON ACCORDÉ 🚫", "CRÉDIT ACCORDÉ 🥳")

# app
app = FastAPI()

@app.get("/predic_client/{id_client}")
def predict(id_client: int):
    all_id_client = list(lecture_x_test_original_clean()['ID_CLIENT'])
    
    ID = id_client
    ID = int(ID)
    if ID not in all_id_client:
        number="L'identifiant que vous avez saisi n'est pas valide !"
        prediction="NA"
        solvabilite="NA"
        decision="NA"
    else :
        number="Identifiant client trouvé"
        score = y_pred_proba_df[y_pred_proba_df['ID_CLIENT']==ID]
        prediction = round(score.proba_classe_1.iloc[0]*100, 1)
        solvabilite = y_pred_rf_df.loc[y_pred_rf_df['ID_CLIENT']==ID, "client"].values
        solvabilite = solvabilite[0]
        decision = y_pred_rf_df.loc[y_pred_rf_df['ID_CLIENT']==ID, "decision"].values
        decision = decision[0]
    liste = [{"number": number, "prediction": prediction, "solvabilite": solvabilite, "decision": decision}]    
    return liste


# Définir la route pour récupérer les informations du client en fonction de son ID
@app.get("/client_info/{client_id}")
def get_client_info(client_id: int):
    # Recherche des informations du client en fonction de son ID dans les données nettoyées
    client_info = lecture_x_test_original_clean()[lecture_x_test_original_clean()['ID_CLIENT'] == client_id].to_dict(orient='records')
    
    # Si le client est trouvé, renvoyer ses informations
    if client_info:
        return client_info[0]
    # Sinon, renvoyer un message d'erreur
    else:
        return {"error": "Client not found"}



@app.get("/shap_values/{client_id}")
def get_shap_values(client_id: int):  
    # Débogage : afficher l'ID_CLIENT reçu dans la requête
    print(f"ID_CLIENT reçu : {client_id}")
    
    # Vérifier si l'ID_CLIENT existe dans le dictionnaire shap_dict
    if client_id in shap_dict:
        # Débogage : afficher un message si l'ID_CLIENT est trouvé
        print("ID_CLIENT trouvé dans le dictionnaire shap_dict")
        
        # Récupérer les valeurs SHAP associées à cet ID_CLIENT
        shap_values_for_client = shap_dict[client_id]
        
        # Lecture des données nettoyées
        x_test_clean = lecture_x_test_original_clean()
        
        # Récupérer les informations du client en fonction de son ID
        client_info = x_test_clean[x_test_clean['ID_CLIENT'] == client_id].to_dict(orient='records')
        
        # Retourner les valeurs SHAP et les informations du client
        return {"shap_values": shap_values_for_client, "client_info": client_info}
    else:
        # Débogage : afficher un message si l'ID_CLIENT n'est pas trouvé
        print("ID_CLIENT non trouvé dans le dictionnaire shap_dict")
        
        # Retourner un message d'erreur
        return {"error": "ID_CLIENT not found"}
