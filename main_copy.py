from joblib import load
from typing import Any, Hashable
import hashlib
import pandas as pd
import traceback
from fastapi import FastAPI,Form, File, UploadFile
import numpy as np


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
y_pred_rf_df['client'] = np.where(y_pred_rf_df.prediction == 1, "non solvable", "solvable")
y_pred_rf_df['decision'] = np.where(y_pred_rf_df.prediction == 1, "refuser", "accorder")

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


