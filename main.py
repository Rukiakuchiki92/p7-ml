from fastapi import FastAPI
import pandas as pd
import json
from pydantic import BaseModel
import pickle as p

#s
app = FastAPI()
df = pd.read_csv("app_train_sample_clean.csv")
df.to_pickle("hello_world.pkl")

class item(BaseModel):
    ID_CLIENT: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

def test_predict(df):
    with open('hello_world.pkl', 'rb') as lib:
        corn = p.load(lib)
    return predict

@app.post("/items")
def scoring_endpoint(item:item):
    df = pd.DataFrame([item.model_dump().values()], columns=item.model_dump().keys())
    lo = test_predict(df)
    return {"predicttion":int(lo)}    


#def read_item(item_id: int, q: str = None):



    # file = open("hellow_horld.json")
    # return json.load(file)

