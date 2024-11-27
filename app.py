import pandas as pd
import pickle
from enum import Enum
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from io import BytesIO
from joblib import load
from pydantic import BaseModel, Field
from typing import List, Optional

'''
utils.py файл в котором определен код ноутбука
к сожалению, uvicorn запускает не __main__, а __mp_main__, 
а файлы из ноутбука естественно находятся в модуле __main__
поэтому было решено обойти это просто с точки зрения количества строк кода и времени.
На практике видимо нужно определять все классы 
в отдельных файлах и импортировать их в ноутбук.
'''

import __mp_main__
import utils

for attr_name in dir(utils):
    if not attr_name.startswith("__"):
        setattr(__mp_main__, attr_name, getattr(utils, attr_name))

pipeline = load('pipeline.pkl')
    
app = FastAPI()

class FuelType(str, Enum):
    diesel = "Diesel"
    petrol = "Petrol"
    cng = "CNG"
    lpg = "LPG"


class SellerType(str, Enum):
    individual = "Individual"
    dealer = "Dealer"
    trustmark_dealer = "Trustmark Dealer"

class TransmissionType(str, Enum):
    manual = "Manual"
    automatic = "Automatic"

class OwnerType(str, Enum):
    first_owner = "First Owner"
    second_owner = "Second Owner"
    third_owner = "Third Owner"
    fourth_and_above_owner = "Fourth & Above Owner"
    test_drive_car = "Test Drive Car"

class Item(BaseModel):
    name: Optional[str] = None
    year: Optional[int] = Field(None, ge=1900, lt=2025, description="Не имеет смысла брать странный год")
    km_driven: Optional[int] = Field(None, ge=0, description="Километры не могут быть меньше нуля")
    fuel: Optional[FuelType] = None
    seller_type: Optional[SellerType] = None
    transmission: Optional[TransmissionType] = None
    owner: Optional[OwnerType] = None
    mileage: Optional[str] = None
    engine: Optional[str] = None
    max_power: Optional[str] = None
    torque: Optional[str] = None
    seats: Optional[float] = Field(None, gt=0, description="Сидений должно быть больше 0")

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([jsonable_encoder(item)])
    prediction = pipeline.predict(df).tolist()[0]

    return prediction
'''
В условии противоречие:
сначала просят файлы,
потом просят дополнить код, где явно используются List[item] и List[float]
По совету из чата решил сделать файлы
'''
@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    df['selling_price'] = pipeline.predict(df).tolist()

    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predicted_data.csv"}
    )
