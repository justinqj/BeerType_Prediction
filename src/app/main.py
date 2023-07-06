from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and encoders
model = joblib.load("../models/stefan_dev.joblib")
le = joblib.load("../models/label_encoder.joblib")
encoder = joblib.load("../models/target_encoder.joblib")

# Replace with your own median ABV
median_abv = 5.0

class Beer(BaseModel):
    brewery_name: str
    review_aroma: float
    review_appearance: float
    review_palate: float
    review_taste: float
    beer_abv: float

@app.post("/beer_style/")
async def predict_beer_style(beer: Beer):
    # Create dataframe for target encoding
    brewery_df = pd.DataFrame([brewery_name], columns=['brewery_name'])
    encoded_brewery_name = encoder.transform(brewery_df)['brewery_name'].values[0]
    
    # Create final input_data
    input_data = [beer.review_aroma, beer.review_appearance, beer.review_palate, beer.review_taste, beer_abv, encoded_brewery_name]
    
    prediction = model.predict([np.array(input_data)])
    
    # Convert the prediction to actual beer_style
    predicted_beer_style = le.inverse_transform([np.argmax(prediction)])

    return {"beer_style": predicted_beer_style[0]}
