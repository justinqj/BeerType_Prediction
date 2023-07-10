from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

# Load model and encoders
model = load("/app/models/stefan_dev.joblib")

# Root
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Health Check
@app.get('/health', status_code=200)
def healthcheck():
    return 'GMM Clustering is all ready to go!'

# Enforce datatypes
def format_features(Appearance: int, Aroma: int, Palate: int, Taste: int, ABV: int, Brewery: str):
    return {
        'review_appearance': [Appearance],
        'review_aroma': [Aroma],
        'review_palate': [Palate],
        'review_taste': [Taste],
        'beer_abv': [ABV],
        'brewery_name': [Brewery]
    }

@app.get("/beer/type")
def predict(review_appearance: str,	review_aroma: int, review_palate: int, review_taste: int, beer_abv: int, brewery_name: str):
    features = format_features(review_appearance, review_aroma, review_palate, review_taste, beer_abv, brewery_name)
    obs = pd.DataFrame(features)
    pred = model.predict(obs)
    return JSONResponse(pred.tolist())