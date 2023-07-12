from fastapi import FastAPI
from starlette.responses import JSONResponse
import tensorflow as tf
import pandas as pd

app = FastAPI()

# Initialize the model variable
model = None

model = tf.keras.models.load_model("/src/models/best.h5")
# model = keras.models.load_model("models/best.h5")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('/health', status_code=200)
def healthcheck():
    return 'Deep Learning Model is ready to go!'

@app.get("/model/architecture")
async def architecture():
    return print(model.summary())

# # Enforce datatypes
# def format_features(Appearance: int, Aroma: int, Palate: int, Taste: int, ABV: int, Brewery: str):
#     return {
#         'review_appearance': [Appearance],
#         'review_aroma': [Aroma],
#         'review_palate': [Palate],
#         'review_taste': [Taste],
#         'beer_abv': [ABV],
#         'brewery_name': [Brewery]
#     }

# @app.get("/beer/type")
# def predict(review_appearance: int, review_aroma: int, review_palate: int, review_taste: int, beer_abv: int, brewery_name: str):
#     features = format_features(review_appearance, review_aroma, review_palate, review_taste, beer_abv, brewery_name)
#     obs = pd.DataFrame(features)
#     global model
#     pred = model.predict(obs)
#     return JSONResponse(pred.tolist())
