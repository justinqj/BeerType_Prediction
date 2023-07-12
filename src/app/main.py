# # from fastapi import FastAPI
# # from starlette.responses import JSONResponse
# # from tensorflow.keras.models import load_model
# # import pandas as pd

# # app = FastAPI()

# # # Initialize the model variable
# # model = None

# # # # Event handler for loading the model before the server starts
# # @app.on_event("startup")
# # def load_model_on_startup():
# #     global model
# #     model = load_model("C:/Users/justi/adsi-at2/adsi-at2/models/best.h5")

# # @app.get("/")
# # def read_root():
# #     return {"Hello": "World"}

# # @app.get('/health', status_code=200)
# # def healthcheck():
# #     return 'GMM Clustering is all ready to go!'

# # # # Enforce datatypes
# # def format_features(Appearance: int, Aroma: int, Palate: int, Taste: int, ABV: int, Brewery: str):
# #     return {
# #         'review_appearance': [Appearance],
# #         'review_aroma': [Aroma],
# #         'review_palate': [Palate],
# #         'review_taste': [Taste],
# #         'beer_abv': [ABV],
# #         'brewery_name': [Brewery]
# #     }

# # @app.get("/beer/type")
# # def predict(review_appearance: int, review_aroma: int, review_palate: int, review_taste: int, beer_abv: int, brewery_name: str):
# #     features = format_features(review_appearance, review_aroma, review_palate, review_taste, beer_abv, brewery_name)
# #     obs = pd.DataFrame(features)
# #     global model
# #     pred = model.predict(obs)
# #     return JSONResponse(pred.tolist())

# from fastapi import FastAPI, Form
# from starlette.responses import HTMLResponse
# from tensorflow.keras.models import load_model
# import pandas as pd
# import joblib
# from sklearn.preprocessing import LabelEncoder
# import streamlit as st
# import requests

# app = FastAPI()

# # Initialize the model variable
# model = None

# # Event handler for loading the model before the server starts
# @app.on_event("startup")
# def load_model_on_startup():
#     global model
#     model = load_model("C:/Users/justi/adsi-at2/adsi-at2/models/best.h5")

# @app.get("/", response_class=HTMLResponse)
# def read_root():
#     return """
#     <form method="post" action="/beer/type">
#     <input type="number" min="1" max="5" step="1" name="review_appearance" placeholder="Appearance">
#     <input type="number" min="1" max="5" step="1" name="review_aroma" placeholder="Aroma">
#     <input type="number" min="1" max="5" step="1" name="review_palate" placeholder="Palate">
#     <input type="number" min="1" max="5" step="1" name="review_taste" placeholder="Taste">
#     <input type="number" min="0" step="0.1" name="beer_abv" placeholder="ABV">
#     <input type="text" name="brewery_name" placeholder="Brewery">
#     <input type="submit">
#     </form>
#     """

# @app.get('/health', status_code=200)
# def healthcheck():
#     return 'GMM Clustering is all ready to go!'

# # Enforce datatypes
# def format_features(Appearance: float, Aroma: float, Palate: float, Taste: float, ABV: float, Brewery: str):
#     return {
#         'review_appearance': [Appearance],
#         'review_aroma': [Aroma],
#         'review_palate': [Palate],
#         'review_taste': [Taste],
#         'beer_abv': [ABV],
#         'brewery_name': [Brewery]
#     }


# # Load the training data
# df = pd.read_csv('C:/Users/justi/adsi-at2/adsi-at2/data/beer_reviews.csv')

# # Fit the LabelEncoder on the 'brewery_name' column
# encoder = LabelEncoder()
# encoder.fit(df['brewery_name'])

# # # Define the URL of your FastAPI application
# # url = 'http://localhost:8000/beer/type'

# # # Create a form
# # with st.form(key='beer_form'):
# #     st.write("Fill in the beer characteristics:")
# #     appearance = st.number_input(label='Appearance', min_value=0)
# #     aroma = st.number_input(label='Aroma', min_value=0)
# #     palate = st.number_input(label='Palate', min_value=0)
# #     taste = st.number_input(label='Taste', min_value=0)
# #     abv = st.number_input(label='ABV', min_value=0.0)
# #     brewery = st.text_input(label='Brewery')
# #     submit_button = st.form_submit_button(label='Predict Beer Type')

# # # When the user clicks the "Predict Beer Type" button, send a GET request to the FastAPI application
# # if submit_button:
# #     response = requests.get(url, params={
# #         'review_appearance': appearance,
# #         'review_aroma': aroma,
# #         'review_palate': palate,
# #         'review_taste': taste,
# #         'beer_abv': abv,
# #         'brewery_name': brewery,
# #     })
# #     if response.status_code == 200:
# #         st.write(f'Predicted beer type: {response.json()}')
# #     else:
# #         st.write('An error occurred.')

# @app.post("/beer/type")
# def predict(review_appearance: float, review_aroma: float, review_palate: float, review_taste: float, beer_abv: float, brewery_name: str):
#     # Encode the brewery_name
#     brewery_name_encoded = encoder.transform([brewery_name])

#     features = format_features(review_appearance, review_aroma, review_palate, review_taste, beer_abv, brewery_name_encoded)
#     obs = pd.DataFrame(features)
#     obs = obs.astype('float32')  # Convert data to float32
#     global model
#     pred = model.predict(obs)
#     return JSONResponse(pred.tolist())


from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model = load_model("C:/Users/justi/adsi-at2/adsi-at2/models/best.h5")

# Define the application
app = FastAPI()

# Define the model input
class ModelInput(BaseModel):
    review_aroma: float
    review_appearance: float
    review_palate: float
    review_taste: float
    beer_abv: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Beer Type Predictor"}

@app.get("/health/")
def health_check():
    return {"message": "API is healthy"}

@app.post("/beer/type/")
def predict_beer_type(input_data: ModelInput):
    # Convert the input to a numpy array
    input_array = np.array([
        input_data.review_aroma,
        input_data.review_appearance,
        input_data.review_palate,
        input_data.review_taste,
        input_data.beer_abv
    ]).reshape(1, -1)
    # Make the prediction
    prediction = model.predict(input_array)
    # Return the prediction
    return {"prediction": prediction.tolist()}

@app.post("/beers/type/")
def predict_beer_types(input_data: List[ModelInput]):
    # Convert the input to a numpy array
    input_array = np.array([
        [data.review_aroma, data.review_appearance, data.review_palate, data.review_taste, data.beer_abv]
        for data in input_data
    ])
    # Make the predictions
    predictions = model.predict(input_array)
    # Return the predictions
    return {"predictions": predictions.tolist()}

@app.get("/model/architecture/")
def model_architecture():
    # Get the model's architecture
    architecture = [layer.get_config() for layer in model.layers]
    # Return the architecture
    return {"architecture": architecture}