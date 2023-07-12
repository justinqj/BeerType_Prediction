from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import tensorflow as tf
from typing import List


# Define the app
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model, le_brewery, le_beer_style, le_beer_name  # Make them global so they can be accessed in other routes
    try:
        # Load the model
        model = load_model("../../models/best.h5")

        # Load the encoders
        with open("le_brewery.pkl", "rb") as f:
            le_brewery = pickle.load(f)
        with open("le_beer_style.pkl", "rb") as f:
            le_beer_style = pickle.load(f)
        with open("le_beer_name.pkl", "rb") as f:
            le_beer_name = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def welcome():
    return {"Description":"The aim of this project is to create a neural network machine learning model that will accurately predict a type of beer based on some rating criterias such as appearance, aroma, palate or taste. This model will then be packaged in a docker container along with a fastapi script to create a webapp for hosting this model, capable of making ongoing predictions in real time.",
            "Endpoints":"'/','health','beer/type','beers/type','model/architecture'",
            "Input":"brewery_name: str, beer_name: str, review_aroma: float, review_appearance: float, review_palate: float, review_taste: float, beer_abv: float",
            "Output":"predicted_beer_style: str",
            "Github Repo":"https://github.com/danish-sven/adsi-at2"
            }

@app.get('/health', status_code=200)
def healthcheck():
    return 'Neural Net is all ready to go!'

@app.get('/model/architecture')
def architecture():
    return {"Model Architecture": [layer.get_config() for layer in model.layers]}

# Define the input model
class BeerReview(BaseModel):
    brewery_name: str
    beer_name: str
    review_aroma: float
    review_appearance: float
    review_palate: float
    review_taste: float
    beer_abv: float

@app.post("/beer/type")
def predict(beer_review: BeerReview):
    try:
        # Encode the brewery_name
        brewery_name = le_brewery.transform([beer_review.brewery_name])

        # Encode the beer_name
        beer_name = le_beer_name.transform([beer_review.beer_name])

        # Create the feature vector
        features = np.array([brewery_name, beer_name, beer_review.review_aroma, beer_review.review_appearance, 
                            beer_review.review_palate, beer_review.review_taste, beer_review.beer_abv]).reshape(1, -1)

        # Convert numpy array to tensorflow tensor
        features = tf.convert_to_tensor(features, dtype=tf.float32)

        # Make the prediction
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)

        # Decode the prediction
        beer_style = le_beer_style.inverse_transform([predicted_class])

        return {"predicted_beer_style": beer_style[0]}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/beers/type")
def predict_multiple(beer_reviews: List[BeerReview]):
    predictions = []
    for beer_review in beer_reviews:
        try:
            # Encode the brewery_name
            brewery_name = le_brewery.transform([beer_review.brewery_name])

            # Encode the beer_name
            beer_name = le_beer_name.transform([beer_review.beer_name])

            # Create the feature vector
            features = np.array([brewery_name, beer_name, beer_review.review_aroma, beer_review.review_appearance, 
                                beer_review.review_palate, beer_review.review_taste, beer_review.beer_abv]).reshape(1, -1)

            # Convert numpy array to tensorflow tensor
            features = tf.convert_to_tensor(features, dtype=tf.float32)

            # Make the prediction
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction)

            # Decode the prediction
            beer_style = le_beer_style.inverse_transform([predicted_class])

            predictions.append({"brewery_name": beer_review.brewery_name, "predicted_beer_style": beer_style[0]})
        except Exception as e:
            predictions.append({"brewery_name": beer_review.brewery_name, "error": str(e)})
    return predictions

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"An error occurred: {exc.detail}"},
    )