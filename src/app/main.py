from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Define the app
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global model, le_brewery, le_beer_style  # Make them global so they can be accessed in other routes

    try:
        # Load the model
        model = load_model("C:/Users/justi/adsi-at2/adsi-at2/models/beer_style_predictor.h5")

        # Load the encoders
        with open("C:/Users/justi/adsi-at2/adsi-at2/src/app/le_brewery.pkl", "rb") as f:
            le_brewery = pickle.load(f)
        with open("C:/Users/justi/adsi-at2/adsi-at2/src/app/le_beer_style.pkl", "rb") as f:
            le_beer_style = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # Load the model
# model = load_model("C:/Users/justi/adsi-at2/adsi-at2/models/beer_style_predictor.h5")

# # Load the encoders
# with open("C:/Users/justi/adsi-at2/adsi-at2/src/app/le_brewery.pkl", "rb") as f:
#     le_brewery = pickle.load(f)

# with open("C:/Users/justi/adsi-at2/adsi-at2/src/app/le_beer_style.pkl", "rb") as f:
#     le_beer_style = pickle.load(f)

# Define the input model
class BeerReview(BaseModel):
    brewery_name: str
    review_aroma: float
    review_appearance: float
    review_palate: float
    review_taste: float
    beer_abv: float

@app.post("/predict")
def predict(beer_review: BeerReview):
    try:
        # Encode the brewery_name
        brewery_name = le_brewery.transform([beer_review.brewery_name])

        # Create the feature vector
        features = np.array([brewery_name, beer_review.review_aroma, beer_review.review_appearance, 
                            beer_review.review_palate, beer_review.review_taste, beer_review.beer_abv]).reshape(1, -1)

        # Make the prediction
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction)

        # Decode the prediction
        beer_style = le_beer_style.inverse_transform([predicted_class])

        return {"predicted_beer_style": beer_style[0]}
    except Exception as e:
        return {"error": str(e)}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": f"An error occurred: {exc.detail}"},
    )
