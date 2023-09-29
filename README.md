Beer Type Prediction using CNN model
==============================
The aim of this project is to create a neural network machine learning model that will accurately predict a type of beer based on some rating criterias such as appearance, aroma, palate or taste. This model will then be packaged in a docker container along with a fastapi script to create a webapp for hosting this model, capable of making ongoing predictions in real time.

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── Pipfile            <- Generated list of packages for reproducing dev environment as 
    |                         a virtual environment, eg generated with `pipenv update`
    ├── Pipfile.lock       <- Generated list of packages and their dependencies, similar to    
    |                         Pipfile, only non-human-readable.
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, eg.
    │                         generated with `pip freeze > requirements.txt`
    │
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── app            <- folder housing main.py. Core functionality of the webapp
        │
        └── models         <- Model artefacts to load into the webapp


--------

## Methods for Generating Predictions
1. Deploying Webapp on localhost through uvicorn \
navigate to `src/app` and run the command `uvicorn main:app --reload` in your terminal \
then navigate to `https://localhost:8000` \
from here you can access the API endpoints and enter in values to generate one or more predictions

2. Deploy a docker container to run the webapp \
from the `src` folder, run the command `docker build -t webapp:latest .` to build the docker image and `docker run -dit --rm --name apptest -p 8080:80 webapp:latest` to build a docker container to run the image. \
From here, navigate to `https:localhost:8080` to similarly access the endpoints.

3. Go to `https://stark-scrubland-51062-3ad44a35e5f2.herokuapp.com/` to access the webapp \
-- still in development --

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
