adsi-at2
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

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
