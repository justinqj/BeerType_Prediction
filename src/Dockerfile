# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# WORKDIR /src

# COPY ./requirements.txt ./requirements.txt
# RUN pip3 install -r requirements.txt

# COPY ./app ./app
# COPY ./models ./models

# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "app.main:app"]


# # Using the FastAPI image
# FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# # Setting the working directory in the Docker image
# WORKDIR /src

# # Copying your requirements file into the image
# COPY ./requirements.txt ./requirements.txt

# # Install the Python dependencies
# RUN pip3 install -r requirements.txt

# # Copy your application code to the image
# COPY ./app ./app
# COPY ./models ./models

# # Run the application
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]


FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY ./app /app

COPY ./models /models

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-c", "/gunicorn_conf.py", "main:app"]