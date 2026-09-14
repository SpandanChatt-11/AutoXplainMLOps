# AutoXplain inference app.
#
# IMPORTANT: run `python -m src.train` locally at least once BEFORE building this
# image. That step creates models/, mlflow.db, and mlruns/ - the trained model
# artifacts this image bakes in. `docker build` will still succeed without them,
# but the container will fail at startup with no model to load.

FROM python:3.11-slim

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "app.app:app", "--bind", "0.0.0.0:5000", "--timeout", "120"]
