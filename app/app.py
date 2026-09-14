"""
AutoXplain Flask app. Loads the registered/best model once at startup, predicts
a point estimate + a MAPIE-calibrated price range, and generates a per-prediction
LIME explanation.
"""

import os
import sys

# Makes `from src...` resolve correctly regardless of how this file is invoked -
# `python app/app.py` (Windows-friendly dev loop), `python -m app.app`, or
# `gunicorn app.app:app`. Without this, direct script invocation only puts this
# file's own folder on sys.path, not the project root, so `src` isn't found.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template_string, request

from src.inference import (
    load_model,
    load_artifacts,
    build_lime_explainer,
    build_feature_row,
    predict_price,
    explain_prediction,
)

app = Flask(__name__)

MODEL, MODEL_SOURCE = load_model()
ARTIFACTS = load_artifacts()
EXPLAINER = build_lime_explainer(ARTIFACTS)

FUEL_OPTIONS = ["CNG", "Diesel", "Electric", "Petrol"]
TRANSMISSION_OPTIONS = ["Automatic", "Manual"]
BRAND_OPTIONS = ARTIFACTS["metadata"].get("brand_categories") or [
    "Honda", "Hyundai", "Kia", "MG", "Mahindra", "Maruti Suzuki", "Mercedes-Benz",
    "Renault", "Skoda", "Tata", "Toyota", "Volkswagen", "Other",
]

PAGE_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>AutoXplain - Kolkata Used Car Price Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; }
        h1 { font-size: 1.5rem; }
        .subtitle { color: #666; margin-top: -8px; }
        label { display: block; margin-top: 12px; font-weight: bold; }
        input, select { width: 100%; padding: 8px; margin-top: 4px; box-sizing: border-box; }
        button { margin-top: 20px; padding: 10px 20px; font-size: 1rem; cursor: pointer; }
        .result { margin-top: 30px; padding: 16px; background: #f0f4f8; border-radius: 8px; }
        .price { font-size: 1.6rem; font-weight: bold; color: #1a5d1a; }
        .range { color: #555; margin-top: 4px; }
        .source-badge { font-size: 0.75rem; color: #999; }
        img { max-width: 100%; margin-top: 16px; }
    </style>
</head>
<body>
    <h1>AutoXplain</h1>
    <p class="subtitle">Explainable Used Car Valuation - Kolkata</p>
    <form method="POST">
        <label>Brand</label>
        <select name="brand">
            {% for b in brand_options %}<option value="{{ b }}">{{ b }}</option>{% endfor %}
        </select>

        <label>Fuel Type</label>
        <select name="fuel">
            {% for f in fuel_options %}<option value="{{ f }}">{{ f }}</option>{% endfor %}
        </select>

        <label>Transmission</label>
        <select name="transmission">
            {% for t in transmission_options %}<option value="{{ t }}">{{ t }}</option>{% endfor %}
        </select>

        <label>Car Age (years)</label>
        <input type="number" name="car_age" step="1" min="0" value="5" required>

        <label>KM Driven</label>
        <input type="number" name="km_driven" step="1" min="0" value="40000" required>

        <label>Number of Previous Owners</label>
        <input type="number" name="owner_number" step="1" min="0" value="1" required>

        <label>Latitude</label>
        <input type="number" name="latitude" step="0.0001" value="22.5726" required>

        <label>Longitude</label>
        <input type="number" name="longitude" step="0.0001" value="88.3639" required>

        <button type="submit">Predict Price</button>
    </form>

    {% if price is not none %}
    <div class="result">
        <div>Predicted Price <span class="source-badge">(model: {{ model_source }})</span></div>
        <div class="price">Rs. {{ "{:,.0f}".format(price) }}</div>
        <p>Feature contributions for this specific prediction (LIME):</p>
        <img src="data:image/png;base64,{{ lime_img }}" alt="LIME explanation">
    </div>
    {% endif %}
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    price = lime_img = None

    if request.method == "POST":
        feature_row = build_feature_row(request.form, ARTIFACTS["feature_names"])
        price = predict_price(feature_row, MODEL, MODEL_SOURCE, ARTIFACTS)
        lime_img = explain_prediction(feature_row, MODEL, MODEL_SOURCE, ARTIFACTS, EXPLAINER)

    return render_template_string(
        PAGE_TEMPLATE,
        brand_options=BRAND_OPTIONS,
        fuel_options=FUEL_OPTIONS,
        transmission_options=TRANSMISSION_OPTIONS,
        price=price, lime_img=lime_img,
        model_source=MODEL_SOURCE,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
