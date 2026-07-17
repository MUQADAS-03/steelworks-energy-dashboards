"""
Steelworks Energy Control — Flask Dashboard
Week 3 - Part 2

Loads the PCA + Random Forest pipeline saved in model.joblib and serves a
3-page dark-industrial dashboard:

  /            - home page with navigation
  /dashboard   - Page 1: Overview & KPIs
  /trends      - Page 2: Trends, CO2, Load Types
  /predict     - Page 3: Prediction & Model Performance

Run flask_training/train_pipeline.py first to create model.joblib and
metrics.json, and flask_training/generate_dashboard_plots.py to create the
static/plots/*.png charts and kpis.json / trends.json used on pages 1 and 2.
"""

import json
import os

import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")
KPIS_PATH = os.path.join(BASE_DIR, "kpis.json")
TRENDS_PATH = os.path.join(BASE_DIR, "trends.json")
PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")

# ---------------------------------------------------------------------------
# Load the trained pipeline bundle (scaler + pca + model + feature order)
# ---------------------------------------------------------------------------
bundle = None
scaler = pca = model = None
n_components = None
feature_names = []
model_loaded = False

if os.path.exists(MODEL_PATH):
    bundle = joblib.load(MODEL_PATH)
    scaler = bundle["scaler"]
    pca = bundle["pca"]
    n_components = bundle["n_components"]
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    model_loaded = True

# Raw numeric columns that appear directly in feature_names (unchanged by
# one-hot encoding). Everything else in feature_names is a one-hot dummy
# column built from WeekStatus / Day_of_week / Load_Type.
NUMERIC_FEATURES = [
    "Lagging_Current_Reactive.Power_kVarh",
    "Leading_Current_Reactive_Power_kVarh",
    "CO2(tCO2)",
    "Lagging_Current_Power_Factor",
    "Leading_Current_Power_Factor",
    "NSM",
    "Hour",
    "DayOfWeek_Num",
    "Month",
    "Is_Weekend",
    "Power_Factor_Ratio",
]

CATEGORICAL_FEATURES = {
    "WeekStatus": ["Weekday", "Weekend"],
    "Day_of_week": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    "Load_Type": ["Light_Load", "Medium_Load", "Maximum_Load"],
}

# ---------------------------------------------------------------------------
# Fallback model performance numbers (from the Week 3 Part 1 notebook run).
# Overwritten by metrics.json if flask_training/train_pipeline.py has produced one.
# ---------------------------------------------------------------------------
DEFAULT_METRICS = [
    {"name": "Original (Week 2, all features)", "rmse": "1.0532", "r2": "0.9990", "n_features": 20, "deployed": False},
    {"name": "PCA - 3 components", "rmse": "7.1519", "r2": "0.9550", "n_features": 3, "deployed": False},
    {"name": "PCA - 10 components (95% variance)", "rmse": "3.2098", "r2": "0.9909", "n_features": 10, "deployed": True},
]


def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_metrics():
    data = load_json(METRICS_PATH)
    if data:
        return data
    return DEFAULT_METRICS


def plots_present():
    names = ["energy_by_hour", "correlation_heatmap", "energy_by_load_type", "co2_by_load_type", "weekday_weekend"]
    return {n: os.path.exists(os.path.join(PLOTS_DIR, n + ".png")) for n in names}


@app.route("/")
def home():
    return render_template("index.html", active="home")


@app.route("/dashboard")
def dashboard():
    kpis = load_json(KPIS_PATH)
    return render_template("dashboard.html", active="dashboard", kpis=kpis, plots=plots_present())


@app.route("/trends")
def trends():
    trends_data = load_json(TRENDS_PATH)
    return render_template("trends.html", active="trends", trends=trends_data, plots=plots_present())


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None
    form_values = {}

    if request.method == "POST":
        form_values = request.form.to_dict()
        try:
            if not model_loaded:
                raise RuntimeError("Model pipeline is not loaded.")

            # Start every engineered column at 0, then fill in what the user gave us.
            row = {f: 0.0 for f in feature_names}

            for feat in NUMERIC_FEATURES:
                if feat in row:
                    row[feat] = float(request.form.get(feat, 0))

            for cat, options in CATEGORICAL_FEATURES.items():
                selected = request.form.get(cat)
                col_name = f"{cat}_{selected}"
                if col_name in row:
                    row[col_name] = 1.0
                # if selected is the drop_first baseline category, every
                # dummy column for that categorical correctly stays 0.

            input_df = pd.DataFrame([row], columns=feature_names)

            scaled = scaler.transform(input_df)
            pca_transformed = pca.transform(scaled)[:, :n_components]
            pred = model.predict(pca_transformed)[0]
            prediction = round(float(pred), 3)

        except Exception as exc:
            error = f"Could not generate a prediction: {exc}"

    return render_template(
        "predict.html",
        active="predict",
        prediction=prediction,
        error=error,
        form_values=form_values,
        model_loaded=model_loaded,
        metrics=load_metrics(),
    )


if __name__ == "__main__":
    app.run(debug=True)
