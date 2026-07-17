# Steelworks Energy Control — Dimensionality Reduction & Flask Dashboard

**Week 3 Internship Task** — Steel Industry Energy Consumption

A two-part project: (1) apply PCA to reduce the Week 2 feature set while
preserving Random Forest accuracy, and (2) deploy the resulting pipeline
behind a 3-page Flask dashboard built for plant-level decision-making.

---

## 1. Project Overview

Steel plants log energy usage every 15 minutes across three operating load
bands — Light, Medium, and Maximum. The Week 2 project engineered 20 input
features from these readings (electrical measurements, time-based features,
one-hot encoded categories) and trained a Random Forest to predict
`Usage_kWh` with very high accuracy (R² = 0.999).

This week asks two questions:

1. **Can that accuracy survive dimensionality reduction?** If most of the
   20 features carry redundant information, PCA should be able to compress
   them into far fewer components with only a small accuracy cost — useful
   for lighter, faster, more memory-efficient deployment.
2. **Can the result be put in front of a non-technical decision-maker?**
   The second half of the task deploys the chosen pipeline inside a Flask
   dashboard: three pages covering plant overview, consumption trends and
   emissions, and a live prediction tool with model performance comparison.

---

## 2. Dataset Information

| | |
|---|---|
| **Source** | Preprocessed & engineered dataset from Week 2 (`steel_industry_engineered.csv`) |
| **Records** | 35,040 rows (15-minute interval readings) |
| **Date range** | 2018-01-01 to 2018-12-12 |
| **Target variable** | `Usage_kWh` |
| **Raw columns** | `date`, `Usage_kWh`, `Lagging_Current_Reactive.Power_kVarh`, `Leading_Current_Reactive_Power_kVarh`, `CO2(tCO2)`, `Lagging_Current_Power_Factor`, `Leading_Current_Power_Factor`, `NSM`, `WeekStatus`, `Day_of_week`, `Load_Type`, `Hour`, `DayOfWeek_Num`, `Month`, `Is_Weekend`, `Power_Factor_Ratio`, `High_Load` |
| **Dropped before modeling** | `date` (identifier, not a feature), `High_Load` (leakage risk) |
| **Categorical columns (one-hot encoded)** | `WeekStatus`, `Day_of_week`, `Load_Type` — encoded with `drop_first=True` |
| **Final input feature count** | 20 |
| **Train/test split** | 80/20, `random_state=42`, identical split to Week 2 for fair comparison |

**Headline KPIs computed from the full dataset** (see Overview & KPIs page):

| Metric | Value |
|---|---|
| Records analyzed | 35,040 |
| Average usage | 27.39 kWh |
| Peak usage | 157.18 kWh |
| Total CO₂ output | 403.8 tCO₂ |
| Average power factor | 80.6 |
| Weekend share of readings | 28.5% |

---

## 3. Environment Setup

### Requirements
- Python 3.10–3.12 recommended (3.13+ works but may need unpinned package
  versions — see note below)
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/steelworks-energy-dashboard.git
cd steelworks-energy-dashboard

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

> **Very new Python versions (3.13+):** if pinned versions in
> `requirements.txt` fail to build from source, install unpinned instead —
> pip will resolve compatible versions automatically:
> `pip install flask pandas numpy scikit-learn joblib matplotlib seaborn`

### Add the dataset
Place the Week 2 engineered CSV in the project root, named exactly:
```
steel_industry_engineered.csv
```

---

## 4. Steps to Reproduce

### Part 1 — PCA analysis (notebook)
Open `notebooks/week3_pca.ipynb` and run all cells top to bottom. It will:
1. Load and re-encode the dataset exactly as in Week 2
2. Fit `StandardScaler` and `PCA` on the training set only
3. Plot a scree plot and cumulative explained-variance curve
4. Retrain the Random Forest on 3 PCA components, then on the number of
   components needed for 95% variance
5. Compare RMSE/R² across the original model and both PCA variants
6. Plot a PCA loading heatmap (which raw features drive PC1–PC3)
7. Save the final pipeline as `model.joblib`

### Part 2 — Flask dashboard
```bash
python flask_training/train_pipeline.py           # trains model, saves model.joblib + metrics.json
python flask_training/generate_dashboard_plots.py  # builds charts + kpis.json + trends.json
python app.py                                       # starts the dashboard
```
Open **http://127.0.0.1:5000** in a browser.

---

## 5. Model Training Process

1. **Feature engineering** — reused Week 2's 20-column feature set: 11
   numeric readings (electrical values, time features) plus 9 one-hot
   dummy columns from `WeekStatus`, `Day_of_week`, and `Load_Type`.
2. **Leakage control** — `StandardScaler` and `PCA` are fit **only** on the
   training split, then applied (never re-fit) to the test split, so no
   information from test rows leaks into the transformation.
3. **Full-variance PCA pass** — PCA is first run with `n_components` equal
   to the total feature count (20) purely to inspect how variance is
   distributed, not to reduce dimensions yet.
4. **Component selection** — the cumulative explained-variance curve shows
   how many components are needed to reach 95% of total variance;
   **10 components** cross that threshold.
5. **Three Random Forests trained side by side** on the identical train/test
   split for a fair comparison:
   - **Original** — all 20 features, no PCA
   - **PCA-3** — only the first 3 principal components
   - **PCA-10** — the 10 components covering 95% of variance (**deployed model**)
6. **Pipeline bundling** — the fitted `scaler`, `pca`, `n_components`,
   trained `model`, and the exact `feature_names` order are bundled into one
   `model.joblib` file with `joblib.dump()`, so the Flask app never has to
   re-fit or guess column order at inference time.

---

## 6. Results

| Model | Features | RMSE | R² |
|---|---|---|---|
| Original (Week 2, all features) | 20 | 1.0431 | 0.9990 |
| PCA — 3 components | 3 | 7.1494 | 0.9550 |
| **PCA — 10 components (95% variance)** — *deployed* | 10 | 3.2140 | 0.9909 |

- Explained variance climbs steeply through the first few components (see
  the scree plot in the notebook) and crosses 95% cumulative variance at
  **10 of 20** components — half the original feature count is redundant
  enough to drop with only a modest accuracy cost.
- The **10-component model** is the deployment choice: R² stays above 0.99
  while cutting the input footprint in half.
- The **3-component model** trades away noticeably more accuracy (R² drops
  to 0.955), since compressing everything into 3 axes loses signal tied to
  `Load_Type` and the time-based features — only advisable for severely
  memory-constrained scenarios.
- The PCA loading heatmap shows the electrical-reading features
  (`Lagging_Current_Reactive.Power_kVarh`, `CO2(tCO2)`) dominate the first
  principal component, while time-based features spread across later
  components.

---

## 7. Conclusions

**Did accuracy drop significantly?**
Only marginally, for the model actually deployed. Going from 20 features to
10 PCA components costs about 2 RMSE units and under 1 point of R² — a
reasonable trade for a smaller input pipeline. The 3-component version drops
further, showing there's a real floor below which compression starts
discarding meaningful signal.

**How many features can safely be removed?**
Roughly half. Ten of the twenty engineered/one-hot columns are redundant
enough with each other (e.g. `CO2(tCO2)` moves closely with the electrical
readings) that removing them costs almost nothing in accuracy.

**Would PCA be recommended for a memory-constrained device?**
Yes, with a caveat. PCA meaningfully shrinks the *input and scaler side* of
the pipeline — fewer columns to store, transform, and pass to the model.
The caveat is that a Random Forest's own memory footprint is driven by tree
depth and count, not input width, so PCA alone doesn't shrink the trained
model itself. Pairing PCA with a lighter model (a shallow tree or a linear
model on the components) would give a bigger combined win for a genuinely
constrained device.

**Is the dashboard usable by a non-technical decision-maker?**
Yes — the three pages separate "what's happening now" (Overview & KPIs),
"what's driving it" (Trends & CO2), and "what happens next" (Prediction &
Model Performance), each built around a handful of numbers and charts rather
than raw tables.

---

## 8. Screenshots

| Page | Preview |
|---|---|
| Home | ![Home](screenshots/01_home.png) |
| Overview & KPIs | ![Overview & KPIs](screenshots/02_dashboard.png) |
| Trends & CO2 | ![Trends & CO2](screenshots/03_trends.png) |
| Prediction & Model Performance | ![Prediction & Model Performance](screenshots/04_predict.png) |

> Save your own screenshots into `screenshots/` using the filenames above —
> GitHub will render them inline automatically. To capture a full scrollable
> page in Chrome: open DevTools (`F12`) → `Ctrl+Shift+P` → type **"Capture
> full size screenshot"** → Enter.

---

## 9. Tech Stack

- **Backend:** Flask, Python
- **Modeling:** scikit-learn (`StandardScaler`, `PCA`, `RandomForestRegressor`), joblib
- **Data:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Frontend:** Jinja2 templates, vanilla CSS (dark industrial theme, no framework)

---

## 10. Project Structure

```
steel_dashboard/
├── app.py                              # Flask app — all routes
├── requirements.txt
├── README.md
├── model.joblib                        # generated by train_pipeline.py
├── metrics.json                        # generated by train_pipeline.py
├── kpis.json                           # generated by generate_dashboard_plots.py
├── trends.json                         # generated by generate_dashboard_plots.py
├── steel_industry_engineered.csv       # Week 2 dataset (add this yourself)
├── templates/
│   ├── base.html                       # nav bar + shared shell
│   ├── index.html                      # home page
│   ├── dashboard.html                  # Page 1: Overview & KPIs
│   ├── trends.html                     # Page 2: Trends & CO2
│   └── predict.html                    # Page 3: Prediction & Model Performance
├── static/
│   ├── style.css                       # dark industrial theme
│   └── plots/                          # generated PNGs
│       ├── energy_by_hour.png
│       ├── correlation_heatmap.png
│       ├── energy_by_load_type.png
│       ├── co2_by_load_type.png
│       └── weekday_weekend.png
├── flask_training/
│   ├── train_pipeline.py               # trains model, saves model.joblib + metrics.json
│   └── generate_dashboard_plots.py     # builds PNGs + kpis.json + trends.json
├── notebooks/
│   └── week3_pca.ipynb                 # Part 1: PCA analysis, all cells run
└── screenshots/                        # dashboard screenshots for this README
```

---

## 11. Deliverables Checklist

- [x] `app.py` and all Flask project files
- [x] `templates/` and `static/` folders
- [x] `notebooks/week3_pca.ipynb` with all cells run and outputs visible
- [x] `model.joblib`
- [x] `README.md` and `requirements.txt`
- [ ] Screenshots added to `screenshots/`
- [ ] LinkedIn post with dashboard screenshot and write-up

---

## Author

Muqadas — University Internship Task, Week 3
