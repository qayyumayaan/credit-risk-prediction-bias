# Credit Risk Prediction + Bias Mitigation

End-to-end project for **credit default risk prediction** with an added focus on **fairness / bias mitigation**.  
Includes multiple model baselines (Logistic Regression, Random Forest, LightGBM), model export utilities, and a simple web app for interactive inference.

**Authors**
- Ayaan Qayyum (aaq2109)
- Swapnil Banerjee (sb5041)
- Vatsalam Krishna Jha (vkj2107)


## What's inside

This repository is organized into the following modules/folders:

- `Logistic Regression/` — baseline pipeline using Logistic Regression
- `Random Forest/` — Random Forest baseline
- `LGBM+bias_mitigation/` — LightGBM model + bias mitigation experiments
- `Model_export/` — exporting trained artifacts (e.g., joblib/pickle) for deployment
- `Webapp/` — lightweight web application for running predictions via UI
- `requirements.txt` — Python dependencies

> If you’re using a specific dataset (e.g., Home Credit Default Risk), keep the raw data **out of git** and follow the “Data” section below.

---

## Project goals

1. Train and compare multiple **credit risk prediction** models.
2. Measure model performance using standard classification metrics (e.g., AUC, accuracy, precision/recall).
3. Evaluate **bias/fairness** across sensitive attributes (e.g., gender, age bucket, etc.).
4. Apply bias mitigation techniques and compare:
   - predictive performance (AUC/accuracy)
   - fairness metrics (e.g., demographic parity / equal opportunity gaps)
5. Export the best-performing model bundle(s) and serve predictions via a web UI.

---

## Setup

### 1) Clone
```bash
git clone https://github.com/qayyumayaan/credit-risk-prediction-bias.git
cd credit-risk-prediction-bias
```

### 2) Create a venv
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### 3) Run the webapp
```bash
cd Webapp
flask run
```