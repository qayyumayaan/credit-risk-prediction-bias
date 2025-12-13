import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ===============================
# Load model bundle
# ===============================
MODEL_PATH = "biased_bundle.joblib"

bundle = joblib.load(MODEL_PATH)
ct = bundle["ct"]        # ColumnTransformer
model = bundle["model"]  # LightGBM model

# ===============================
# Prepare column metadata
# ===============================

# Raw columns expected by ColumnTransformer
cols = list(ct.feature_names_in_)

# Identify categorical columns from transformers
binary_cols = [c for name, _, c in ct.transformers_ if name == "binary"][0]
onehot_cols = [c for name, _, c in ct.transformers_ if name == "onehot"][0]

cat_cols = list(binary_cols) + list(onehot_cols)
num_cols = [c for c in cols if c not in cat_cols]

# Extract encoders
ord_enc = ct.named_transformers_["binary"]
ohe_enc = ct.named_transformers_["onehot"]

# Safe default categorical values (always valid)
cat_defaults = {
    binary_cols[i]: ord_enc.categories_[i][0]
    for i in range(len(binary_cols))
}
cat_defaults.update({
    onehot_cols[i]: ohe_enc.categories_[i][0]
    for i in range(len(onehot_cols))
})

# ===============================
# Routes
# ===============================

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input
    if request.is_json:
        params = request.get_json()
        print("JSON input:", params)
    else:
        params = request.form.to_dict()

    # Build DataFrame
    X = pd.DataFrame([params])

    # Add missing raw columns
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[cols]

    # Force numeric columns
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Fill categorical columns with safe known values
    for c in cat_cols:
        X[c] = X[c].astype(object)
        X.loc[X[c].isna(), c] = cat_defaults[c]

    # Predict
    prob = float(model.predict_proba(ct.transform(X))[0, 1])
    pred = int(prob >= 0.5)

    return jsonify({
        "prediction": pred,
        "label": "Default" if pred else "No default",
        "probability": round(prob, 4)
    })

# ===============================
# Run app
# ===============================
if __name__ == "__main__":
    app.run(debug=True)
