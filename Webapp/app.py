import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

BUNDLES = {
    "biased": joblib.load("biased_bundle.joblib"),
    "unbiased": joblib.load("unbiased_bundle.joblib"),
}

def build_meta(bundle):
    ct, model = bundle["ct"], bundle["model"]
    cols = list(ct.feature_names_in_)
    allowed = set(cols)

    # identify categorical columns
    tmap = {name: cols_ for name, _, cols_ in ct.transformers_}
    binary_cols = list(tmap.get("binary", []))
    onehot_cols = list(tmap.get("onehot", []))
    cat_cols = binary_cols + onehot_cols
    num_cols = [c for c in cols if c not in cat_cols]

    # safe defaults for categoricals
    cat_defaults = {}
    if "binary" in ct.named_transformers_ and binary_cols:
        ord_enc = ct.named_transformers_["binary"]
        for i, c in enumerate(binary_cols):
            cat_defaults[c] = ord_enc.categories_[i][0]
    if "onehot" in ct.named_transformers_ and onehot_cols:
        ohe_enc = ct.named_transformers_["onehot"]
        for i, c in enumerate(onehot_cols):
            cat_defaults[c] = ohe_enc.categories_[i][0]

    return {
        "ct": ct, "model": model,
        "cols": cols, "allowed": allowed,
        "cat_cols": cat_cols, "num_cols": num_cols,
        "cat_defaults": cat_defaults,
    }

META = {k: build_meta(v) for k, v in BUNDLES.items()}

def preprocess_params(params: dict, meta: dict) -> pd.DataFrame:
    # keep only features the model knows
    params = {k: v for k, v in params.items() if k in meta["allowed"]}

    X = pd.DataFrame([params]).reindex(columns=meta["cols"])  # fast, no fragmentation

    # numeric coercion
    for c in meta["num_cols"]:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # categorical NaN -> safe known category
    for c in meta["cat_cols"]:
        X[c] = X[c].astype(object)
        if c in meta["cat_defaults"]:
            X.loc[X[c].isna(), c] = meta["cat_defaults"][c]

    return X

def predict_one(params: dict, meta: dict):
    ct, model = meta["ct"], meta["model"]
    X = preprocess_params(params, meta)

    prob = float(model.predict_proba(ct.transform(X))[0, 1])  # "default rate" (prob of TARGET=1)
    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "label": "Default" if pred else "No default",
        "default_rate": round(prob, 6),
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compare", methods=["POST"])
def compare():
    try:
        params = request.get_json(force=True) if request.is_json else request.form.to_dict()

        out = {
            "biased": predict_one(params, META["biased"]),
            "unbiased": predict_one(params, META["unbiased"]),
        }
        return jsonify(out)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# optional debugging: see expected raw columns
@app.route("/schema")
def schema():
    return jsonify({
        "biased_expected_columns": META["biased"]["cols"],
        "unbiased_expected_columns": META["unbiased"]["cols"],
    })

if __name__ == "__main__":
    app.run(debug=True, port=7000)
