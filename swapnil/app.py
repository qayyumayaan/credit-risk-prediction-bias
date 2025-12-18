# app.py (biased bundle only)
from flask import Flask, request, jsonify, render_template_string
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# ----------------------------
# Load biased bundle at startup
# ----------------------------
b = joblib.load("biased_bundle.joblib")
ct, model = b["ct"], b["model"]

# Raw columns expected by this ColumnTransformer
cols = list(ct.feature_names_in_)

# Identify categorical columns from ct (robust)
tmap = {name: cols_ for name, _, cols_ in ct.transformers_}
binary_cols = list(tmap.get("binary", []))
onehot_cols  = list(tmap.get("onehot", []))
cat_cols = binary_cols + onehot_cols
num_cols = [c for c in cols if c not in cat_cols]

# Safe defaults from fitted encoders (always "known" categories)
cat_defaults = {}
if "binary" in ct.named_transformers_ and len(binary_cols) > 0:
    ord_enc = ct.named_transformers_["binary"]
    for i, c in enumerate(binary_cols):
        cat_defaults[c] = ord_enc.categories_[i][0]

if "onehot" in ct.named_transformers_ and len(onehot_cols) > 0:
    ohe_enc = ct.named_transformers_["onehot"]
    for i, c in enumerate(onehot_cols):
        cat_defaults[c] = ohe_enc.categories_[i][0]


def preprocess_params(params: dict) -> pd.DataFrame:
    """Turn partial JSON/dict into a 1-row DataFrame ready for ct.transform()."""
    X = pd.DataFrame([params])

    # Add missing raw columns as NaN + order exactly as training expected
    for c in cols:
        if c not in X.columns:
            X[c] = np.nan
    X = X[cols]

    # Coerce numeric passthrough columns to numeric (prevents isnan crash)
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Fill categorical NaNs with safe known defaults (prevents encoder NaN crash)
    for c in cat_cols:
        X[c] = X[c].astype(object)
        if c in cat_defaults:
            X.loc[X[c].isna(), c] = cat_defaults[c]

    return X


def predict_from_params(params: dict):
    X = preprocess_params(params)
    prob = float(model.predict_proba(ct.transform(X))[0, 1])
    target = int(prob >= 0.5)  # fixed rule
    risk_level = "HIGH" if target == 1 else "LOW"
    return target, prob, risk_level


HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Biased Model Predictor</title>
  <style>
    body { font-family: system-ui; margin: 24px; max-width: 980px; }
    .row { display:flex; gap:16px; flex-wrap:wrap; }
    .card { border:1px solid #ddd; border-radius:10px; padding:16px; flex:1; min-width: 280px; }
    textarea { width:100%; height:240px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    pre { background:#0b1020; color:#e8eefc; padding:12px; border-radius:10px; overflow:auto; }
    .badge { display:inline-block; padding:6px 10px; border-radius:999px; font-weight:700; }
    .low { background:#e8fff0; color:#0a6b2c; }
    .high { background:#ffecec; color:#a60000; }
    button { padding:8px 12px; cursor:pointer; }
  </style>
</head>
<body>
  <h2>Home Credit Default Prediction (Biased Model)</h2>

  <div class="row">
    <div class="card">
      <h3>Request</h3>
      <p>Paste JSON features below (you can provide only a few keys):</p>
      <textarea id="params">{
  "AMT_INCOME_TOTAL": 120000,
  "EXT_SOURCE_1": 0.55,
  "NAME_INCOME_TYPE": "Working"
}</textarea>
      <button onclick="send()">Predict</button>
    </div>

    <div class="card">
      <h3>Response</h3>
      <div>Risk: <span id="risk" class="badge low">—</span></div>
      <pre id="out">{}</pre>
    </div>
  </div>

<script>
async function send() {
  const paramsText = document.getElementById("params").value;

  let params;
  try { params = JSON.parse(paramsText); }
  catch(e) {
    document.getElementById("out").textContent = "Invalid JSON:\\n" + e;
    return;
  }

  const res = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({params})
  });

  const text = await res.text();
  let data;
  try { data = JSON.parse(text); }
  catch(e) {
    document.getElementById("out").textContent =
      "Server did not return JSON. Status: " + res.status + "\\n\\nRaw:\\n" + text;
    return;
  }

  document.getElementById("out").textContent = JSON.stringify(data, null, 2);

  const badge = document.getElementById("risk");
  const lvl = data.risk_level || "—";
  badge.textContent = lvl;
  badge.className = "badge " + (lvl === "HIGH" ? "high" : "low");
}
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(HTML)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict_api():
    try:
        body = request.get_json(force=True)
        params = body.get("params", {})
        if not isinstance(params, dict):
            return jsonify({"error": "params must be a JSON object"}), 400

        target, prob, risk = predict_from_params(params)
        return jsonify({"TARGET": target, "risk_level": risk, "probability": prob})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # change port if you want
    app.run(host="0.0.0.0", port=7000, debug=False)
