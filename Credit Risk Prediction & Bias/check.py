import joblib

bundle = joblib.load("biased_bundle.joblib")
print(type(bundle))

if isinstance(bundle, dict):
    print(bundle.keys())
