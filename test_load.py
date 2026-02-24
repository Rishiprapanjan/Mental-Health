import pickle
import os

MODEL_PATH = "mental_health_model.pkl"
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            ml_model = pickle.load(f)
        print(f"Model loaded successfully from {MODEL_PATH}")
        print(f"Model type: {type(ml_model)}")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"{MODEL_PATH} not found")
