import pickle
import sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

with open("mental_health_model.pkl", "rb") as f:
    model = pickle.load(f)

coping_mechanisms = {
    "Anxiety": "Try deep breathing, meditation, or journaling. Consider speaking to a therapist if it persists.",
    "Depression": "Engage in physical activities, maintain a routine, and talk to loved ones. Seek professional help if needed.",
    "Stress": "Practice mindfulness, take regular breaks, and ensure a healthy work-life balance.",
    "Neutral": "You're doing fine! Stay mindful and take care of your mental well-being."
}

def predict_mental_health(statement):
    prediction = model.predict([statement])[0]
    coping_advice = coping_mechanisms.get(prediction, "Please consult a professional for personalized guidance.")
    return prediction, coping_advice

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
        condition, advice = predict_mental_health(user_input)
        print(f"Predicted Mental Health Condition: {condition}")
        print(f"Suggested Coping Mechanism: {advice}")
    else:
        print("Usage: python predict_mental_health.py '<your statement here>'")
