import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Expanded training data for a better "dummy" experience
data = {
    "statement": [
        "I feel very anxious and scared about the future",
        "The panic attacks are getting worse every day",
        "I am so depressed and I don't want to get out of bed",
        "Everything feels hopeless and I am constantly crying",
        "The workload is so stressful and I can't sleep",
        "I am overwhelmed by my responsibilities and feel burnt out",
        "I am feeling quite good today, very peaceful",
        "Just a normal day, nothing special happened",
        "I want to end my life, there is no point anymore",
        "I am thinking about self harm because the pain is too much",
        "I feel like my mood swings are out of control lately",
        "One moment I am high energy, the next I am devastated",
        "Social situations make me extremely nervous and shaky",
        "I feel like everyone is judging me all the time"
    ],
    "status": [
        "Anxiety", "Anxiety",
        "Depression", "Depression",
        "Stress", "Stress",
        "Normal", "Normal",
        "Suicidal", "Suicidal",
        "Bipolar", "Bipolar",
        "Social Anxiety", "Social Anxiety"
    ]
}

df = pd.DataFrame(data)

# Create pipeline with LogisticRegression (supports predict_proba for LIME)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression())
])

# Train
pipeline.fit(df['statement'], df['status'])

# Save
with open("mental_health_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Better dummy model created and saved as mental_health_model.pkl")
