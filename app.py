import streamlit as st
import os
import pickle
import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(
    page_title="Mental Health Companion",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Constants
MODEL_PATH = "mental_health_model.pkl"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    [data-testid="stSidebar"] {
        background-color: rgba(30, 41, 59, 0.7) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    .emergency-card {
        background: rgba(220, 38, 38, 0.2);
        border-left: 4px solid #ef4444;
        padding: 15px;
        margin-bottom: 20px;
        border-radius: 8px;
        color: #f8fafc;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        background-color: #3b82f6;
        color: white;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
    }
    /* Glassmorphism for tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD MODEL ---
@st.cache_resource
def load_ml_model():
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
    return None

ml_model = load_ml_model()

# --- HELPER FUNCTIONS ---
def predict_status(statement):
    if ml_model:
        try:
            return ml_model.predict([statement])[0]
        except:
            return "Unknown"
    return "Unknown (Model missing)"

def lime_predictor(texts):
    if ml_model:
        probs = ml_model.predict_proba(texts)
        return np.array(probs)
    return np.array([])

def get_lime_explanation(statement):
    if not ml_model:
        return "<div style='color:red'>Model not loaded. Please train the model first.</div>"
    
    if not statement.strip():
        return "Please enter a statement."

    try:
        class_names = list(ml_model.classes_)
        explainer = LimeTextExplainer(class_names=class_names)
        
        predicted_class = ml_model.predict([statement])[0]
        label_idx = class_names.index(predicted_class)
        
        exp = explainer.explain_instance(
            statement, 
            lime_predictor, 
            num_features=6, 
            labels=[label_idx],
            num_samples=500 
        )
        return exp.as_html()
    except Exception as e:
        # Fallback: Keyword Highlighting
        keywords = {
            "Anxiety": ["anxious", "scared", "future", "panic", "worry", "nervous"],
            "Depression": ["depressed", "sad", "hopeless", "crying", "energy", "devastated"],
            "Stress": ["stressful", "workload", "sleep", "overwhelmed", "burnt out"],
            "Suicidal": ["end", "life", "point", "self harm", "pain"],
            "Bipolar": ["mood", "swings", "high", "energy", "devastated"],
            "Social Anxiety": ["social", "nervous", "shaky", "judging"]
        }
        
        predicted = predict_status(statement)
        relevant_keywords = keywords.get(predicted, [])
        highlighted = statement
        for kw in relevant_keywords:
            highlighted = re.sub(f"({kw})", r"<span style='background-color: #fbbf24; color: black; padding: 2px; border-radius: 4px;'>\1</span>", highlighted, flags=re.IGNORECASE)
        
        fallback_html = f"""
        <div style='padding: 20px; border-radius: 8px; background: rgba(255,255,255,0.05); color: white;'>
            <h4>Visualization Fallback (Predicted: {predicted})</h4>
            <p>We couldn't generate a full LIME report, but here are the key terms in your statement that likely influenced the prediction:</p>
            <p style='font-size: 1.2em; line-height: 1.6;'>{highlighted}</p>
            <small style='color: #94a3b8'>Note: Full LIME reports may require more computational resources.</small>
        </div>
        """
        return fallback_html

def offline_advice(status, message=""):
    physical_keywords = ["cold", "cough", "fever", "pain", "flu", "headache", "sick"]
    if any(kw in message.lower() for kw in physical_keywords):
        return "It sounds like you might be experiencing physical symptoms. I recommend resting, staying hydrated, and consulting a medical doctor or healthcare professional for your physical health."

    advice = {
        "Anxiety": "It sounds like you're feeling anxious. Try the 4-7-8 breathing technique: Inhale for 4s, hold for 7s, exhale for 8s.",
        "Depression": "I'm sorry you're feeling this way. Remember that you're not alone. Try to reach out to one person you trust today.",
        "Stress": "Stress can be overwhelming. Try to break your tasks into tiny, manageable steps and take a 5-minute break.",
        "Suicidal": "Please reach out for help immediately. You are valuable. Call a crisis hotline (e.g., 988 in the US/Canada).",
        "Normal": "It's good to hear you're doing okay! Practicing gratitude can help maintain this positive state.",
        "Bipolar": "Managing mood swings can be tough. Keeping a mood journal might help you identify patterns to share with a professional.",
        "Social Anxiety": "Social situations can be draining. Try to focus on one person at a time and remember to be kind to yourself."
    }
    return advice.get(status, "I'm here to listen and support you. How can I help further?")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🏥 Quick Resources")
    st.markdown("""
        <div class='emergency-card'>
            <strong>🚨 Crisis Support</strong><br>
            988 Suicide & Crisis Lifeline<br>
            Text 'HOME' to 741741
        </div>
    """, unsafe_allow_html=True)
    
    api_key_input = st.text_input("Groq API Key (Optional)", type="password", placeholder="gsk_...")
    st.info("The app will use the pre-configured key if this is left blank.")
    
    st.divider()
    st.markdown("### 💡 Try these examples:")
    if st.button("I feel so anxious"):
        st.session_state.example_input = "I feel so anxious"
    if st.button("I'm overwhelmed with work"):
        st.session_state.example_input = "I'm overwhelmed with work"
    if st.button("Everything is hopeless"):
        st.session_state.example_input = "Everything is hopeless"

# --- MAIN UI ---
st.title("🧠 Mental Health Companion")

tab1, tab2, tab3 = st.tabs(["💬 Support Chat", "📊 Model Insights", "🌿 Self-Care Hub"])

# Support Chat Tab
with tab1:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("How are you feeling today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            api_key = api_key_input.strip() if api_key_input else os.getenv("GROQ_API_KEY")
            status = predict_status(prompt)

            if not api_key or "your_api_key" in api_key:
                full_response = f"**[Offline Mode - Predicted Status: {status}]**\n\n{offline_advice(status, prompt)}\n\n*Note: To enable full AI conversation, please add a Groq API Key.*"
                message_placeholder.markdown(full_response)
            else:
                try:
                    llm = ChatGroq(api_key=api_key, model_name=GROQ_MODEL, temperature=0.3)
                    messages = [
                        SystemMessage(content=f"You are an empathetic clinical psychologist assistant. User status: {status}. If they mention physical health (fever/cold), advise seeing a doctor.")
                    ]
                    for msg in st.session_state.messages:
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        else:
                            messages.append(AIMessage(content=msg["content"]))
                    
                    for chunk in llm.stream(messages):
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"❌ **Error**: {str(e)}\n\n*Falling back to local advice:*\n\n{offline_advice(status, prompt)}"
                    message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Model Insights Tab
with tab2:
    st.markdown("### Why did the model make this prediction?")
    
    # Initialize example input
    if "example_input" not in st.session_state:
        st.session_state.example_input = ""
        
    explain_input = st.text_input("Statement to analyze", value=st.session_state.example_input, placeholder="I've been feeling...")
    
    if st.button("Analyze Statement", key="analyze_btn") or (explain_input and "last_analyzed" not in st.session_state):
        if explain_input:
            with st.spinner("Generating explanation..."):
                explanation_html = get_lime_explanation(explain_input)
                components.html(explanation_html, height=400, scrolling=True)
                st.session_state.last_analyzed = explain_input

# Self-Care Hub Tab
with tab3:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### ✨ Recommended Activities")
        st.markdown("- **Journaling**: Write down your thoughts.\n- **Nature Walk**: Get some fresh air.\n- **Creative Expression**: Draw, paint, or listen to music.")
    with col2:
        st.markdown("### 📚 Educational Snippets")
        st.markdown("**Anxiety**: The body's natural response to stress.")
        st.markdown("**Depression**: A persistent feeling of sadness or loss of interest.")
