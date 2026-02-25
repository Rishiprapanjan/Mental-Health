import os
import pickle
import pandas as pd
import gradio as gr
import numpy as np
import re
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from lime.lime_text import LimeTextExplainer

# Load environment variables
load_dotenv()

# Constants
MODEL_PATH = "mental_health_model.pkl"
GROQ_MODEL = "llama-3.3-70b-versatile"

# GLOBAL VARIABLES for models
ml_model = None

def load_models():
    global ml_model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as f:
                ml_model = pickle.load(f)
            print(f"Model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Classification will not work.")

load_models()

# --- HELPER FUNCTIONS ---

def predict_status(statement):
    if ml_model:
        try:
            return ml_model.predict([statement])[0]
        except:
            return "Unknown"
    return "Unknown (Model missing)"

# LIME predictor must be at top level for multiprocessing pickling on Windows
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
        
        # Lower num_samples for speed/stability on Windows
        exp = explainer.explain_instance(
            statement, 
            lime_predictor, 
            num_features=6, 
            labels=[label_idx],
            num_samples=500 
        )
        return exp.as_html()
    except Exception as e:
        # Premium Fallback: Keyword Highlighting
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
        <div style='padding: 20px; border-radius: 8px; background: rgba(255,255,255,0.05);'>
            <h4>Visualization Fallback (Predicted: {predicted})</h4>
            <p>We couldn't generate a full LIME report, but here are the key terms in your statement that likely influenced the prediction:</p>
            <p style='font-size: 1.2em; line-height: 1.6;'>{highlighted}</p>
            <small style='color: #94a3b8'>Note: Full LIME reports may require a local environment with more resources.</small>
        </div>
        """
        return fallback_html

def offline_advice(status, message=""):
    # Check for physical symptoms if the prediction is "Normal"
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

def chat_response(message, history, api_key_input):
    api_key = api_key_input.strip() if api_key_input else os.getenv("GROQ_API_KEY")
    status = predict_status(message)

    if not api_key or "your_api_key" in api_key:
        fallback = f"**[Offline Mode - Predicted Status: {status}]**\n\n{offline_advice(status, message)}\n\n*Note: To enable full AI conversation, please add a Groq API Key.*"
        yield fallback
        return

    try:
        llm = ChatGroq(api_key=api_key, model_name=GROQ_MODEL, temperature=0.3)
        messages = [
            SystemMessage(content=f"""
                You are an empathetic, professional clinical psychologist assistant.
                User's predicted mental health status: {status}.
                Guidelines:
                1. Provide validating and supportive responses.
                2. Prioritize safety if self-harm is mentioned.
                3. If the user mentions physical symptoms like fever, cough, or cold, clarify that you are a mental health assistant and recommend they see a medical doctor.
            """)
        ]
        
        # Handle Gradio 6 history (list of ChatMessage objects)
        for msg in history:
            if hasattr(msg, 'role'):
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                elif msg.role == "assistant":
                    messages.append(AIMessage(content=msg.content))
            elif isinstance(msg, dict): # Fallback for dict-style history
                if msg.get("role") == "user":
                    messages.append(HumanMessage(content=msg.get("content", "")))
                elif msg.get("role") == "assistant":
                    messages.append(AIMessage(content=msg.get("content", "")))
            elif isinstance(msg, (list, tuple)) and len(msg) >= 2: # Fallback for old style list of pairs
                messages.append(HumanMessage(content=msg[0]))
                messages.append(AIMessage(content=msg[1]))

        messages.append(HumanMessage(content=message))
        
        partial_message = ""
        for chunk in llm.stream(messages):
            partial_message += chunk.content
            yield partial_message
    except Exception as e:
        yield f"❌ **Error**: {str(e)}\n\n*Falling back to local advice:*\n\n{offline_advice(status, message)}"

# --- CUSTOM CSS ---
custom_css = """
body { font-family: 'Inter', sans-serif; height: 100vh; margin: 0; }
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%) !important;
    color: #f8fafc !important;
}
.glass {
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
.emergency-card {
    background: rgba(220, 38, 38, 0.2);
    border-left: 4px solid #ef4444;
    padding: 15px; margin-bottom: 15px; border-radius: 8px;
}
"""

with gr.Blocks() as demo:
    with gr.Row():
        # Sidebar
        with gr.Column(scale=1, elem_id="sidebar", variant="panel"):
            gr.Markdown("## 🏥 Quick Resources")
            gr.HTML("""
                <div class='emergency-card'>
                    <strong>🚨 Crisis Support</strong><br>
                    988 Suicide & Crisis Lifeline<br>
                    Text 'HOME' to 741741
                </div>
            """)
            api_key_box = gr.Textbox(label="Groq API Key", placeholder="gsk_...", type="password")
            gr.Markdown("---")
            gr.Markdown("### Try these examples:")
            ex1 = gr.Button("I feel so anxious", variant="secondary", size="sm")
            ex2 = gr.Button("I'm overwhelmed with work", variant="secondary", size="sm")
            ex3 = gr.Button("Everything is hopeless", variant="secondary", size="sm")

        # Main Content
        with gr.Column(scale=4):
            gr.Markdown("# 🧠 Mental Health Companion")
            
            with gr.Tabs():
                with gr.TabItem("💬 Support Chat"):
                    chat_interface = gr.ChatInterface(
                        fn=chat_response,
                        chatbot=gr.Chatbot(height=500, label="Your safe space"),
                        additional_inputs=[api_key_box]
                    )
                
                with gr.TabItem("📊 Model Insights (LIME)"):
                    gr.Markdown("### Why did the model make this prediction?")
                    explain_input = gr.Textbox(label="Statement to analyze", placeholder="I've been feeling...")
                    explain_btn = gr.Button("Analyze Statement", variant="primary")
                    explain_output = gr.HTML(label="Explanation Output")
                    
                    explain_btn.click(fn=get_lime_explanation, inputs=explain_input, outputs=explain_output)

                with gr.TabItem("🌿 Self-Care Hub"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### ✨ Recommended Activities\n- **Journaling**\n- **Nature Walk**\n- **Creative Expression**")
                        with gr.Column():
                            gr.Markdown("### 📚 Educational Snippets\n**Anxiety**: The body's natural response to stress.")

    # Example interactions
    def set_example(val): return val
    ex1.click(lambda: "I feel so anxious", None, explain_input).then(get_lime_explanation, explain_input, explain_output)
    ex2.click(lambda: "I'm overwhelmed with work stress", None, explain_input).then(get_lime_explanation, explain_input, explain_output)
    ex3.click(lambda: "Everything feels hopeless and sad", None, explain_input).then(get_lime_explanation, explain_input, explain_output)

if __name__ == "__main__":
    demo.launch(
        css=custom_css,
        show_error=True, 
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        debug=True
    )
