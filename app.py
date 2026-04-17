
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re

# Page config
st.set_page_config(
    page_title="Mental Health Crisis Detector",
    page_icon="🧠",
    layout="centered"
)

# Load models
@st.cache_resource
def load_models():
    BASE = "/kaggle/input/notebooks/vidhi0405/mental-health-nlp-data-collection"
    with open(f"{BASE}/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(f"{BASE}/lr_model.pkl", "rb") as f:
        lr = pickle.load(f)
    with open(f"{BASE}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return tfidf, lr, le

tfidf, lr, le = load_models()

# Label colours
COLORS = {
    "Normal":               "#1D9E75",
    "Anxiety":              "#185FA5",
    "Depression":           "#7F77DD",
    "Suicidal":             "#D85A30",
    "Stress":               "#BA7517",
    "Bipolar":              "#D4537E",
    "Personality disorder": "#888780"
}

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s\.\!\?\,\']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Header
st.title("🧠 Mental Health Crisis Detector")
st.markdown("This tool analyses text and predicts the mental health category using NLP.")
st.warning("⚠️ This tool is for demonstration purposes only - not a clinical or diagnostic tool.")
st.markdown("---")

# Input
st.subheader("Enter your text below")
user_input = st.text_area(
    label="Type or paste any text here",
    placeholder="e.g. I have been feeling really low lately and can't seem to find joy in anything...",
    height=150
)

if st.button("Analyse Text", type="primary"):
    if not user_input.strip():
        st.error("Please enter some text first.")
    else:
        cleaned = clean_text(user_input)
        vec     = tfidf.transform([cleaned])
        pred    = lr.predict(vec)[0]
        proba   = lr.predict_proba(vec)[0]
        label   = le.inverse_transform([pred])[0]
        
        st.markdown("---")
        st.subheader("Result")
        
        color = COLORS.get(label, "#888780")
        st.markdown(
            f"<div style=\"background:{color}22; border-left: 4px solid {color}; padding: 16px; border-radius: 8px;\">"
            f"<h3 style=\"color:{color}; margin:0\">Predicted Category: {label}</h3>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        st.markdown("###")
        
        # Confidence chart
        st.subheader("Confidence scores")
        labels      = le.classes_
        colors_list = [COLORS.get(l, "#888780") for l in labels]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(labels, proba * 100, color=colors_list, edgecolor="none")
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, 100)
        for bar, prob in zip(bars, proba):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f"{prob*100:.1f}%", va="center", fontsize=10)
        ax.set_title("Model confidence by category")
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        if label == "Suicidal":
            st.error("🆘 If you or someone you know is in crisis, please contact the Samaritans: 116 123 (free, 24/7)")
        elif label in ["Depression", "Anxiety", "Bipolar", "Personality disorder"]:
            st.info("💙 If you are struggling, please consider speaking to a GP or mental health professional.")

st.markdown("---")
st.markdown("**About this project** - Built using Python, scikit-learn, and Streamlit. Dataset: Sentiment Analysis for Mental Health (Kaggle). Model: TF-IDF + Logistic Regression (76.12% accuracy).")
