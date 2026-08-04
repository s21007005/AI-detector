"""
app.py — Streamlit front-end for ai_text_classifier.py

Run locally:
    streamlit run app.py

Repo layout expected:
    ├── app.py                     (this file — set as the Main file path on Streamlit Cloud)
    ├── ai_text_classifier.py
    ├── sample_training_data.csv   (small bundled demo dataset)
    ├── requirements.txt
    └── ai_text_classifier.joblib  (optional — created by training)

Behaviour
---------
- If a trained model (ai_text_classifier.joblib) is present, the app uses it.
- If not, and sample_training_data.csv is present in the repo, the app
  auto-trains a starter model from it on first load (small demo dataset —
  fine for testing the app, not a substitute for a proper research dataset).
- You can always replace it with a bigger/better dataset via the sidebar
  upload, which retrains and overrides the demo model.
"""

import os
import streamlit as st
import pandas as pd

import ai_text_classifier as atc

st.set_page_config(page_title="AI Text Detector", page_icon="🤖", layout="centered")

SAMPLE_DATA_PATH = "sample_training_data.csv"


@st.cache_resource(show_spinner=False)
def get_model():
    """Returns (model, source) where source is 'trained', 'demo', or None."""
    if os.path.exists(atc.MODEL_PATH):
        return atc.load_model(), "trained"
    if os.path.exists(SAMPLE_DATA_PATH):
        with st.spinner("First run — training a starter model from the bundled sample dataset..."):
            atc.train(SAMPLE_DATA_PATH, "svm")
        return atc.load_model(), "demo"
    return None, None


def train_from_upload(uploaded_file, model_choice: str):
    tmp_path = "uploaded_training_data.csv"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner(f"Training {model_choice.upper()} model..."):
        atc.train(tmp_path, model_choice)
    st.cache_resource.clear()
    os.remove(tmp_path)


# ---------------- Sidebar ----------------
st.sidebar.header("Model")
model, model_source = get_model()

if model_source == "trained":
    st.sidebar.success("Trained ML model loaded ✅")
elif model_source == "demo":
    st.sidebar.info("Running on the small bundled demo dataset ℹ️")
    st.sidebar.caption("Fine for testing the app — upload a larger, proper dataset below for real accuracy.")
else:
    st.sidebar.warning("No trained model found yet.")

with st.sidebar.expander("Train / retrain model"):
    st.caption("CSV must have columns: text, label ('ai' or 'human')")
    model_choice = st.selectbox("Classifier", ["svm", "nb"], format_func=lambda x: "SVM" if x == "svm" else "Naive Bayes")
    uploaded_csv = st.file_uploader("Upload training CSV", type=["csv"])
    if uploaded_csv is not None and st.button("Train model"):
        train_from_upload(uploaded_csv, model_choice)
        st.rerun()

# ---------------- Main UI ----------------
st.title("🤖 AI Text Detector")
st.write("Paste in a piece of text to check whether it looks AI-generated or human-written.")

if model is None:
    st.info("No trained model is loaded yet — train one from the sidebar before analysing text.")

with st.form("text_form"):
    text_input = st.text_area("Text to analyse", height=250, placeholder="Paste text here...")
    submitted = st.form_submit_button("Analyse text", use_container_width=True, disabled=model is None)

if submitted and model is not None:
    if not text_input.strip():
        st.error("Please paste in some text first.")
    else:
        result = atc.predict(text_input, model)
        label = result["label"]
        confidence = result.get("confidence", {})

        if label == "ai":
            st.error("🤖 Prediction: **AI-GENERATED**")
        else:
            st.success("🧑 Prediction: **HUMAN-WRITTEN**")

        if confidence:
            st.subheader("Confidence")
            conf_df = pd.DataFrame({
                "Class": list(confidence.keys()),
                "Confidence": [float(v.strip("%")) for v in confidence.values()],
            })
            st.bar_chart(conf_df.set_index("Class"))
