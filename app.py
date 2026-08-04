"""
app.py — Streamlit front-end for phishing_email_classifier.py

Run locally:
    streamlit run app.py

Repo layout expected:
    ├── app.py                      (this file)
    ├── phishing_email_classifier.py
    ├── requirements.txt
    └── phishing_email_classifier.joblib   (optional — created by training)

Behaviour
---------
- If a trained model (phishing_email_classifier.joblib) is present, the app
  uses it for ML predictions (TF-IDF + ComplementNB/LinearSVC).
- If no trained model exists yet, the app falls back to a transparent
  heuristic risk score (based on the same engineered signals) so it's still
  usable out of the box, and offers a sidebar option to train a model from
  an uploaded CSV.
"""

import os
import streamlit as st
import pandas as pd

import phishing_email_classifier as pec

st.set_page_config(page_title="Phishing Email Classifier", page_icon="🎣", layout="centered")

HEURISTIC_WEIGHTS = {
    "num_urls": 4,
    "num_ip_urls": 20,
    "urgency_count": 10,
    "generic_greeting": 12,
    "free_email_sender": 8,
    "domain_mismatch": 20,
    "exclamations": 3,
    "uppercase_ratio": 15,
}


def heuristic_risk_score(signals: dict) -> float:
    """Rough 0-100 risk score used only when no trained ML model is available."""
    score = 0.0
    score += min(signals["num_urls"], 5) * HEURISTIC_WEIGHTS["num_urls"]
    score += signals["num_ip_urls"] * HEURISTIC_WEIGHTS["num_ip_urls"]
    score += min(signals["urgency_count"], 5) * HEURISTIC_WEIGHTS["urgency_count"]
    score += signals["generic_greeting"] * HEURISTIC_WEIGHTS["generic_greeting"]
    score += signals["free_email_sender"] * HEURISTIC_WEIGHTS["free_email_sender"]
    score += signals["domain_mismatch"] * HEURISTIC_WEIGHTS["domain_mismatch"]
    score += min(signals["exclamations"], 5) * HEURISTIC_WEIGHTS["exclamations"]
    score += signals["uppercase_ratio"] * HEURISTIC_WEIGHTS["uppercase_ratio"]
    return min(round(score, 1), 100.0)


@st.cache_resource(show_spinner=False)
def get_model():
    if os.path.exists(pec.MODEL_PATH):
        return pec.load_model()
    return None


def train_from_upload(uploaded_file, model_choice: str):
    tmp_path = "uploaded_training_data.csv"
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    with st.spinner(f"Training {model_choice.upper()} model..."):
        pec.train(tmp_path, model_choice)
    st.cache_resource.clear()
    os.remove(tmp_path)


# ---------------- Sidebar ----------------
st.sidebar.header("Model")
model = get_model()

if model is not None:
    st.sidebar.success("Trained ML model loaded ✅")
else:
    st.sidebar.warning("No trained model found — using heuristic scoring only.")

with st.sidebar.expander("Train / retrain model"):
    st.caption("CSV must have columns: sender, subject, body, label ('phishing' or 'legitimate')")
    model_choice = st.selectbox("Classifier", ["svm", "nb"], format_func=lambda x: "SVM" if x == "svm" else "Naive Bayes")
    uploaded_csv = st.file_uploader("Upload training CSV", type=["csv"])
    if uploaded_csv is not None and st.button("Train model"):
        train_from_upload(uploaded_csv, model_choice)
        st.rerun()

# ---------------- Main UI ----------------
st.title("🎣 Phishing Email Classifier")
st.write("Enter the details of an email below to check whether it looks like phishing.")

with st.form("email_form"):
    sender = st.text_input("Sender address", placeholder="e.g. support@amaz0n-security.com")
    subject = st.text_input("Subject", placeholder="e.g. Urgent: Verify your account now!")
    body = st.text_area("Body", height=220, placeholder="Paste the email body here...")
    submitted = st.form_submit_button("Analyse email", use_container_width=True)

if submitted:
    if not (sender or subject or body):
        st.error("Please fill in at least one field.")
    else:
        signals = pec.extract_features(sender, subject, body)
        signals.pop("text", None)

        if model is not None:
            result = pec.predict(sender, subject, body, model)
            label = result["label"]
            confidence = result.get("confidence", {})

            if label == "phishing":
                st.error(f"⚠️ Prediction: **PHISHING** ")
            else:
                st.success(f"✅ Prediction: **LEGITIMATE**")

            if confidence:
                st.subheader("Confidence")
                conf_df = pd.DataFrame({
                    "Class": list(confidence.keys()),
                    "Confidence": [float(v.strip("%")) for v in confidence.values()],
                })
                st.bar_chart(conf_df.set_index("Class"))
        else:
            risk = heuristic_risk_score(signals)
            if risk >= 50:
                st.error(f"⚠️ Heuristic risk score: **{risk}/100** — looks suspicious")
            else:
                st.success(f"✅ Heuristic risk score: **{risk}/100** — looks lower-risk")
            st.caption("This is a rule-based estimate. Train a model in the sidebar for ML-based predictions.")

        st.subheader("Signals detected")
        signal_df = pd.DataFrame(signals.items(), columns=["Signal", "Value"])
        st.dataframe(signal_df, use_container_width=True, hide_index=True)
