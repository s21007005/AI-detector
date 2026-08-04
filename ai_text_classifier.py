"""
ai_text_classifier.py

Classifies a piece of text as AI-generated or human-written.

Approach
--------
TF-IDF features + a choice of ComplementNB or LinearSVC (both used in the
dissertation's dual-classification tool), trained on a labelled CSV of
(text, label) pairs where label is "ai" or "human".

If no trained model exists yet, the script will train one from a CSV you
supply. Once trained, the model is saved to disk (joblib) so future runs
just load it and classify.

Usage
-----
1. Train (or retrain) a model from a labelled dataset:
   python ai_text_classifier.py train --data dataset.csv --model svm

   dataset.csv must have two columns: "text" and "label"
   label values must be "ai" or "human" (case-insensitive)

2. Classify text from the command line:
   python ai_text_classifier.py predict --text "Some text to check..."

3. Classify text from a file:
   python ai_text_classifier.py predict --file some_document.txt

4. Interactive mode (no args):
   python ai_text_classifier.py

Dependencies
------------
pip install scikit-learn joblib pandas --break-system-packages
"""

import argparse
import sys
import os
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = "ai_text_classifier.joblib"


def clean_text(text: str) -> str:
    """Light normalisation before vectorising."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def build_pipeline(model_choice: str) -> Pipeline:
    """
    Build the TF-IDF + classifier pipeline.

    model_choice:
        "nb"  -> ComplementNB (fast, handles class imbalance well,
                 matches the Naive Bayes approach used in the dissertation)
        "svm" -> LinearSVC wrapped in CalibratedClassifierCV so it can
                 output probabilities as well as labels
    """
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
    )

    if model_choice == "nb":
        classifier = ComplementNB()
    elif model_choice == "svm":
        classifier = CalibratedClassifierCV(LinearSVC(), cv=3)
    else:
        raise ValueError("model_choice must be 'nb' or 'svm'")

    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", classifier),
    ])


def train(data_path: str, model_choice: str, text_col: str = "text", label_col: str = "label"):
    if not os.path.exists(data_path):
        sys.exit(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    if text_col not in df.columns or label_col not in df.columns:
        sys.exit(f"CSV must contain '{text_col}' and '{label_col}' columns. Found: {list(df.columns)}")

    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str).apply(clean_text)
    df[label_col] = df[label_col].astype(str).str.lower().str.strip()

    valid_labels = {"ai", "human"}
    if not set(df[label_col].unique()).issubset(valid_labels):
        sys.exit(f"Labels must be 'ai' or 'human'. Found: {df[label_col].unique()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df[text_col], df[label_col], test_size=0.2, random_state=42, stratify=df[label_col]
    )

    pipeline = build_pipeline(model_choice)
    print(f"Training {model_choice.upper()} model on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("\nHeld-out evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        sys.exit(
            "No trained model found. Train one first:\n"
            "  python ai_text_classifier.py train --data dataset.csv --model svm"
        )
    return joblib.load(MODEL_PATH)


def predict(text: str, pipeline: Pipeline = None):
    pipeline = pipeline or load_model()
    text = clean_text(text)

    label = pipeline.predict([text])[0]
    result = {"label": label}

    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba([text])[0]
        classes = pipeline.classes_
        result["confidence"] = {
            cls: f"{p * 100:.2f}%" for cls, p in zip(classes, proba)
        }

    return result


def interactive_mode():
    pipeline = load_model()
    print("AI-text classifier — paste text and press Enter (Ctrl+C to quit).\n")
    while True:
        try:
            text = input("> ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not text.strip():
            continue
        result = predict(text, pipeline)
        print(f"Prediction: {result['label'].upper()}")
        if "confidence" in result:
            print(f"Confidence: {result['confidence']}\n")


def main():
    parser = argparse.ArgumentParser(description="Classify text as AI-generated or human-written.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train a model from a labelled CSV")
    train_parser.add_argument("--data", required=True, help="Path to CSV with 'text' and 'label' columns")
    train_parser.add_argument("--model", choices=["nb", "svm"], default="svm", help="Classifier to use")

    predict_parser = subparsers.add_parser("predict", help="Classify a piece of text")
    predict_parser.add_argument("--text", help="Text to classify")
    predict_parser.add_argument("--file", help="Path to a text file to classify")

    args = parser.parse_args()

    if args.command == "train":
        train(args.data, args.model)
    elif args.command == "predict":
        if args.file:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        elif args.text:
            text = args.text
        else:
            sys.exit("Provide --text or --file")
        result = predict(text)
        print(f"Prediction: {result['label'].upper()}")
        if "confidence" in result:
            print(f"Confidence: {result['confidence']}")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
