import argparse
import pickle
import re
from pathlib import Path

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


MODEL_PATH = Path("model.pkl")
VECTORIZER_PATH = Path("vectorizer.pkl")


def ensure_nltk_data():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")


def get_stopwords():
    ensure_nltk_data()
    return set(stopwords.words("english"))


STOPWORDS = get_stopwords()
STEMMER = PorterStemmer()


def create_dataset():
    return pd.DataFrame(
        {
            "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "message": [
                "congratulations ! You won a free lottery ticket",
                "Hey, are we meeting today?",
                "Claim your free prize now",
                "Let's have lunch tomorrow",
                "You have been selected for a cash reward",
                "Can you send me the report?",
                "Win money now!!! Click here",
                "Project meeting is scheduled at 5 PM",
                "Get free coupons now",
                "See you at the event",
            ],
        }
    )


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text: str) -> str:
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOPWORDS]
    tokens = [STEMMER.stem(word) for word in tokens]
    return " ".join(tokens)


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cleaned"] = df["message"].apply(clean_text)
    df["processed"] = df["cleaned"].apply(preprocess_text)
    return df


def train_models(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_train, y_train)

    models = {"MultinomialNB": mnb, "LogisticRegression": lr}
    return x_train, x_test, y_train, y_test, models


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0
        ),
    }


def save_artifacts(model, vectorizer, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)


def load_artifacts(model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


def predict_email(text: str, vectorizer, model) -> str:
    processed = preprocess_text(clean_text(text))
    features = vectorizer.transform([processed])
    label = model.predict(features)[0]
    return "Spam" if label == 1 else "Not Spam"


def print_metrics(name: str, metrics: dict):
    print(f"\n{name} results:")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 score: {metrics['f1']:.4f}")
    print("\nClassification report:")
    print(metrics["classification_report"])


def parse_args():
    parser = argparse.ArgumentParser(description="Train and use a simple spam classifier.")
    parser.add_argument(
        "--predict",
        type=str,
        help="Predict whether the provided text is spam.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Show a sample prediction after training.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the trained model and vectorizer to disk.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    df = create_dataset()
    print("Dataset shape:", df.shape)
    print(df.head())

    prepared_df = prepare_data(df)
    print("\nSample cleaned and processed text:")
    print(prepared_df[["message", "cleaned", "processed"]].head())

    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(prepared_df["processed"])
    y = prepared_df["label"]

    x_train, x_test, y_train, y_test, models = train_models(x, y)

    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, x_test, y_test)

    print("\nModel evaluation:")
    for name, metrics in results.items():
        print_metrics(name, metrics)

    best_model_name = max(results, key=lambda n: results[n]["accuracy"])
    best_model = models[best_model_name]
    print(f"\nSelected best model: {best_model_name}")

    if not args.no_save:
        save_artifacts(best_model, vectorizer)
        print(f"\nSaved model to {MODEL_PATH} and vectorizer to {VECTORIZER_PATH}")

    if args.sample:
        sample_text = "Congratulations, you have won cash rewards! Click to claim now."
        prediction = predict_email(sample_text, vectorizer, best_model)
        print(f"\nSample prediction: {prediction}")

    if args.predict:
        prediction = predict_email(args.predict, vectorizer, best_model)
        print(f"\nPrediction for provided text: {prediction}")


if __name__ == "__main__":
    main()



