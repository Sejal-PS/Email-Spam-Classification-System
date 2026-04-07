# Spam Classifier

This repository contains a simple spam classification script implemented in Python.

## Files

- `spam_classifier.py`: Main script that trains a spam classifier, evaluates models, saves the best model, and supports command-line prediction.
- `requirements.txt`: Python dependencies needed to run the script.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the script to train the models and save the best classifier:

```bash
python spam_classifier.py
```

Run the script with a sample prediction:

```bash
python spam_classifier.py --sample
```

Predict a custom text string:

```bash
python spam_classifier.py --predict "Congratulations! You have won a prize."
```

Skip saving the trained model and vectorizer:

```bash
python spam_classifier.py --no-save
```

## Notes

- The script uses `TfidfVectorizer` with a small sample dataset.
- It trains two models: `MultinomialNB` and `LogisticRegression`, then selects and saves the best-performing model.
- The model artifacts are saved as `model.pkl` and `vectorizer.pkl`.
