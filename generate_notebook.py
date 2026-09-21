import json

cells = []

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [s + "\n" for s in source.strip().split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [s + "\n" for s in source.strip().split("\n")]
    }

# Title & Overview
cells.append(md_cell("""# Spam SMS Detection: End-to-End Machine Learning Pipeline

A disciplined, phase-by-phase implementation of an SMS Spam Classifier enforcing:
- **Zero test-set leakage**: Deduplication prior to splitting; the test set is touched exactly once at the end.
- **Stratified 5-Fold Cross-Validation**: All model comparisons and threshold tuning use out-of-fold training data.
- **Custom Text Normalization**: Domain-specific tokenization (`urltoken`, `moneytoken`, `longnumtoken`) preserving signal while mitigating vocabulary explosion.
- **Precision-Constrained Decision Threshold**: Enforcing Precision >= 0.99 to protect legitimate messages from false spam classification.
- **Statistical Rigor**: 95% Bootstrap Confidence Intervals for generalization metrics.
"""))

# Phase 0 & Setup
cells.append(md_cell("""## Phase 0: Setup & Imports"""))
cells.append(code_cell("""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, cross_val_predict, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    f1_score
)

# Custom feature module from src/features.py
from src.features import normalize, extra_features
"""))

# Phase 1: Load and Clean
cells.append(md_cell("""## Phase 1: Load and Clean Dataset
Kaggle's `spam.csv` is latin-1 encoded with junk auxiliary columns. We load the first two columns, map labels (`ham` -> 0, `spam` -> 1), inspect duplicates, and deduplicate before splitting to prevent train-test data leakage.
"""))
cells.append(code_cell("""df = pd.read_csv("data/spam.csv", encoding="latin-1").iloc[:, :2]
df.columns = ["label", "text"]
df["label"] = df["label"].map({"ham": 0, "spam": 1})

print("Raw dataset shape:", df.shape)
print("Missing values:", df.isna().sum().to_dict())
print("Duplicate messages:", df.duplicated("text").sum())
print("Conflicting labels:", (df.groupby("text")["label"].nunique() > 1).sum())

# Deduplicate
df = df.drop_duplicates("text").reset_index(drop=True)
print("\\nShape after deduplication:", df.shape)
print("Normalized class distribution:")
print(df["label"].value_counts(normalize=True).round(4))
"""))

# Phase 2: EDA
cells.append(md_cell("""## Phase 2: Exploratory Data Analysis (EDA)
Analyzing message length, digit counts, and top n-grams for both spam and ham.
"""))
cells.append(code_cell("""df["length"] = df["text"].str.len()
df["n_digits"] = df["text"].str.count(r"\\d")

print(df.groupby("label")[["length", "n_digits"]].describe().T)

fig, ax = plt.subplots(figsize=(7, 4))
df.boxplot(column="length", by="label", ax=ax)
plt.title("Message Length Distribution by Class")
plt.suptitle("")
plt.xlabel("Label (0: Ham, 1: Spam)")
plt.ylabel("Character Length")
plt.show()

def top_words(texts, n=15):
    cv = CountVectorizer(stop_words="english")
    counts = cv.fit_transform(texts).sum(axis=0).A1
    return pd.Series(counts, index=cv.get_feature_names_out()).nlargest(n)

print("Top 15 words in Spam:")
print(top_words(df.loc[df.label == 1, "text"]))
print("\\nTop 15 words in Ham:")
print(top_words(df.loc[df.label == 0, "text"]))
"""))

# Phase 3: Train / Test Split
cells.append(md_cell("""## Phase 3: Train / Test Split
Stratified 80/20 train/test split. `X_test` will remain untouched until Phase 8.
"""))
cells.append(code_cell("""X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2, stratify=df["label"], random_state=42,
)
print(f"Training set: {len(X_train)} messages ({y_train.sum()} spam, {len(y_train) - y_train.sum()} ham)")
print(f"Test set:     {len(X_test)} messages ({y_test.sum()} spam, {len(y_test) - y_test.sum()} ham)")
"""))

# Phase 4: Text Normalization
cells.append(md_cell("""## Phase 4: Text Normalization
`src/features.py` maps phone numbers/shortcodes to `longnumtoken`, currency amounts to `moneytoken`, and URLs to `urltoken`.
"""))
cells.append(code_cell("""sample = "URGENT! Claim your £1000 prize or visit http://claim.com now! Call 08712400602"
print("Original text:  ", sample)
print("Normalized text:", normalize(sample))
"""))

# Phase 5: Baselines
cells.append(md_cell("""## Phase 5: Baselines with 5-Fold Cross-Validation"""))
cells.append(code_cell("""def make_pipeline_model(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize)),
        ("clf", clf),
    ])

baselines = {
    "Dummy": make_pipeline_model(DummyClassifier(strategy="most_frequent")),
    "NaiveBayes": make_pipeline_model(MultinomialNB()),
    "LogReg": make_pipeline_model(LogisticRegression(max_iter=2000, random_state=42)),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["precision", "recall", "f1", "average_precision"]

rows = {}
for name, model in baselines.items():
    r = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
    rows[name] = {k[5:]: v.mean() for k, v in r.items() if k.startswith("test_")}

pd.DataFrame(rows).T.round(3)
"""))

# Phase 6: Error Analysis
cells.append(md_cell("""## Phase 6: Out-of-Fold Error Analysis on Training Data"""))
cells.append(code_cell("""pipe_lr = baselines["LogReg"]
oof = cross_val_predict(pipe_lr, X_train, y_train, cv=cv)

errs = pd.DataFrame({"text": X_train.values, "true": y_train.values, "pred": oof})
false_pos = errs[(errs.true == 0) & (errs.pred == 1)]
false_neg = errs[(errs.true == 1) & (errs.pred == 0)]

print(f"False Positives (Ham flagged as Spam): {len(false_pos)}")
print(false_pos.head(6))
print(f"\\nFalse Negatives (Spam that slipped through): {len(false_neg)}")
print(false_neg.head(6))

# Model coefficients
pipe_lr.fit(X_train, y_train)
words = np.array(pipe_lr.named_steps["tfidf"].get_feature_names_out())
coef = pipe_lr.named_steps["clf"].coef_[0]
print("\\nTop 15 Spam Features (Largest Positive Weights):")
print(words[np.argsort(coef)[-15:]][::-1])
print("\\nTop 15 Ham Features (Largest Negative Weights):")
print(words[np.argsort(coef)[:15]])
"""))

# Phase 7: Hyperparameter Tuning
cells.append(md_cell("""## Phase 7: Hyperparameter Tuning via GridSearchCV"""))
cells.append(code_cell("""tfidf_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__min_df": [1, 2],
    "tfidf__stop_words": [None, "english"],
}
param_grid = [
    {
        "clf": [MultinomialNB()],
        "clf__alpha": [0.01, 0.05, 0.1, 0.5, 1.0],
        **tfidf_grid
    },
    {
        "clf": [LogisticRegression(max_iter=2000, random_state=42)],
        "clf__C": [0.3, 1, 3, 10, 30],
        "clf__class_weight": [None, "balanced"],
        **tfidf_grid
    },
]

gs = GridSearchCV(make_pipeline_model(MultinomialNB()), param_grid,
                  scoring="f1", cv=cv, n_jobs=-1)
gs.fit(X_train, y_train)

print("Best Parameters:", gs.best_params_)
print(f"Best CV F1: {gs.best_score_:.4f}")
best = gs.best_estimator_
"""))

# Phase 8: Decision Threshold & One-time Test Evaluation
cells.append(md_cell("""## Phase 8: Threshold Calibration & One-Time Test Evaluation
We enforce a business rule: Precision >= 0.99 to strictly avoid blocking legitimate messages.
"""))
cells.append(code_cell("""oof_scores = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof_scores)

ok = prec[:-1] >= 0.99
threshold = thr[ok][np.argmax(rec[:-1][ok])] if ok.any() else 0.5
print(f"Calibrated Threshold (Precision >= 0.99): {threshold:.4f}")

# Single test evaluation
best.fit(X_train, y_train)
proba = best.predict_proba(X_test)[:, 1]

for label, t in [("Default 0.5", 0.5), ("Tuned Threshold", threshold)]:
    pred = (proba >= t).astype(int)
    print(f"\\n--- {label} (t={t:.3f}) ---")
    print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=3))
    print("Confusion Matrix:\\n", confusion_matrix(y_test, pred))

# 95% Bootstrap CI
rng = np.random.default_rng(42)
y_arr = y_test.values
p_arr = (proba >= threshold).astype(int)
boots = [
    f1_score(y_arr[idx], p_arr[idx])
    for idx in (rng.integers(0, len(y_arr), len(y_arr)) for _ in range(1000))
]
print(f"\\nTest F1 95% Bootstrap CI: {np.percentile(boots, [2.5, 97.5]).round(3)}")

fig, ax = plt.subplots(figsize=(6, 4))
PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax)
plt.title("Test Set Precision-Recall Curve")
plt.show()
"""))

# Phase 9: Feature Engineering Experiment
cells.append(md_cell("""## Phase 9: Feature Engineering Experiment (FeatureUnion)"""))
cells.append(code_cell("""features = FeatureUnion([
    ("tfidf", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2)),
    ("extra", make_pipeline(FunctionTransformer(extra_features), StandardScaler())),
])

enriched = Pipeline([
    ("features", features),
    ("clf", LogisticRegression(C=10, max_iter=2000, class_weight="balanced", random_state=42))
])

plain = Pipeline([
    ("features", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(C=10, max_iter=2000, class_weight="balanced", random_state=42))
])

for name, m in [("TF-IDF only", plain), ("TF-IDF + extras", enriched)]:
    s = cross_val_score(m, X_train, y_train, cv=cv, scoring="f1")
    print(f"{name}: {s.mean():.4f} +/- {s.std():.4f}")
"""))

# Phase 10: Model Serialization & Prediction Interface
cells.append(md_cell("""## Phase 10: Model Serialization and Inference Interface"""))
cells.append(code_cell("""joblib.dump({"model": best, "threshold": float(threshold)}, "models/spam_clf.joblib")
print("Saved pipeline and threshold to models/spam_clf.joblib")

bundle = joblib.load("models/spam_clf.joblib")

def predict_spam(messages):
    p = bundle["model"].predict_proba(messages)[:, 1]
    return [
        {"text": m, "spam_prob": round(float(s), 4), "is_spam": bool(s >= bundle["threshold"])}
        for m, s in zip(messages, p)
    ]

sample_texts = [
    "WINNER!! You have been selected for a £900 prize. Call 09061701461 now",
    "hey are we still meeting at 6?",
    "Your parcel is waiting, confirm details at http://bit.ly/x1",
    "Can you send me the lecture notes when you get home?",
    "URGENT! We noticed an unauthorized login attempt to your bank account. Reply YES to verify."
]

for item in predict_spam(sample_texts):
    label = "SPAM" if item["is_spam"] else "HAM"
    print(f"[{label}] (p={item['spam_prob']:.4f}) : {item['text']}")
"""))

notebook_data = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook_data, f, indent=2)

print("notebook.ipynb created successfully via pure JSON.")
