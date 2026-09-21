"""
End-to-End Spam SMS Detection Pipeline (Phases 1 - 10)
Strictly adheres to train/test isolation, cross-validation, out-of-fold error analysis,
threshold calibration, and model serialization.
"""

import os
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

# Ensure local imports work
sys.path.insert(0, os.path.abspath("."))
from src.features import normalize, extra_features


def main():
    print("=" * 70)
    print("PHASE 1: LOAD AND CLEAN")
    print("=" * 70)
    
    df = pd.read_csv("data/spam.csv", encoding="latin-1").iloc[:, :2]
    df.columns = ["label", "text"]
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    print("Initial shape:", df.shape)
    print("Missing values:", df.isna().sum().to_dict())
    print("Duplicate texts:", df.duplicated("text").sum())
    print("Conflicting labels:", (df.groupby("text")["label"].nunique() > 1).sum())

    # Deduplicate before splitting to prevent train-test data leakage
    df = df.drop_duplicates("text").reset_index(drop=True)
    print("\nShape after deduplication:", df.shape)
    print("Label distribution (normalized):")
    print(df["label"].value_counts(normalize=True).round(4))
    print(f"Total Ham (0): {(df['label'] == 0).sum()}, Total Spam (1): {(df['label'] == 1).sum()}")

    print("\n" + "=" * 70)
    print("PHASE 2: EXPLORATORY ANALYSIS")
    print("=" * 70)

    df["length"] = df["text"].str.len()
    df["n_digits"] = df["text"].str.count(r"\d")

    desc = df.groupby("label")[["length", "n_digits"]].describe().T
    print(desc)

    def top_words(texts, n=15):
        cv = CountVectorizer(stop_words="english")
        counts = cv.fit_transform(texts).sum(axis=0).A1
        return pd.Series(counts, index=cv.get_feature_names_out()).nlargest(n)

    print("\nTop 15 words in SPAM:")
    print(top_words(df.loc[df.label == 1, "text"]))
    print("\nTop 15 words in HAM:")
    print(top_words(df.loc[df.label == 0, "text"]))

    # Save EDA boxplot figure
    os.makedirs("models", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="length", by="label", ax=ax)
    plt.title("Message Length Distribution by Label (0=Ham, 1=Spam)")
    plt.suptitle("")
    plt.xlabel("Label (0: Ham, 1: Spam)")
    plt.ylabel("Character Length")
    plt.savefig("eda_length_boxplot.png", bbox_inches="tight")
    plt.close()
    print("\nSaved eda_length_boxplot.png")

    print("\n" + "=" * 70)
    print("PHASE 3: SPLIT (TRAIN / TEST)")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=42,
    )
    print(f"Training set: {len(X_train)} samples ({y_train.sum()} spam, {len(y_train) - y_train.sum()} ham)")
    print(f"Test set:     {len(X_test)} samples ({y_test.sum()} spam, {len(y_test) - y_test.sum()} ham)")
    print("Test set is locked until Phase 8 evaluation.")

    print("\n" + "=" * 70)
    print("PHASE 4: TEXT NORMALIZATION & TF-IDF PREVIEW")
    print("=" * 70)
    sample_text = "WINNER!! Call 09061701461 to claim your £900 prize or visit http://win.com now!"
    print(f"Raw text:        {sample_text}")
    print(f"Normalized text: {normalize(sample_text)}")

    print("\n" + "=" * 70)
    print("PHASE 5: BASELINES WITH 5-FOLD CROSS-VALIDATION")
    print("=" * 70)

    def make_baseline_pipe(clf):
        return Pipeline([
            ("tfidf", TfidfVectorizer(preprocessor=normalize)),
            ("clf", clf),
        ])

    baselines = {
        "Dummy": make_baseline_pipe(DummyClassifier(strategy="most_frequent")),
        "NaiveBayes": make_baseline_pipe(MultinomialNB()),
        "LogReg": make_baseline_pipe(LogisticRegression(max_iter=2000, random_state=42)),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["precision", "recall", "f1", "average_precision"]

    rows = {}
    for name, model in baselines.items():
        r = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring)
        rows[name] = {k[5:]: v.mean() for k, v in r.items() if k.startswith("test_")}

    cv_results_df = pd.DataFrame(rows).T.round(3)
    print(cv_results_df)

    print("\n" + "=" * 70)
    print("PHASE 6: ERROR ANALYSIS ON TRAINING DATA (OUT-OF-FOLD)")
    print("=" * 70)

    pipe_lr = baselines["LogReg"]
    oof = cross_val_predict(pipe_lr, X_train, y_train, cv=cv)

    errs = pd.DataFrame({"text": X_train.values, "true": y_train.values, "pred": oof})
    false_pos = errs[(errs.true == 0) & (errs.pred == 1)]   # ham flagged as spam
    false_neg = errs[(errs.true == 1) & (errs.pred == 0)]   # spam that got through

    print(f"False Positives (Ham flagged as Spam): {len(false_pos)}")
    print(false_pos.head(6))
    print(f"\nFalse Negatives (Spam that slipped through): {len(false_neg)}")
    print(false_neg.head(6))

    # Feature importances / coefficients
    pipe_lr.fit(X_train, y_train)
    words = np.array(pipe_lr.named_steps["tfidf"].get_feature_names_out())
    coef = pipe_lr.named_steps["clf"].coef_[0]
    print("\nTop 15 Most Spammy features (positive weights):")
    print(words[np.argsort(coef)[-15:]][::-1])
    print("\nTop 15 Most Hammy features (negative weights):")
    print(words[np.argsort(coef)[:15]])

    print("\n" + "=" * 70)
    print("PHASE 7: HYPERPARAMETER TUNING")
    print("=" * 70)

    tfidf_grid = {
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

    base_estimator = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalize)),
        ("clf", MultinomialNB()),
    ])

    gs = GridSearchCV(base_estimator, param_grid, scoring="f1", cv=cv, n_jobs=-1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    print(f"Best CV F1: {gs.best_score_:.4f}")
    res = pd.DataFrame(gs.cv_results_).sort_values("rank_test_score")
    print("\nTop 5 Grid Configurations:")
    for idx, r in res.head(5).iterrows():
        print(f"Rank {r['rank_test_score']}: Mean F1 = {r['mean_test_score']:.4f} (+/- {r['std_test_score']:.4f}) | {r['params']}")

    best = gs.best_estimator_

    print("\n" + "=" * 70)
    print("PHASE 8: THRESHOLD SELECTION & ONE-TIME TEST EVALUATION")
    print("=" * 70)

    # Decision threshold chosen purely from out-of-fold training probabilities
    oof_scores = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof_scores)

    # Business requirement: Precision >= 0.99 to protect real user messages
    ok = prec[:-1] >= 0.99
    threshold = thr[ok][np.argmax(rec[:-1][ok])] if ok.any() else 0.5
    print(f"Chosen decision threshold (Precision >= 0.99): {threshold:.4f}")

    # Now and ONLY now: evaluate once on X_test
    best.fit(X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]

    for label, t in [("Default Threshold (0.500)", 0.5), ("Tuned Threshold (Precision >= 0.99)", threshold)]:
        pred = (proba >= t).astype(int)
        print(f"\nEvaluation with {label} [t={t:.3f}]:")
        print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=3))
        cm = confusion_matrix(y_test, pred)
        print("Confusion Matrix:")
        print(cm)

    # Bootstrap 95% Confidence Interval on Test Set F1
    rng = np.random.default_rng(42)
    y_arr = y_test.values
    p_arr = (proba >= threshold).astype(int)
    boots = [
        f1_score(y_arr[idx], p_arr[idx])
        for idx in (rng.integers(0, len(y_arr), len(y_arr)) for _ in range(1000))
    ]
    ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
    print(f"\nFinal Test F1: {f1_score(y_test, p_arr):.4f}")
    print(f"Test F1 95% Bootstrap Confidence Interval: [{ci_low:.3f}, {ci_high:.3f}]")

    # Plot and save Precision-Recall curve
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax, name="Tuned Classifier")
    plt.title("Precision-Recall Curve on Test Set")
    plt.savefig("test_precision_recall_curve.png", bbox_inches="tight")
    plt.close()
    print("Saved test_precision_recall_curve.png")

    print("\n" + "=" * 70)
    print("PHASE 9: FEATURE ENGINEERING EXPERIMENT (FEATUREUNION)")
    print("=" * 70)

    features = FeatureUnion([
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

    print("\n" + "=" * 70)
    print("PHASE 10: SAVE MODEL AND DEMO PREDICTIONS")
    print("=" * 70)

    model_path = "models/spam_clf.joblib"
    joblib.dump({"model": best, "threshold": float(threshold)}, model_path)
    print(f"Saved production pipeline and calibrated threshold to '{model_path}'")

    # Verify reloading from disk
    bundle = joblib.load(model_path)
    print("Successfully verified joblib re-load.")

    def predict_spam(messages):
        probs = bundle["model"].predict_proba(messages)[:, 1]
        thr = bundle["threshold"]
        return [
            {
                "text": m,
                "spam_prob": round(float(p), 4),
                "is_spam": bool(p >= thr)
            }
            for m, p in zip(messages, probs)
        ]

    demo_msgs = [
        "WINNER!! You have been selected for a £900 prize. Call 09061701461 now",
        "hey are we still meeting at 6?",
        "Your parcel is waiting, confirm details at http://bit.ly/x1",
        "Can you send me the lecture notes when you get home?",
        "URGENT! We noticed an unauthorized login attempt to your bank account. Reply YES to verify.",
        "Ok see you tomorrow at lunch!"
    ]

    print("\nInference Demonstration:")
    for res_item in predict_spam(demo_msgs):
        status = "[SPAM]" if res_item["is_spam"] else "[HAM] "
        print(f"{status} (p={res_item['spam_prob']:.4f}) : {res_item['text']}")

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
