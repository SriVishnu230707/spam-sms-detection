# SMS Spam Detection: End-to-End ML Pipeline

An end-to-end Machine Learning system for classifying SMS text messages as **Ham** (legitimate) or **Spam** (unsolicited / fraudulent).

This project follows a strict, disciplined machine learning engineering methodology:
1. **Zero Data Leakage:** Full deduplication prior to train/test split.
2. **Untouched Test Set:** The test set is evaluated exactly once at the conclusion.
3. **Out-of-Fold Tuning:** All model comparisons, hyperparameter grid searches, error analyses, and decision threshold selections are performed on training folds using Stratified 5-Fold Cross-Validation.
4. **Business-Aligned Calibration:** Decision threshold calibrated to guarantee Precision $\ge 0.99$, minimizing destructive false positives.
5. **Statistical Uncertainty:** Final test F1 quantified via 95% Bootstrap Confidence Intervals.

---

## 1. Problem and Metric Choice

- **Why Accuracy is Deceptive:** The SMS Spam dataset has an ~87% ham / ~13% spam imbalance. A naive `DummyClassifier` predicting all messages as "ham" achieves 87.4% accuracy while catching 0% of spam (F1 = 0.000).
- **Metric Selection:** We evaluate models using **Precision**, **Recall**, **F1-Score**, and **Average Precision (PR-AUC)**.
- **Cost Asymmetry (False Positive vs. False Negative):** In SMS filtering, **False Positives** (blocking a legitimate personal, medical, or banking message) are far more damaging to user trust than **False Negatives** (letting an occasional spam slip into the inbox). Consequently, our deployment objective is to require **Precision $\ge 0.99$** and then maximize Recall.

---

## 2. Data Notes & Preprocessing

- **Source:** UCI Machine Learning Repository SMS Spam Collection / Kaggle SMS Spam dataset (`5,572` initial records).
- **Deduplication:** There are **403 exact duplicate messages** in the raw corpus. If data is split before deduplication, identical texts appear simultaneously in train and test folds, artificially inflating generalization metrics. Deduplication yields **5,169 unique messages**.
- **Class Balance:**
  - **Ham (0):** 4,516 messages (87.37%)
  - **Spam (1):** 653 messages (12.63%)
- **Train / Test Split:** Stratified 80/20 split (`random_state=42`), producing:
  - **Training Set:** 4,135 samples (3,613 ham, 522 spam)
  - **Test Set:** 1,034 samples (903 ham, 131 spam)
- **Feature Normalization (`src/features.py`):**
  Standard TF-IDF tokenizers drop punctuation and treat arbitrary numbers or URLs as unique, low-frequency tokens. We implement regex pre-tokenization:
  - URLs $\to$ `urltoken`
  - Currency amounts (`£100`, `$50`, `€20`) $\to$ `moneytoken`
  - Long numbers $\ge 5$ digits (phone numbers, shortcodes) $\to$ `longnumtoken`

---

## 3. Cross-Validation Benchmark Results

Evaluated across **5 Stratified Folds** on the training set:

| Model Pipeline | CV Precision | CV Recall | CV F1 | CV Avg Precision (PR-AUC) |
| :--- | :---: | :---: | :---: | :---: |
| **DummyClassifier** (Most Frequent) | 0.000 | 0.000 | 0.000 | 0.126 |
| **Multinomial Naive Bayes** | **1.000** | 0.663 | 0.797 | 0.956 |
| **Logistic Regression** (Baseline C=1.0) | 0.998 | 0.774 | 0.871 | 0.978 |
| **Tuned Logistic Regression** (C=30, Balanced) | 0.941 | **0.977** | **0.959** | **0.989** |

---

## 4. Hyperparameter Tuning (Phase 7)

Grid search over TF-IDF n-grams, document frequencies, stop words, and classifier parameters:
- **Best Estimator:** `LogisticRegression(C=30, class_weight='balanced', max_iter=2000)`
- **Vectorizer:** `TfidfVectorizer(ngram_range=(1, 1), min_df=2, stop_words=None)`
- **Best Cross-Validation F1:** **0.9586 $\pm$ 0.0086**

---

## 5. Decision Threshold Calibration & Final Test Evaluation (Phase 8)

Decision threshold was chosen **strictly from out-of-fold cross-validation probabilities** on training data:
$$\text{Threshold} = 0.709 \quad (\text{Precision } \ge 0.99)$$

### Test Set Performance Comparison (1,034 messages, evaluated once):

| Threshold Setting | Precision (Spam) | Recall (Spam) | F1-Score (Spam) | Ham False Positives | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Default (0.500)** | 0.947 | 0.947 | 0.947 | 7 / 903 | 98.6% |
| **Tuned ($t = 0.709$)** | **0.975** | **0.885** | **0.928** | **3 / 903** | 98.3% |

### Confusion Matrix on Test Set ($t = 0.709$):
```
                 Predicted Ham    Predicted Spam
Actual Ham            900               3         (99.7% Specificity)
Actual Spam            15             116         (88.5% Sensitivity)
```

### Statistical Significance:
- **95% Bootstrap Confidence Interval for Test F1 (1,000 iterations):** `[0.895, 0.958]`
- Because the test set contains 131 spam instances, a single misclassification shifts recall by ~0.76 percentage points. The bootstrap interval rigorously captures this sampling variance.

---

## 6. Error Analysis (Out-of-Fold Training Data)

Inspecting out-of-fold prediction errors provides direct insight into model weaknesses:

### Selected Annotated Error Cases:

1. **False Positive:** `"Waiting for your call."` (Actual: Ham, Predicted: Spam)
   - *Explanation:* The word `"call"` carries a high positive spam weight because it frequently appears in spam call-to-actions ("Call 0800..."). In this ultra-short message without context, `"call"` drove the prediction above the spam threshold.
2. **False Negative:** `"SplashMobile: Choose from 1000s of gr8 tones each wk..."` (Actual: Spam, Predicted: Ham)
   - *Explanation:* Heavy use of leetspeak and abbreviated slang (`"gr8"`, `"wk"`) diluted standard keyword activations, slipping past the unigram vocabulary.
3. **False Negative:** `"WIN: We have a winner! Mr. T. Foley won an iPod..."` (Actual: Spam, Predicted: Ham)
   - *Explanation:* Phrased like a personal announcement naming an individual ("Mr. T. Foley") without explicit phone numbers or payment links.
4. **False Negative:** `"18 days to Euro2004 kickoff! U will be kept informed..."` (Actual: Spam, Predicted: Ham)
   - *Explanation:* Promotional broadcast without aggressive commercial keywords or urgent financial incentives.

### Top Learned Coefficients:
- **Most Spammy:** `longnumtoken`, `moneytoken`, `urltoken`, `call`, `reply`, `free`, `txt`, `text`, `mobile`, `stop`, `150p`
- **Most Hammy:** `my`, `me`, `that`, `gt`, `lt`, `but`, `ok`, `it`, `in`, `da`, `how`, `so`, `at`, `going`, `ll`

---

## 7. Feature Engineering Experiment: TF-IDF vs. FeatureUnion (Phase 9)

We compared standard TF-IDF against a `FeatureUnion` incorporating dense metadata features (message length, digit count, capital letter ratio, URL indicator, currency symbol indicator):
- **TF-IDF only:** $0.9564 \pm 0.0056$ F1
- **TF-IDF + Extra Features:** $0.9425 \pm 0.0139$ F1
- **Conclusion:** Dense metadata features did not improve performance. The custom preprocessor already converts currency, phone numbers, and URLs into discrete tokens (`moneytoken`, `longnumtoken`, `urltoken`), which TF-IDF handles with higher discriminative power and lower variance than scaled dense numeric columns.

---

## 8. Limitations

1. **Temporal & Geographic Drift:** The SMS Spam Collection dataset dates from 2012 and primarily originates from UK and European phone networks (noticeable UK currency `£` and prefixes `087...`, `150p`). Modern spam often utilizes OTP scams, delivery phishing (`usps-tracking`, `dhl-alert`), and cryptocurrency fraud.
2. **Obfuscation (Adversarial Text):** Spammers increasingly disguise trigger words using homoglyphs or spacing (e.g. `c a s h`, `p r ! z e`). Character n-grams (`analyzer="char_wb"`) or transformer-based embeddings can be added for deeper robustness.
3. **Language Scope:** Model is trained exclusively on English SMS texts.

---

## 9. Project Structure

```
c:\projects\Spam SMS detection\
├── data/
│   ├── spam.csv                 # Kaggle format dataset (latin-1)
│   └── SMSSpamCollection        # UCI tab-separated dataset
├── models/
│   └── spam_clf.joblib          # Serialized production pipeline & calibrated threshold
├── src/
│   ├── __init__.py
│   └── features.py              # Custom token normalization & extra feature extractor
├── web/                         # Modern Standalone Web UI (HTML5, Vanilla CSS, JS)
│   ├── index.html               # Semantic UI with preset chips, threshold slider & gauge
│   ├── style.css                # Dark mode glassmorphic styling & micro-animations
│   └── app.js                   # Client logic with debounced live auto-detection
├── web_app.py                   # Standalone Web Server & REST API (zero extra dependencies)
├── app.py                       # Interactive Streamlit application
├── run_pipeline.py              # CLI runner script for full Phase 1-10 pipeline
├── generate_notebook.py         # Notebook generator
├── notebook.ipynb               # Fully structured Jupyter Notebook
├── eda_length_boxplot.png       # Generated EDA visualization
├── test_precision_recall_curve.png # Generated test PR curve
├── requirements.txt             # Python dependencies
└── README.md                    # Technical documentation
```

---

## 10. Running the Project

### Option A: Run the Modern Standalone Web UI (Recommended)
Zero extra framework installations required (runs on Python's standard library):
```bash
python web_app.py
```
Then visit **`http://localhost:5000`** in your browser.

#### REST API Endpoint
You can also send prediction requests directly to the REST API:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER! You won £1,000 cash. Call 09061701461 now!"}'
```

Response:
```json
{
  "text": "WINNER! You won £1,000 cash. Call 09061701461 now!",
  "spam_probability": 0.9941,
  "spam_percentage": 99.4,
  "is_spam": true,
  "verdict": "SPAM",
  "threshold": 0.7088,
  "normalized_text": "winner! you won  moneytoken  cash. call  longnumtoken  now!",
  "extracted_tokens": [
    { "token": "moneytoken", "label": "Currency Amount (£, $, €)", "type": "money" },
    { "token": "longnumtoken", "label": "Phone Number / Shortcode (5+ digits)", "type": "number" }
  ]
}
```

### Option B: Run via Streamlit
```bash
streamlit run app.py
```

### Option C: Execute the Full Training Pipeline
```bash
python run_pipeline.py
```

### Option D: Standalone Python Inference
```python
import joblib
from src.features import normalize

bundle = joblib.load("models/spam_clf.joblib")
model = bundle["model"]
threshold = bundle["threshold"]

messages = [
    "WINNER!! You have been selected for a £900 prize. Call 09061701461 now",
    "hey are we still meeting at 6?"
]
probabilities = model.predict_proba(messages)[:, 1]

for text, prob in zip(messages, probabilities):
    label = "SPAM" if prob >= threshold else "HAM"
    print(f"[{label}] (p={prob:.4f}) : {text}")
```

