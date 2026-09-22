import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sys
import html

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath("."))
from src.features import normalize

st.set_page_config(
    page_title="SMS Spam Shield | ML Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', -apple-system, sans-serif;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 24px;
        border: 1px solid #334155;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 9999px;
        margin-right: 8px;
    }
    .badge-cyan { background-color: #0e7490; color: #cffafe; }
    .badge-emerald { background-color: #065f46; color: #d1fae5; }
    .badge-amber { background-color: #92400e; color: #fef3c7; }
    .card-spam {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        color: #fef2f2;
    }
    .card-ham {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        color: #ecfdf5;
    }
    .token-box {
        background: #1e293b;
        border-radius: 8px;
        padding: 12px;
        font-family: 'Courier New', monospace;
        color: #38bdf8;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_bundle():
    model_path = os.path.join(os.path.dirname(__file__), "models", "spam_clf.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Run `python run_pipeline.py` first.")
        st.stop()
    return joblib.load(model_path)


bundle = load_model_bundle()
model = bundle["model"]
default_threshold = bundle.get("threshold", 0.709)

# Sidebar
st.sidebar.title("⚙️ Model Configuration")
st.sidebar.markdown("### Business Rule Calibration")
st.sidebar.info(
    "To protect user experience, false positives (blocking a legitimate message) "
    "are penalized heavily. The production decision threshold is calibrated to guarantee "
    "**Precision ≥ 0.99** on out-of-fold training data."
)

threshold = st.sidebar.slider(
    "Spam Decision Threshold",
    min_value=0.10,
    max_value=0.95,
    value=float(default_threshold),
    step=0.01,
    help="Messages with spam probability at or above this value are flagged as Spam."
)

st.sidebar.markdown(f"**Current Threshold:** `{threshold:.3f}`")
if abs(threshold - default_threshold) < 1e-3:
    st.sidebar.success("✅ Operating at optimal tuned threshold (Precision ≥ 0.99)")
elif threshold < 0.5:
    st.sidebar.warning("⚠️ Low threshold: higher recall, but higher risk of false positives.")
else:
    st.sidebar.info("ℹ️ Custom threshold applied.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Test Generalization")
st.sidebar.markdown("""
- **Final Test F1:** `0.928`
- **95% Bootstrap CI:** `[0.895, 0.958]`
- **Spam Precision:** `97.5%`
- **Spam Recall:** `88.5%`
- **Ham False Positives:** Only 3 / 903
""")

# Main View
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; font-size: 28px;">🛡️ SMS Spam Detection System</h1>
    <p style="margin: 6px 0 12px 0; color: #94a3b8;">
        Industrial-grade TF-IDF + Logistic Regression classification pipeline with custom entity normalization.
    </p>
    <div>
        <span class="badge-pill badge-cyan">Zero Data Leakage</span>
        <span class="badge-pill badge-emerald">Stratified 5-Fold CV</span>
        <span class="badge-pill badge-amber">Business Tuned Threshold</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Preset demo buttons
st.markdown("##### Quick Presets:")
preset_cols = st.columns(5)
preset_text = None

if preset_cols[0].button("💰 Prize Spam"):
    preset_text = "WINNER!! You have been selected for a £900 prize. Call 09061701461 now to claim your cash award."
if preset_cols[1].button("🚨 Phishing Alert"):
    preset_text = "URGENT! We detected suspicious activity on your bank account. Verify your identity at http://secure-login.bit.ly"
if preset_cols[2].button("☕ Casual Friend"):
    preset_text = "Hey are we still meeting up for coffee at 6pm today? Let me know!"
if preset_cols[3].button("📚 Study Note"):
    preset_text = "Can you please email me the lecture slides from this morning whenever you get home?"
if preset_cols[4].button("📦 Grey Area"):
    preset_text = "Your package 849203 is arriving today. Confirm delivery address at http://track-pkg.info"

# Input text box
input_message = st.text_area(
    "Enter SMS Message to Analyze:",
    value=preset_text if preset_text else "WINNER!! You have won a £1,000 cash reward. Text CLAIM to 87007 or visit http://win-now.com to collect your voucher.",
    height=110
)

col1, col2 = st.columns([1.2, 1])

if input_message.strip():
    # Model inference
    spam_prob = float(model.predict_proba([input_message])[0, 1])
    is_spam = spam_prob >= threshold
    normalized_input = normalize(input_message)

    with col1:
        st.markdown("### Classification Verdict")
        if is_spam:
            st.markdown(f"""
            <div class="card-spam">
                <h2 style="margin:0; color:#fca5a5;">🚨 SPAM DETECTED</h2>
                <p style="margin: 8px 0; font-size: 16px;">
                    This message exhibits strong characteristics of fraudulent or promotional unsolicited content.
                </p>
                <div style="font-size: 20px; font-weight: bold; margin-top: 10px;">
                    Spam Probability: {spam_prob * 100:.1f}% &nbsp; (Threshold: {threshold * 100:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-ham">
                <h2 style="margin:0; color:#6ee7b7;">✅ LEGITIMATE (HAM)</h2>
                <p style="margin: 8px 0; font-size: 16px;">
                    This message appears to be safe and legitimate personal or transactional communication.
                </p>
                <div style="font-size: 20px; font-weight: bold; margin-top: 10px;">
                    Spam Probability: {spam_prob * 100:.1f}% &nbsp; (Threshold: {threshold * 100:.1f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.progress(min(max(spam_prob, 0.0), 1.0))

    with col2:
        st.markdown("### 🔍 Feature Preprocessing Inspection")
        st.markdown("**Normalized Text (Tokenized for Model Input):**")
        escaped_normalized = html.escape(normalized_input)
        st.markdown(f"<div class='token-box'>{escaped_normalized}</div>", unsafe_allow_html=True)

        detected_tokens = []
        if "urltoken" in normalized_input:
            detected_tokens.append("`urltoken` (Web link / URL detected)")
        if "moneytoken" in normalized_input:
            detected_tokens.append("`moneytoken` (Currency amount detected)")
        if "longnumtoken" in normalized_input:
            detected_tokens.append("`longnumtoken` (Phone number / shortcode detected)")

        if detected_tokens:
            st.markdown("**Extracted Semantic Tokens:**")
            for tok in detected_tokens:
                st.markdown(f"- {tok}")
        else:
            st.markdown("*(No high-risk entity tokens triggered)*")

# Detailed inspection tabs
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📈 Test Set Performance & Confusion Matrix", "🔬 Model Architecture", "📋 Dataset Integrity"])

with tab1:
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### Test Set Confusion Matrix (Threshold = 0.709)")
        cm_df = pd.DataFrame(
            [[900, 3], [15, 116]],
            index=["Actual Ham", "Actual Spam"],
            columns=["Predicted Ham", "Predicted Spam"]
        )
        st.dataframe(cm_df, use_container_width=True)
        st.caption("Only 3 false positives out of 903 legitimate test messages (99.7% ham specificity).")

    with col_t2:
        st.markdown("#### Benchmark Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["Spam Precision", "Spam Recall", "Spam F1-Score", "Overall Accuracy", "F1 95% Bootstrap CI"],
            "Score": ["97.5%", "88.5%", "0.928", "98.3%", "[0.895, 0.958]"]
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

with tab2:
    st.markdown("""
    **Pipeline Structure:**
    1. **Pre-processor:** Custom regex substitution (`src/features.py:normalize`) replacing raw URLs, currency amounts, and phone numbers with generic semantic tokens.
    2. **Vectorizer:** `TfidfVectorizer(min_df=2, ngram_range=(1, 1), stop_words=None)`
    3. **Classifier:** `LogisticRegression(C=30, class_weight='balanced', max_iter=2000)`
    4. **Decision Boundary:** Calibrated via out-of-fold Precision-Recall curve to require $P \ge 0.99$.
    """)

with tab3:
    st.markdown("""
    - **Raw samples:** 5,572
    - **Deduplication:** 403 exact duplicate messages removed before splitting to prevent leakage between train and test sets.
    - **Unique corpus:** 5,169 samples (87.37% Ham, 12.63% Spam).
    - **Train / Test split:** Stratified 80% train (4,135 samples) / 20% test (1,034 samples).
    """)
