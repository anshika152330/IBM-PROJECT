"""
Streamlit web application for the Fake News Detection System.

Provides an interactive UI to classify news articles as REAL or FAKE using a
Linear Regression model trained on TF-IDF features.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import streamlit as st

from preprocessing import get_text_stats, preprocess_text

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
MODEL_PATH = MODELS_DIR / "linear_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS - Dark theme and professional styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);
        color: #e8e8e8;
    }

    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        animation: fadeInDown 0.8s ease-out;
    }

    .sub-header {
        text-align: center;
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-out;
    }

    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }

    .stat-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2));
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .stat-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }

    .stat-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 0.25rem;
    }

    .result-real {
        background: linear-gradient(135deg, rgba(46, 204, 113, 0.2), rgba(39, 174, 96, 0.3));
        border: 2px solid #2ecc71;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: successPulse 1.5s ease-in-out;
    }

    .result-fake {
        background: linear-gradient(135deg, rgba(231, 76, 60, 0.2), rgba(192, 57, 43, 0.3));
        border: 2px solid #e74c3c;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: errorShake 0.6s ease-in-out;
    }

    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .preview-box {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #667eea;
        font-size: 0.95rem;
        color: #cbd5e0;
        max-height: 150px;
        overflow-y: auto;
    }

    .footer {
        text-align: center;
        padding: 2rem 0 1rem;
        color: #718096;
        font-size: 0.85rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: 3rem;
    }

    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes successPulse {
        0% { transform: scale(0.95); opacity: 0; }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); opacity: 1; }
    }

    @keyframes errorShake {
        0%, 100% { transform: translateX(0); }
        20% { transform: translateX(-8px); }
        40% { transform: translateX(8px); }
        60% { transform: translateX(-4px); }
        80% { transform: translateX(4px); }
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }

    .stTextArea textarea {
        background-color: #000 !important;
        border: none !important;
        border-radius: 12px !important;
        color: #e8e8e8 !important;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Load trained model, vectorizer, and metrics from disk."""
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None, None

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(VECTORIZER_PATH, "rb") as file:
        vectorizer = pickle.load(file)

    metrics = None
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

    return model, vectorizer, metrics


def predict_news(text: str, model, vectorizer) -> dict:
    """
    Run inference on input text.

    Returns prediction label, probability, confidence, and elapsed time.
    """
    start_time = time.time()

    cleaned_text = preprocess_text(text)
    features = vectorizer.transform([cleaned_text])
    raw_prediction = float(model.predict(features)[0])
    probability = float(np.clip(raw_prediction, 0.0, 1.0))
    is_real = raw_prediction >= THRESHOLD
    confidence = probability if is_real else 1.0 - probability
    elapsed = time.time() - start_time

    return {
        "is_real": is_real,
        "raw_prediction": raw_prediction,
        "probability": probability,
        "confidence": confidence,
        "elapsed_time": elapsed,
        "cleaned_text": cleaned_text,
    }


def render_sidebar(metrics) -> None:
    """Render sidebar with project information."""
    st.sidebar.title("📋 Navigation")

    section = st.sidebar.radio(
        "Explore",
        [
            "Project Overview",
            "Dataset Information",
            "Algorithm Used",
            "Model Performance",
            "Developer Information",
        ],
    )

    st.sidebar.markdown("---")

    if section == "Project Overview":
        st.sidebar.markdown(
            """
            **Fake News Detector** is an end-to-end machine learning system
            that classifies news articles as **Real** or **Fake** using
            **Linear Regression** on TF-IDF text features.

            Built for academic demonstration of regression-based
            binary classification with a 0.5 threshold.
            """
        )

    elif section == "Dataset Information":
        if metrics:
            st.sidebar.metric("Total Samples", f"{metrics.get('total_samples', 'N/A'):,}")
            st.sidebar.metric("Fake Articles", f"{metrics.get('fake_count', 'N/A'):,}")
            st.sidebar.metric("Real Articles", f"{metrics.get('real_count', 'N/A'):,}")
            st.sidebar.metric("Train Size", f"{metrics.get('train_size', 'N/A'):,}")
            st.sidebar.metric("Test Size", f"{metrics.get('test_size', 'N/A'):,}")
        else:
            st.sidebar.warning("Train the model first to view dataset stats.")

        st.sidebar.markdown(
            """
            **Sources:**
            - `Fake.csv` — Fake news articles (Label: 0)
            - `True.csv` — Real news articles (Label: 1)
            """
        )

    elif section == "Algorithm Used":
        st.sidebar.markdown(
            """
            | Component | Details |
            |-----------|---------|
            | **Algorithm** | Linear Regression |
            | **Vectorizer** | TF-IDF |
            | **Max Features** | 5,000 |
            | **N-grams** | (1, 2) |
            | **Threshold** | 0.5 |
            | **Split** | 80/20 |

            **Decision Rule:**
            - Prediction ≥ 0.5 → Real News
            - Prediction < 0.5 → Fake News
            """
        )

    elif section == "Model Performance":
        if metrics:
            st.sidebar.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            st.sidebar.metric("Precision", f"{metrics['precision']:.2%}")
            st.sidebar.metric("Recall", f"{metrics['recall']:.2%}")
            st.sidebar.metric("F1 Score", f"{metrics['f1_score']:.2%}")
            st.sidebar.metric("R² Score", f"{metrics['r2_score']:.4f}")
            st.sidebar.metric("RMSE", f"{metrics['rmse']:.4f}")
        else:
            st.sidebar.warning("Run `python train_model.py` to generate metrics.")

    elif section == "Developer Information":
        st.sidebar.markdown(
            """
            **Project:** Fake News Detector using Linear Regression and Streamlit

            **Technologies:**
            - Python 3
            - Scikit-Learn
            - Pandas & NumPy
            - Matplotlib & Seaborn
            - Streamlit
            - NLTK

            **Purpose:** Academic / Final Year Project
            """
        )


def render_visualizations() -> None:
    """Display training visualizations if available."""
    viz_files = {
        "Accuracy Metrics": "accuracy_graph.png",
        "Label Distribution": "label_distribution.png",
        "Prediction Distribution": "prediction_distribution.png",
        "Confusion Matrix": "confusion_matrix.png",
        "Feature Importance": "feature_importance.png",
    }

    available = {name: ASSETS_DIR / fname for name, fname in viz_files.items() if (ASSETS_DIR / fname).exists()}

    if not available:
        st.info("Visualizations will appear here after running `python train_model.py`.")
        return

    st.markdown("### 📊 Model Visualizations")
    cols = st.columns(2)

    for index, (title, path) in enumerate(available.items()):
        with cols[index % 2]:
            st.markdown(f"**{title}**")
            st.image(str(path), use_container_width=True)


def main() -> None:
    """Main application entry point."""
    model, vectorizer, metrics = load_artifacts()

    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detector</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Detect fake news articles using Linear Regression & TF-IDF | '
        "Academic ML Project</p>",
        unsafe_allow_html=True,
    )

    render_sidebar(metrics)

    # Model status check
    if model is None or vectorizer is None:
        st.error(
            "⚠️ Model not found! Please train the model first:\n\n"
            "```\npython train_model.py\n```"
        )
        st.stop()

    # Main content layout
    col_input, col_stats = st.columns([2, 1])

    with col_input:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### ✍️ Enter News Article")
        news_text = st.text_area(
            "Paste or type a news article below:",
            height=250,
            placeholder="Enter the news article text here for classification...",
            label_visibility="collapsed",
            key="news_input",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col_stats:
        stats = get_text_stats(news_text)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📈 Text Statistics")

        stat_cols = st.columns(2)
        with stat_cols[0]:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{stats["word_count"]}</div>'
                f'<div class="stat-label">Words</div></div>',
                unsafe_allow_html=True,
            )
        with stat_cols[1]:
            st.markdown(
                f'<div class="stat-card"><div class="stat-value">{stats["character_count"]}</div>'
                f'<div class="stat-label">Characters</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="stat-card" style="margin-top:0.5rem;">'
            f'<div class="stat-value">{stats["reading_time_minutes"]} min</div>'
            f'<div class="stat-label">Est. Reading Time</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Live text preview
    if news_text.strip():
        preview = news_text.strip()[:500] + ("..." if len(news_text.strip()) > 500 else "")
        st.markdown("### 👁️ Live Text Preview")
        st.markdown(f'<div class="preview-box">{preview}</div>', unsafe_allow_html=True)

    # Action buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])

    with btn_col1:
        predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

    with btn_col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state.pop("prediction_result", None)
        st.rerun()

    # Prediction logic
    if predict_btn:
        if not news_text.strip():
            st.warning("⚠️ Please enter some news text before predicting.")
        elif len(news_text.strip()) < 20:
            st.warning("⚠️ Text is too short. Please enter at least 20 characters for reliable prediction.")
        else:
            progress = st.progress(0, text="Initializing prediction...")
            for step in range(100):
                time.sleep(0.005)
                progress.progress(step + 1, text="Analyzing text features...")

            with st.spinner("🔄 Running Linear Regression model..."):
                result = predict_news(news_text, model, vectorizer)

            progress.progress(100, text="Complete!")
            time.sleep(0.3)
            progress.empty()
            st.session_state["prediction_result"] = result

    # Display results
    if "prediction_result" in st.session_state:
        result = st.session_state["prediction_result"]
        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")

        if result["is_real"]:
            st.markdown(
                f"""
                <div class="result-real">
                    <div class="result-title">🟢 REAL NEWS</div>
                    <p style="font-size:1.1rem;">This article appears to be <strong>authentic</strong>.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.success("Classification: REAL NEWS ✅")
        else:
            st.markdown(
                f"""
                <div class="result-fake">
                    <div class="result-title">🔴 FAKE NEWS</div>
                    <p style="font-size:1.1rem;">This article appears to be <strong>misleading or fake</strong>.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.error("Classification: FAKE NEWS ❌")

        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Confidence Score", f"{result['confidence']:.2%}")
        with metric_cols[1]:
            st.metric("Probability (Real)", f"{result['probability']:.4f}")
        with metric_cols[2]:
            st.metric("Raw Prediction", f"{result['raw_prediction']:.4f}")
        with metric_cols[3]:
            st.metric("Prediction Time", f"{result['elapsed_time']:.4f}s")

    # Visualizations section
    st.markdown("---")
    render_visualizations()

    # Footer
    st.markdown(
        """
        <div class="footer">
            <p>📰 <strong>Fake News Detector</strong> | Linear Regression + TF-IDF + Streamlit</p>
            <p>Built with ❤️ for Academic & Research Purposes | © 2026</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
