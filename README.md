# 📰 Fake News Detector using Linear Regression and Streamlit

An end-to-end **Fake News Detection System** built with **Python**, **Linear Regression**, **Scikit-Learn**, and **Streamlit**. This project classifies news articles as **Real** or **Fake** using TF-IDF text features and converts continuous regression outputs into binary predictions using a **0.5 threshold**.

---

## 📌 Project Description

Misinformation spreads rapidly across digital platforms, making automated fake news detection a critical research area. This system analyzes news article text, extracts meaningful linguistic features using **TF-IDF Vectorization**, and applies **Linear Regression** to predict whether an article is authentic or fabricated.

Designed for **academic purposes**, this project demonstrates how a regression algorithm can be adapted for binary classification tasks.

---

## 🎯 Objectives

- Build a complete NLP pipeline for fake news detection
- Preprocess and clean large-scale news datasets
- Engineer text features using TF-IDF (max 5,000 features, bigrams)
- Train a **Linear Regression** model (mandatory algorithm)
- Convert regression outputs to binary labels (≥ 0.5 = Real, < 0.5 = Fake)
- Evaluate using classification and regression metrics
- Deploy an interactive **Streamlit** web application
- Generate visual analytics for model interpretation

---

## 📂 Dataset

This project uses the widely-used **Fake and Real News Dataset**:

| File | Description | Label |
|------|-------------|-------|
| `Fake.csv` | Fake news articles (~23,000+) | 0 (Fake) |
| `True.csv` | Real news articles (~21,000+) | 1 (Real) |

**Columns:** `title`, `text`, `subject`, `date`

**Source:** [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Both files are merged into a single dataframe for training.

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3 |
| Machine Learning | Scikit-Learn (Linear Regression) |
| Data Processing | Pandas, NumPy |
| NLP | NLTK (Porter Stemmer, Stopwords) |
| Feature Engineering | TF-IDF Vectorizer |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Streamlit |

---

## 📁 Project Structure

```
Fake-News-Detector/
│
├── dataset/
│   ├── Fake.csv                  # Fake news dataset
│   └── True.csv                  # Real news dataset
│
├── models/
│   ├── linear_model.pkl          # Trained Linear Regression model
│   ├── tfidf_vectorizer.pkl      # Fitted TF-IDF vectorizer
│   └── metrics.json              # Evaluation metrics
│
├── assets/
│   ├── accuracy_graph.png          # Performance metrics chart
│   ├── label_distribution.png    # Class distribution chart
│   ├── prediction_distribution.png
│   ├── confusion_matrix.png      # Confusion matrix heatmap
│   └── feature_importance.png    # Top TF-IDF coefficients
│
├── screenshots/                  # App screenshots (placeholder)
│
├── preprocessing.py              # Reusable text preprocessing module
├── train_model.py                # Model training script
├── app.py                        # Streamlit web application
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone or download the project:**

```bash
cd Fake-News-Detector
```

2. **Create a virtual environment (recommended):**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Ensure datasets are present:**

Place `Fake.csv` and `True.csv` inside the `dataset/` folder if not already included.

---

## 🚀 How to Run

### Step 1: Train the Model

```bash
python train_model.py
```

This will:
- Load and merge both CSV files
- Preprocess all text data
- Train TF-IDF + Linear Regression
- Evaluate and print metrics
- Save model artifacts to `models/`
- Generate visualizations in `assets/`

### Step 2: Launch the Streamlit App

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`) in your browser.

---

## 🔬 Algorithm Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Linear Regression |
| Vectorizer | TF-IDF |
| Max Features | 5,000 |
| N-gram Range | (1, 2) |
| Train/Test Split | 80% / 20% |
| Random State | 42 |
| Classification Threshold | 0.5 |

**Decision Rule:**
- `prediction >= 0.5` → **REAL NEWS** 🟢
- `prediction < 0.5` → **FAKE NEWS** 🔴

---

## 📊 Evaluation Metrics

The model is evaluated using:

**Classification Metrics:**
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report

**Regression Metrics:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🖥️ Streamlit Application Features

- Modern dark-themed responsive UI
- Multi-line text input for news articles
- Real-time text statistics (word count, character count, reading time)
- Live text preview
- Predict and Clear buttons
- Loading spinner and progress bar
- Success/error animations
- Confidence score and probability display
- Sidebar with project info, dataset stats, and model performance
- Embedded visualization gallery
- Input validation with empty text warnings

---

## 📸 Screenshots

> Place application screenshots in the `screenshots/` folder.

| Screenshot | Description |
|------------|-------------|
| `screenshots/home.png` | Main application page |
| `screenshots/prediction_real.png` | Real news prediction result |
| `screenshots/prediction_fake.png` | Fake news prediction result |
| `screenshots/sidebar.png` | Sidebar navigation panel |

---

## 🔮 Future Scope

- Experiment with feature selection techniques
- Add cross-validation for robust evaluation
- Integrate word embeddings (Word2Vec, GloVe) as additional features
- Deploy the app on cloud platforms (Streamlit Cloud, Heroku, AWS)
- Add batch prediction via CSV upload
- Implement model explainability with SHAP/LIME
- Extend to multi-class classification (satire, propaganda, etc.)
- Add user authentication and prediction history

---

## 👤 Author

**Anshika Singh** ([@anshika152330](https://github.com/anshika152330))

- Academic / Final Year College Project
- Machine Learning & NLP Specialization

---

## 📄 License

This project is intended for **educational and academic purposes only**.

---

## 🙏 Acknowledgements

- [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) by Clément Bisaillon
- Scikit-Learn Documentation
- Streamlit Documentation
- NLTK Project
