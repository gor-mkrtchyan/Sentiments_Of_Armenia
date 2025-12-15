# sentiment_model.py

import pickle
import os
import re
import nltk
from nltk.corpus import stopwords

# ----------------------------------------------------
# 1. Configure your NLTK data path
# ----------------------------------------------------
# Ensures Streamlit can find your downloaded NLTK data
nltk.data.path.append("/Users/gomkrtchyan/Desktop/venv/nltk_data")

# ----------------------------------------------------
# 2. Ensure stopwords are available
# ----------------------------------------------------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", download_dir="/Users/gomkrtchyan/Desktop/venv/nltk_data")
    stop_words = set(stopwords.words("english"))

# ----------------------------------------------------
# 3. clean_text function from your notebook
# ----------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

# ----------------------------------------------------
# 4. Load Model + Vectorizer from /models directory
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = pickle.load(open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb"))

# ----------------------------------------------------
# 5. Sentiment scoring (supports 3 classes)
# Positive = +1, Neutral = 0, Negative = -1
# ----------------------------------------------------
def svm_sentiment_score(text: str):
    """
    Returns a sentiment value in [-1, 0, +1].
    -1 = Negative
     0 = Neutral
    +1 = Positive
    """

    if text is None or not isinstance(text, str) or text.strip() == "":
        return None

    processed = clean_text(text)
    X = vectorizer.transform([processed])

    # ------------------------------------------
    # CASE 1: Model supports probability output
    # ------------------------------------------
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = model.classes_  # e.g., ['Negative','Neutral','Positive']

        # Compute probability-weighted sentiment
        score = 0.0
        for cls, p in zip(classes, proba):
            cls_str = str(cls).lower()
            if cls_str.startswith("neg"):
                score += p * (-1)
            elif cls_str.startswith("neu"):
                score += p * (0)
            elif cls_str.startswith("pos"):
                score += p * (1)
        return float(score)

    # ------------------------------------------
    # CASE 2: Model outputs labels only
    # ------------------------------------------
    pred = str(model.predict(X)[0]).lower()

    if pred.startswith("pos"):
        return 1.0
    elif pred.startswith("neu"):
        return 0.0
    elif pred.startswith("neg"):
        return -1.0

    # Unknown or unexpected label → neutral fallback
    return 0.0

# ----------------------------------------------------
# 6. Apply sentiment scoring to a dataframe column
# ----------------------------------------------------
def apply_sentiment(df, text_column="Review Text", new_col="sentiment_score"):
    df[new_col] = df[text_column].apply(svm_sentiment_score)
    return df
