# sentiment_model.py
"""
Sentiment Analysis Module for Armenia Restaurants & Hotels Explorer
------------------------------------------------------------------

This module provides:
1. Text cleaning using NLTK stopwords
2. Loading of pre-trained SVM sentiment classifier + TF-IDF vectorizer
3. A robust sentiment scoring function:
       -1 = Negative, 0 = Neutral, +1 = Positive
4. DataFrame helper to apply sentiment scoring column-wise

The sentiment model was trained offline using restaurant & hotel reviews
and exported as:
    - models/svm_model.pkl
    - models/tfidf_vectorizer.pkl

These models are loaded at runtime and used to compute sentiment scores
for every review in the dataset.
"""

import pickle
import os
import re
import nltk
from nltk.corpus import stopwords


# ----------------------------------------------------
# 1. Configure NLTK data path
# ----------------------------------------------------
# This ensures Streamlit (or any deployed environment) can find NLTK resources.
# Add Streamlit Cloud's writable temp directory for NLTK data
NLTK_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DIR)



# ----------------------------------------------------
# 2. Load English stopwords (download if missing)
# ----------------------------------------------------
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", download_dir=NLTK_DIR)
    stop_words = set(stopwords.words("english"))



# ----------------------------------------------------
# 3. Text cleaning function
# ----------------------------------------------------
def clean_text(text):
    """
    Normalize and clean input text:
      - convert to lowercase
      - remove punctuation and non-letter characters
      - remove stopwords
      - return tokenized/cleaned string

    Parameters
    ----------
    text : str
        Raw text input.

    Returns
    -------
    str
        Cleaned text ready for vectorization.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # keep only letters + whitespace
    tokens = [w for w in text.split() if w not in stop_words]

    return " ".join(tokens)


# ----------------------------------------------------
# 4. Load the sentiment SVM model + TF-IDF vectorizer
# ----------------------------------------------------
# Models are stored inside the "models" folder next to this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = pickle.load(open(os.path.join(MODELS_DIR, "svm_model.pkl"), "rb"))
vectorizer = pickle.load(open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb"))


# ----------------------------------------------------
# 5. Compute sentiment score for a single text
# ----------------------------------------------------
def svm_sentiment_score(text: str):
    """
    Compute sentiment score for a single review.

    The function supports two types of SVM models:
    - Models with `predict_proba()`: returns probability-weighted sentiment score.
    - Models without probability output: uses class label only.

    Returns a float in [-1, 0, +1]:

        -1 → Negative
         0 → Neutral
        +1 → Positive

    Parameters
    ----------
    text : str
        Review text.

    Returns
    -------
    float or None
        Sentiment score, or None if text is invalid.
    """

    # Handle empty or non-string inputs
    if text is None or not isinstance(text, str) or text.strip() == "":
        return None

    # Convert text → cleaned tokens → TF-IDF vector
    processed = clean_text(text)
    X = vectorizer.transform([processed])

    # ------------------------------------------
    # CASE 1: Model supports probability output
    # ------------------------------------------
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = model.classes_  # e.g. ["Negative", "Neutral", "Positive"]

        score = 0.0
        for cls, p in zip(classes, proba):
            c = str(cls).lower()
            if c.startswith("neg"):
                score += p * (-1)
            elif c.startswith("neu"):
                score += p * (0)
            elif c.startswith("pos"):
                score += p * (+1)

        return float(score)

    # ------------------------------------------
    # CASE 2: Model outputs class label only
    # ------------------------------------------
    pred = str(model.predict(X)[0]).lower()

    if pred.startswith("pos"):
        return 1.0
    elif pred.startswith("neu"):
        return 0.0
    elif pred.startswith("neg"):
        return -1.0

    # Fallback for unexpected labels
    return 0.0


# ----------------------------------------------------
# 6. Apply sentiment scoring column-wise on a DataFrame
# ----------------------------------------------------
def apply_sentiment(df, text_column="Review Text", new_col="sentiment_score"):
    """
    Add a sentiment score column to a Pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with text reviews.
    text_column : str
        Name of the column containing review text.
    new_col : str
        Name of the output sentiment column.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with an additional sentiment column.
    """
    df[new_col] = df[text_column].apply(svm_sentiment_score)
    return df
