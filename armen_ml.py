# ============================================================
# armen_ml.py
# Machine Learning Engine for Armenia Explorer Application
#
# This module provides:
#   • Representation learning via SentenceTransformer embeddings
#   • Sentiment, rating, and hotel-quality computation
#   • Natural-language query interpretation (keywords, price, city, province)
#   • Hybrid scoring recommender for restaurants and hotels
#   • Robust utilities for safe text extraction from noisy dataset columns
#
# NOTE:
#   This file is fully documented for academic capstone evaluation
#   and professional maintainability. No logic has been modified.
# ============================================================

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Embedding Model Initialization
# ------------------------------------------------------------
# We use MiniLM-L6-v2 (384-dim) for efficient semantic similarity.
# This model provides excellent performance for short text such as
# reviews, menu descriptions, and location metadata.
model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------------------------------
# Build Embeddings for Each Unique Location
# ------------------------------------------------------------
def build_embeddings(df: pd.DataFrame, kind: str = "Restaurant") -> Dict[str, np.ndarray]:
    """
    Builds a single semantic embedding vector per location.
    Embeddings represent the "essence" of a location using:
      • Representation column (if provided)
      • Cleaned reviews (fallback)
      • Location name (final fallback)
    """
    if "Location Name" not in df.columns:
        raise KeyError("DataFrame must include 'Location Name' column")

    locations = df["Location Name"].dropna().unique().tolist()
    reps = []
    keys = []

    for loc in locations:
        rows = df[df["Location Name"] == loc]

        # Priority 1 — Representation column (handcrafted summary text)
        rep_text = None
        if "Representation" in df.columns:
            try:
                rep_val = rows["Representation"].dropna().iloc[0]
                rep_text = " ".join(map(str, rep_val)) if isinstance(rep_val, (list, tuple)) else str(rep_val)
            except Exception:
                rep_text = None

        # Priority 2 — cleaned review text
        if not rep_text or str(rep_text).strip().lower() in ("none", "nan", ""):
            rep_text = (
                " ".join(rows["clean_text"].dropna().astype(str).tolist())
                if "clean_text" in rows.columns
                else " ".join(rows["Review Text"].dropna().astype(str).tolist())
            )

        # Priority 3 — fallback → use location name itself
        if not rep_text:
            rep_text = loc

        reps.append(rep_text)
        keys.append(loc)

    # Encode all representations into embeddings in one batch
    enc = model.encode(reps, convert_to_numpy=True)
    embeddings = {keys[i]: np.array(enc[i], dtype=float) for i in range(len(keys))}

    logger.info(
        "Built embeddings (%s) for %d locations. dim=%s",
        kind, len(embeddings), next(iter(embeddings.values())).shape if embeddings else None
    )
    return embeddings


# ------------------------------------------------------------
# Sentiment Computation
# ------------------------------------------------------------
def compute_sentiment(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes average sentiment score per location.
    Sentiment values must already exist (computed externally).
    """
    if "sentiment_score" not in df.columns:
        raise KeyError("DataFrame must contain 'sentiment_score' column.")

    df2 = df.copy()
    df2["sentiment_score"] = pd.to_numeric(df2["sentiment_score"], errors="coerce").fillna(0.0)
    return df2.groupby("Location Name")["sentiment_score"].mean().to_dict()


# ------------------------------------------------------------
# Rating Computation
# ------------------------------------------------------------
def compute_rating(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes mean review rating per location.
    Also supports alternate column names (review_rating, Rating).
    """
    if "Review Rating" not in df.columns:
        # Attempt to detect synonyms
        for alt in ["review_rating", "Rating", "rating"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Review Rating"})
                break
        else:
            raise KeyError("DataFrame must contain 'Review Rating' column.")

    df2 = df.copy()
    df2["Review Rating"] = pd.to_numeric(df2["Review Rating"], errors="coerce")
    grouped = df2.groupby("Location Name")["Review Rating"].mean().fillna(0.0)
    return grouped.to_dict()


# ------------------------------------------------------------
# Hotel Quality Score (Option B weight model)
# ------------------------------------------------------------
def compute_hotel_quality(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes weighted hotel quality score from detailed hotel rating categories.
    Used only in Hotel mode, but falls back to mean review rating if unavailable.
    """
    COLS = {
        "Hotel Cleanliness Rating": 0.35,
        "Hotel Service Rating":     0.25,
        "Hotel Location Rating":    0.20,
        "Hotel Rooms Rating":       0.10,
        "Hotel Value Rating":       0.05,
        "Hotel Sleep Quality Rating": 0.05
    }

    # Identify which rating columns exist
    existing_cols = [c for c in COLS if c in df.columns]
    if not existing_cols:
        logging.warning("No hotel quality columns found — falling back to Review Rating.")
        return compute_rating(df)

    df2 = df.copy()

    # Coerce numeric hotel fields
    for col in COLS:
        df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0.0) if col in df2.columns else 0.0

    # Weighted average across hotel attributes
    def weighted_quality(group):
        means = {col: group[col].mean() for col in COLS}
        return sum(means[col] * COLS[col] for col in COLS)

    result = df2.groupby("Location Name").apply(weighted_quality)
    return result.to_dict()


# ------------------------------------------------------------
# NLP — Keyword Extraction (with spaCy fallback)
# ------------------------------------------------------------
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None


def extract_keywords_from_query(query: str, top_k: int = 8) -> List[str]:
    """
    Extracts the most meaningful keywords from the user query.
    If spaCy is available → use POS tagging.
    Otherwise → use naive token filtering.
    """
    if not query:
        return []

    q = str(query)
    if _nlp:
        doc = _nlp(q)
        candidates = [
            token.lemma_.lower().strip()
            for token in doc
            if not token.is_stop and token.pos_ in ("NOUN", "PROPN", "ADJ")
        ]

        # Deduplicate while preserving order
        out, seen = [], set()
        for t in candidates:
            if t not in seen:
                out.append(t)
                seen.add(t)
            if len(out) >= top_k:
                break
        return out

    # Fallback keyword extraction
    tokens = [t.lower() for t in q.split() if len(t) > 2]
    out, seen = [], set()
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= top_k:
            break
    return out


# ------------------------------------------------------------
# Safe, Robust Extraction Utility for City/Province
# ------------------------------------------------------------
def safe_get(value):
    """
    Safely extracts a scalar string from:
      • Pandas Series
      • list/tuple
      • numpy array
      • scalar values
      • missing/None values
    Required because deployed and local environments sometimes
    return different types for the same dataframe cell.
    """
    import pandas as pd
    import numpy as np

    # Pandas Series → get first element
    if isinstance(value, pd.Series):
        return str(value.iloc[0]).strip() if not value.empty else ""

    # List/tuple/array → use first entry
    if isinstance(value, (list, tuple, np.ndarray)):
        return str(value[0]).strip() if len(value) > 0 else ""

    # None or NaN
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""

    # Default case (string/number)
    return str(value).strip()


# ------------------------------------------------------------
# City & Province Extraction from Query
# ------------------------------------------------------------
def extract_city_from_query(query: str, cities: List[str]) -> Optional[str]:
    """
    Detects if the user explicitly mentioned a known city.
    """
    if not query:
        return None
    q = str(query).lower()
    for city in cities:
        if isinstance(city, str) and city.lower() in q:
            return city
    return None


def extract_province_from_query(query: str, provinces: List[str]) -> Optional[str]:
    """
    Detects if the user mentioned a province in the query.
    """
    if not query:
        return None
    q = str(query).lower()
    for prov in provinces:
        if isinstance(prov, str) and prov.lower() in q:
            return prov
    return None


# ------------------------------------------------------------
# Price Preference Extraction
# ------------------------------------------------------------
def extract_price_from_query(query: str) -> Optional[str]:
    """
    Maps natural language phrases into standardized price categories.
    e.g., "cheap", "budget" → "Low Cost"
    """
    q = str(query).lower()
    if any(w in q for w in ["cheap", "budget", "affordable", "low price", "low-cost", "low cost"]):
        return "Low Cost"
    if any(w in q for w in ["moderate", "not too expensive", "reasonable", "medium"]):
        return "Medium Cost"
    if any(w in q for w in ["expensive", "fine dining", "luxury", "upscale", "high-end", "high cost"]):
        return "High Cost"
    return None


# ------------------------------------------------------------
# Similarity Helper (Cosine Similarity)
# ------------------------------------------------------------
def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Computes cosine similarity between two embedding vectors.
    """
    if a is None or b is None:
        return 0.0
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


# ------------------------------------------------------------
# Keyword Boost
# ------------------------------------------------------------
def keyword_boost(name: str, df: pd.DataFrame, keywords: List[str]) -> float:
    """
    Measures keyword match density for a given location.
    Higher boosts indicate better alignment with user intent.
    """
    if not keywords:
        return 0.0

    text = (
        " ".join(df[df["Location Name"] == name]["clean_text"].dropna().astype(str).tolist()).lower()
        if "clean_text" in df.columns
        else " ".join(df[df["Location Name"] == name]["Review Text"].dropna().astype(str).tolist()).lower()
    )

    if not text:
        return 0.0

    matches = sum(1 for k in keywords if k.lower() in text)
    return matches / max(1, len(keywords))


# ------------------------------------------------------------
# Hybrid Score (Semantic + Sentiment + Rating + Price + Quality)
# ------------------------------------------------------------
def hybrid_score(
    q_vec: np.ndarray,
    name: str,
    mode: str,
    df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    sentiment: Dict[str, float],
    rating: Dict[str, float],
    hotel_quality: Optional[Dict[str, float]],
    keywords: List[str],
    price_pref: Optional[str]
) -> float:
    """
    Computes the overall recommendation score by combining:
      • Semantic similarity
      • Keyword match signal
      • Sentiment score
      • Rating score (or hotel quality)
      • Price alignment penalty/bonus
    """
    if name not in embeddings:
        return -1e6  # effectively exclude from ranking

    embed_sim = similarity(q_vec, embeddings[name])
    kw = keyword_boost(name, df, keywords)
    sent = float(sentiment.get(name, 0.0))
    base_rating = float(rating.get(name, 0.0)) / 5.0 if rating.get(name) is not None else 0.0

    # Extract price (if available)
    try:
        rest_price = df[df["Location Name"] == name]["Location Price Range"].iloc[0]
    except Exception:
        rest_price = None

    # Reward or penalize based on price match
    price_boost = 0.0
    if price_pref:
        price_boost = 1.0 if rest_price == price_pref else -0.5

    # Hotel-specific quality score
    if mode == "Hotel" and hotel_quality:
        quality_score = float(hotel_quality.get(name, 0.0)) / 5.0
    else:
        quality_score = base_rating

    # Composite weighted score
    score = (
        embed_sim * 2.0
        + kw * 1.0
        + sent * 1.0
        + quality_score * 1.0
        + price_boost
    )
    return float(score)


# ------------------------------------------------------------
# Main Recommendation Pipeline
# ------------------------------------------------------------
def recommend_from_query(
    query: str,
    df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    sentiment: Dict[str, float],
    rating: Dict[str, float],
    cities: List[str],
    provinces: List[str],
    mode: str = "Restaurant",
    hotel_quality: Optional[Dict[str, float]] = None,
    top_n: int = 5
) -> List[Tuple[str, float]]:
    """
    Processes a user query and returns the top-N matching locations.
    Steps:
      1. Extract keywords, price preference, city/province constraints
      2. Compute semantic encoding of user query
      3. Iterate through all locations and:
           • Filter by city/province if requested
           • Score using hybrid model
      4. Return highest-ranked results
    """

    keywords = extract_keywords_from_query(query)
    price_pref = extract_price_from_query(query)
    q_vec = model.encode(query)

    # Detect explicit city/province in query
    user_city = extract_city_from_query(query, cities)
    user_province = extract_province_from_query(query, provinces)

    scored = []

    for name in embeddings.keys():

        # Retrieve location row
        row_df = df[df["Location Name"] == name].head(1)
        if row_df.empty:
            continue

        # Convert mini-DataFrame → Series
        row = row_df.iloc[0]

        # Robust city/province extraction
        rest_city = safe_get(row.get("Town/City"))
        rest_province = safe_get(row.get("Province"))

        print("CITY:", repr(rest_city), type(rest_city))
        print("PROVINCE:", repr(rest_province), type(rest_province))

        # Apply city/province filter when explicitly requested
        if user_city and rest_city.lower() != user_city.lower():
            continue
        if not user_city and user_province and rest_province.lower() != user_province.lower():
            continue

        # Compute hybrid score
        score = hybrid_score(
            q_vec=q_vec,
            name=name,
            mode=mode,
            df=df,
            embeddings=embeddings,
            sentiment=sentiment,
            rating=rating,
            hotel_quality=hotel_quality,
            keywords=keywords,
            price_pref=price_pref
        )

        scored.append((name, float(score)))

    # Rank by score (descending) and return top-N
    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
