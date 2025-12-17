# armen_ml.py
"""
Armen ML Engine — Core Recommendation Logic
===========================================

This module powers the ML side of the Armenia Restaurants & Hotels Explorer.

It provides:
    - Embedding generation for restaurants & hotels (SentenceTransformer)
    - Sentiment averaging per location
    - Rating averaging per location
    - Weighted hotel quality scores (Option B)
    - Keyword extraction (spaCy or fallback)
    - City & province detection from user queries
    - Normalized price preference extraction
    - Hybrid scoring function (semantic + sentiment + quality + keywords)
    - Unified recommend_from_query() for both Restaurants & Hotels

The output of this module is consumed by:
    - The Streamlit ML recommendation UI
    - Armen ML mode (offline assistant)
    - Armen Gemini mode (grounded recommendation candidates)
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Global Embedding Model
# ============================================================

# SentenceTransformer is loaded once per session.
# Streamlit caching outside this module will keep it memory-efficient.
model = SentenceTransformer("all-MiniLM-L6-v2")


# ============================================================
# Embedding Builder — Restaurants & Hotels
# ============================================================

def build_embeddings(df: pd.DataFrame, kind: str = "Restaurant") -> Dict[str, np.ndarray]:
    """
    Build embeddings for each unique location ("Location Name").

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    kind : str
        "Restaurant" or "Hotel" — only for logging purposes.

    Returns
    -------
    dict
        Mapping: { "Location Name": embedding_vector }
    """

    if "Location Name" not in df.columns:
        raise KeyError("DataFrame must include 'Location Name' column")

    locations = df["Location Name"].dropna().unique().tolist()

    reps = []
    keys = []

    for loc in locations:
        rows = df[df["Location Name"] == loc]

        # Preferred source: BERTopic “Representation” column
        rep_text = None
        if "Representation" in df.columns:
            try:
                rep_val = rows["Representation"].dropna().iloc[0]
                if isinstance(rep_val, (list, tuple)):
                    rep_text = " ".join(map(str, rep_val))
                else:
                    rep_text = str(rep_val)
            except Exception:
                rep_text = None

        # Fallback → aggregated clean_text
        if not rep_text or rep_text.strip().lower() in ("none", "nan", ""):
            if "clean_text" in rows.columns:
                rep_text = " ".join(rows["clean_text"].dropna().astype(str).tolist())
            else:
                # Final fallback → raw review text
                rep_text = " ".join(rows["Review Text"].dropna().astype(str).tolist())

        # Extreme fallback → location name
        if not rep_text:
            rep_text = loc

        reps.append(rep_text)
        keys.append(loc)

    # Batch encode all representations
    enc = model.encode(reps, convert_to_numpy=True)

    embeddings = {
        keys[i]: np.array(enc[i], dtype=float)
        for i in range(len(keys))
    }

    # Log summary
    dim = next(iter(embeddings.values())).shape if embeddings else None
    logger.info("Built %s embeddings for %d locations (dim=%s)",
                kind, len(embeddings), dim)

    return embeddings


# ============================================================
# Sentiment & Rating Computation
# ============================================================

def compute_sentiment(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute mean sentiment_score per location.

    sentiment_score ∈ {-1,0,+1} but averaged to float [-1,+1].

    Returns { "Location Name": sentiment_value }
    """
    if "sentiment_score" not in df.columns:
        raise KeyError("Missing 'sentiment_score'. Run apply_sentiment first.")
    
    df2 = df.copy()
    df2["sentiment_score"] = pd.to_numeric(df2["sentiment_score"], errors="coerce").fillna(0.0)

    return df2.groupby("Location Name")["sentiment_score"].mean().to_dict()


def compute_rating(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute mean numeric Review Rating per location.

    Attempts to normalize column names if mislabeled.
    """
    if "Review Rating" not in df.columns:
        for alt in ["review_rating", "rating", "Rating"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "Review Rating"})
                break
        else:
            raise KeyError("DataFrame must contain 'Review Rating' column.")

    df2 = df.copy()
    df2["Review Rating"] = pd.to_numeric(df2["Review Rating"], errors="coerce")

    grouped = df2.groupby("Location Name")["Review Rating"].mean().fillna(0.0)
    return grouped.to_dict()


# ============================================================
# Hotel Quality Score (Weighted)
# ============================================================

def compute_hotel_quality(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute weighted hotel quality scores using Option B weights:

        Cleanliness ........ 0.35
        Service ............ 0.25
        Location ........... 0.20
        Rooms .............. 0.10
        Value .............. 0.05
        Sleep Quality ...... 0.05

    Only these columns are considered. Others are ignored.

    Returns
    -------
    dict
        { "Hotel Name": weighted_quality (0–5 range) }
    """

    COLS = {
        "Hotel Cleanliness Rating": 0.35,
        "Hotel Service Rating":     0.25,
        "Hotel Location Rating":    0.20,
        "Hotel Rooms Rating":       0.10,
        "Hotel Value Rating":       0.05,
        "Hotel Sleep Quality Rating": 0.05
    }

    # Verify presence of at least one hotel rating column
    existing = [c for c in COLS if c in df.columns]
    if not existing:
        logging.warning("Hotel detail columns missing — falling back to Review Rating.")
        return compute_rating(df)

    df2 = df.copy()

    # Convert columns to numeric (missing → zero)
    for col in COLS:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0.0)
        else:
            df2[col] = 0.0

    # Weighted average per hotel
    def weighted_quality(group):
        means = {col: group[col].mean() for col in COLS}
        return sum(means[col] * weight for col, weight in COLS.items())

    result = df2.groupby("Location Name").apply(weighted_quality)
    return result.to_dict()


# ============================================================
# NLP Extraction (Keywords, City, Province, Price)
# ============================================================

# Try loading spaCy English model
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None  # fallback will be used


def extract_keywords_from_query(query: str, top_k: int = 8) -> List[str]:
    """
    Extract keywords (nouns, adjectives, proper nouns) from user query.
    Uses spaCy if available; otherwise uses simple token fallback.
    """
    if not query:
        return []

    q = str(query)

    if _nlp:
        doc = _nlp(q)
        candidates = [
            token.lemma_.lower().strip()
            for token in doc
            if not token.is_stop
            and not token.is_punct
            and token.pos_ in ("NOUN", "PROPN", "ADJ")
        ]

        # Deduplicate while preserving order
        seen = set()
        out = []
        for t in candidates:
            if t not in seen:
                out.append(t)
                seen.add(t)
            if len(out) >= top_k:
                break
        return out

    # Fallback if spaCy unavailable
    tokens = [t.lower() for t in q.split() if len(t) > 2]
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
        if len(out) >= top_k:
            break
    return out


def extract_city_from_query(query: str, cities: List[str]) -> Optional[str]:
    """
    Detect if the query explicitly references a known city.
    """
    if not query:
        return None
    q = query.lower()
    for city in cities:
        if isinstance(city, str) and city.lower() in q:
            return city
    return None


def extract_province_from_query(query: str, provinces: List[str]) -> Optional[str]:
    """
    Detect if the query explicitly references a province.
    """
    if not query:
        return None
    q = query.lower()
    for prov in provinces:
        if isinstance(prov, str) and prov.lower() in q:
            return prov
    return None


def extract_price_from_query(query: str) -> Optional[str]:
    """
    Normalize price preference from natural text.
    """
    q = str(query).lower()
    if any(w in q for w in ["cheap", "budget", "affordable", "low price", "low cost", "low-cost"]):
        return "Low Cost"
    if any(w in q for w in ["moderate", "reasonable", "not too expensive", "medium"]):
        return "Medium Cost"
    if any(w in q for w in ["expensive", "fine dining", "luxury", "upscale", "high-end"]):
        return "High Cost"
    return None


# ============================================================
# Similarity + Keyword Boost
# ============================================================

def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between embedding vectors.
    """
    if a is None or b is None:
        return 0.0
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


def keyword_boost(name: str, df: pd.DataFrame, keywords: List[str]) -> float:
    """
    Compute keyword matching ratio between query keywords
    and location's aggregated review text.
    """
    if not keywords:
        return 0.0

    if "clean_text" in df.columns:
        text = " ".join(
            df[df["Location Name"] == name]["clean_text"].dropna().astype(str).tolist()
        ).lower()
    else:
        text = " ".join(
            df[df["Location Name"] == name]["Review Text"].dropna().astype(str).tolist()
        ).lower()

    if not text:
        return 0.0

    matches = sum(1 for k in keywords if k.lower() in text)
    return matches / max(1, len(keywords))


# ============================================================
# Hybrid Scoring — Core of Armen ML Engine
# ============================================================

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
    price_pref: Optional[str],
) -> float:
    """
    Compute a final recommendation score for a given location.

    Components:
        2.0 * embedding similarity
        1.0 * keyword boost
        1.0 * sentiment mean
        1.0 * quality (restaurants: rating, hotels: weighted quality)
        ± price preference boost/penalty
    """

    # Missing embedding → invalid
    if name not in embeddings:
        return -1e6

    embed_sim = similarity(q_vec, embeddings[name])
    kw = keyword_boost(name, df, keywords)
    sent = float(sentiment.get(name, 0.0))

    # Restaurant base rating is 0–5 → normalize to [0,1]
    base_rating = float(rating.get(name, 0.0)) / 5.0 if rating.get(name) else 0.0

    # Price match penalty/boost
    try:
        loc_price = df[df["Location Name"] == name]["Location Price Range"].iloc[0]
    except Exception:
        loc_price = None

    price_boost = 1.0 if price_pref and loc_price == price_pref else 0.0
    if price_pref and loc_price != price_pref:
        price_boost = -0.5

    # Hotels get a special weighted quality metric
    if mode == "Hotel" and hotel_quality:
        quality = float(hotel_quality.get(name, 0.0)) / 5.0
    else:
        quality = base_rating

    # Final hybrid score
    score = (
        embed_sim * 2.0 +
        kw * 1.0 +
        sent * 1.0 +
        quality * 1.0 +
        price_boost
    )

    return float(score)


# ============================================================
# Unified Recommender — Hotels + Restaurants
# ============================================================

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
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Run full recommendation pipeline for a user query.

    Steps:
        1. Extract keywords, price preference
        2. Embed the query
        3. Detect city / province filter
        4. Score each location via hybrid_score
        5. Sort and return top-N

    Returns
    -------
    list of (name, score)
    """

    keywords = extract_keywords_from_query(query)
    price_pref = extract_price_from_query(query)

    # Encode the query once
    q_vec = model.encode(query)

    user_city = extract_city_from_query(query, cities)
    user_province = extract_province_from_query(query, provinces)

    scores = []

    for name in embeddings.keys():

        # Get minimal row for metadata checking
        row = df[df["Location Name"] == name].head(1)
        if row.empty:
            continue

        rest_city = str(row.get("Town/City", [""])[0])
        rest_province = str(row.get("Province", [""])[0])

        # City → highest priority
        if user_city:
            if rest_city.lower() != user_city.lower():
                continue

        # Province → secondary filter (only when city not provided)
        elif user_province:
            if rest_province.lower() != user_province.lower():
                continue

        # Score using the hybrid scoring system
        final_score = hybrid_score(
            q_vec=q_vec,
            name=name,
            mode=mode,
            df=df,
            embeddings=embeddings,
            sentiment=sentiment,
            rating=rating,
            hotel_quality=hotel_quality,
            keywords=keywords,
            price_pref=price_pref,
        )

        scores.append((name, final_score))

    # Sort by score descending
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    return scores
