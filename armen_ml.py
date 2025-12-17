# armen_ml.py
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
# Embedding model (exposed as `model`)
# ------------------------------------------------------------
# If loading is slow, Streamlit will cache or you can lazy-load.
model = SentenceTransformer("all-MiniLM-L6-v2")


# ------------------------------------------------------------
# Helper: build embeddings for a dataframe and kind
# kind: "Restaurant" or "Hotel"
# ------------------------------------------------------------
def build_embeddings(df: pd.DataFrame, kind: str = "Restaurant") -> Dict[str, np.ndarray]:
    """
    Build per-location embeddings for either restaurants or hotels.
    Returns dict: {location_name: vector}
    """
    if "Location Name" not in df.columns:
        raise KeyError("DataFrame must include 'Location Name' column")

    locations = df["Location Name"].dropna().unique().tolist()
    reps = []
    keys = []

    for loc in locations:
        rows = df[df["Location Name"] == loc]
        # Prefer Representation column
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

        if not rep_text or str(rep_text).strip().lower() in ("none", "nan", ""):
            # Prefer clean_text if present, else Review Text
            if "clean_text" in rows.columns:
                rep_text = " ".join(rows["clean_text"].dropna().astype(str).tolist())
            else:
                rep_text = " ".join(rows["Review Text"].dropna().astype(str).tolist())

        if not rep_text:
            rep_text = loc

        reps.append(rep_text)
        keys.append(loc)

    # encode in batch
    enc = model.encode(reps, convert_to_numpy=True)
    embeddings = {keys[i]: np.array(enc[i], dtype=float) for i in range(len(keys))}
    logger.info("Built embeddings (%s) for %d locations. dim=%s", kind, len(embeddings),
                next(iter(embeddings.values())).shape if embeddings else None)
    return embeddings


# ------------------------------------------------------------
# Sentiment & Rating computation helpers
# ------------------------------------------------------------
def compute_sentiment(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute mean sentiment_score per Location Name.
    Expects 'sentiment_score' present (numeric).
    """
    if "sentiment_score" not in df.columns:
        raise KeyError("DataFrame must contain 'sentiment_score' column. Call sentiment_model.apply_sentiment first.")
    df2 = df.copy()
    df2["sentiment_score"] = pd.to_numeric(df2["sentiment_score"], errors="coerce").fillna(0.0)
    return df2.groupby("Location Name")["sentiment_score"].mean().to_dict()


def compute_rating(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute mean Review Rating per Location Name.
    """
    if "Review Rating" not in df.columns:
        # attempt alternate column names
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
# Hotel quality computation (Option B weights)
# ------------------------------------------------------------
def compute_hotel_quality(df: pd.DataFrame) -> Dict[str, float]:
    """
    Weighted hotel quality score per Location Name using Option B weights:
        Cleanliness: 0.35
        Service:     0.25
        Location:    0.20
        Rooms:       0.10
        Value:       0.05
        Sleep:       0.05
    
    Only these columns are considered; all other columns are ignored.
    """

    COLS = {
        "Hotel Cleanliness Rating": 0.35,
        "Hotel Service Rating":     0.25,
        "Hotel Location Rating":    0.20,
        "Hotel Rooms Rating":       0.10,
        "Hotel Value Rating":       0.05,
        "Hotel Sleep Quality Rating": 0.05
    }

    # Check if any hotel rating columns exist
    existing_cols = [c for c in COLS.keys() if c in df.columns]

    if not existing_cols:
        # Fallback: use Review Rating mean if hotel detail columns aren't present
        logging.warning("No hotel quality columns found — falling back to Review Rating.")
        return compute_rating(df)

    df2 = df.copy()

    # Convert only the rating columns to numeric
    for col in COLS:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors="coerce").fillna(0.0)
        else:
            df2[col] = 0.0  # missing → treat as 0

    def weighted_quality(group):
        """
        group is a dataframe of all rows for one hotel.
        We compute weighted average of the numeric columns.
        """
        # Compute the mean ONLY for hotel columns
        means = {col: group[col].mean() for col in COLS}
        total = sum(means[col] * weight for col, weight in COLS.items())
        return total  # Already weighted (0–5 scale)

    # Apply per hotel
    result = df2.groupby("Location Name").apply(weighted_quality)

    return result.to_dict()



# ------------------------------------------------------------
# Keyword & City/Province extraction (spaCy fallback)
# ------------------------------------------------------------
try:
    _nlp = spacy.load("en_core_web_sm")
except Exception:
    _nlp = None


def extract_keywords_from_query(query: str, top_k: int = 8) -> List[str]:
    if not query:
        return []
    q = str(query)
    # try spaCy if available
    if _nlp:
        doc = _nlp(q)
        candidates = []
        for token in doc:
            if token.is_stop or token.is_punct or token.is_space:
                continue
            if token.pos_ in ("NOUN", "PROPN", "ADJ"):
                lemma = token.lemma_.lower().strip()
                if lemma:
                    candidates.append(lemma)
        # dedupe preserve order
        out = []
        seen = set()
        for t in candidates:
            if t not in seen:
                out.append(t)
                seen.add(t)
            if len(out) >= top_k:
                break
        return out
    else:
        # naive fallback tokenization
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
    if not query:
        return None
    q = str(query).lower()
    for city in cities:
        if not isinstance(city, str):
            continue
        if city.lower() in q:
            return city
    return None


def extract_province_from_query(query: str, provinces: List[str]) -> Optional[str]:
    if not query:
        return None
    q = str(query).lower()
    for prov in provinces:
        if not isinstance(prov, str):
            continue
        if prov.lower() in q:
            return prov
    return None


# ------------------------------------------------------------
# Price extraction (normalized)
# ------------------------------------------------------------
def extract_price_from_query(query: str) -> Optional[str]:
    q = str(query).lower()
    if any(w in q for w in ["cheap", "budget", "affordable", "low price", "low-cost", "low cost"]):
        return "Low Cost"
    if any(w in q for w in ["moderate", "not too expensive", "reasonable", "medium"]):
        return "Medium Cost"
    if any(w in q for w in ["expensive", "fine dining", "luxury", "upscale", "high-end", "high cost"]):
        return "High Cost"
    return None


# ------------------------------------------------------------
# Similarity helper
# ------------------------------------------------------------
def similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    return float(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0])


# ------------------------------------------------------------
# Keyword boost for either restaurants or hotels
# ------------------------------------------------------------
def keyword_boost(name: str, df: pd.DataFrame, keywords: List[str]) -> float:
    if not keywords:
        return 0.0
    if "clean_text" in df.columns:
        text = " ".join(df[df["Location Name"] == name]["clean_text"].dropna().astype(str).tolist()).lower()
    else:
        text = " ".join(df[df["Location Name"] == name]["Review Text"].dropna().astype(str).tolist()).lower()
    if not text:
        return 0.0
    matches = sum(1 for k in keywords if k.lower() in text)
    return matches / max(1, len(keywords))


# ------------------------------------------------------------
# Hybrid scoring (restaurant OR hotel mode)
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
    mode: "Restaurant" or "Hotel"
    hotel_quality: dict or None. If mode == "Hotel", this should be provided.
    """
    if name not in embeddings:
        return -1e6

    embed_sim = similarity(q_vec, embeddings[name])
    kw = keyword_boost(name, df, keywords)
    sent = float(sentiment.get(name, 0.0))
    base_rating = float(rating.get(name, 0.0)) / 5.0 if rating.get(name) is not None else 0.0

    # price
    try:
        rest_price = df[df["Location Name"] == name]["Location Price Range"].iloc[0]
    except Exception:
        rest_price = None

    price_boost = 0.0
    if price_pref:
        if rest_price == price_pref:
            price_boost = 1.0
        else:
            price_boost = -0.5

    # hotel-specific quality
    quality_score = 0.0
    if mode == "Hotel" and hotel_quality:
        quality_score = float(hotel_quality.get(name, 0.0)) / 5.0 if hotel_quality.get(name) is not None else 0.0
        # scale quality to [0,1] assuming ratings are 0-5; if hotel_quality already 0-5 it's fine
    else:
        # for restaurants, use base_rating as quality proxy
        quality_score = base_rating

    # final weighted
    # embedding strong weight, then keyword, sentiment, quality/rating
    score = embed_sim * 2.0 + kw * 1.0 + sent * 1.0 + quality_score * 1.0 + price_boost
    return float(score)


# ------------------------------------------------------------
# Main recommend_from_query (unified)
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
    Runs the recommender using the provided embeddings/dicts.
    Returns list of tuples: (Location Name, score)
    """

    keywords = extract_keywords_from_query(query)
    price_pref = extract_price_from_query(query)
    q_vec = model.encode(query)

    user_city = extract_city_from_query(query, cities)
    user_province = extract_province_from_query(query, provinces)

    scored = []

    for name in embeddings.keys():
        # fetch row as DataFrame
        row_df = df[df["Location Name"] == name].head(1)
        if row_df.empty:
            continue

        # convert to Series
        row = row_df.iloc[0]

        # City
        rest_city = str(row.get("Town/City", "")).strip()

        # Province
        rest_province = str(row.get("Province", "")).strip()

        print("CITY:", repr(rest_city), type(rest_city))
        print("PROVINCE:", repr(rest_province), type(rest_province))

        # city precedence
        if user_city:
            if rest_city.lower() != user_city.lower():
                continue
        elif user_province:
            if rest_province.lower() != user_province.lower():
                continue

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


    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]
    return scored_sorted
