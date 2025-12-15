# province_aggregation.py
"""
Province-Level Aggregation Module
---------------------------------

This module computes aggregated statistics for each province in Armenia
based on restaurant or hotel review data.

Output includes:
- Number of unique restaurants/hotels per province
- Number of unique cities in each province
- Total number of reviews
- Average rating
- Average sentiment (from -1 to +1)
- Sentiment color mapping (RGB) for visualization on a folium map

The output DataFrame is used directly by the Streamlit application to
colorize provinces and populate summary cards and map tooltips.

Assumptions:
- DataFrame contains at least:
    'Province', 'Town/City', 'Location Name', 'Review Rating', 'sentiment_score'
- sentiment_score ∈ {-1, 0, +1} but averages may be fractional.
"""

import pandas as pd
import numpy as np
from collections import Counter


def compute_province_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute province-level aggregated statistics from review data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset of restaurant or hotel reviews.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing aggregated statistics per province:
            - Province
            - restaurant_count (unique Location Names)
            - city_count (unique Town/City)
            - review_count (# of reviews)
            - avg_rating (0–5)
            - avg_sentiment (-1 to +1)
            - sentiment_color ([R,G,B] for map colorization)
    """

    # Create a defensive copy
    df = df.copy()

    # ------------------------------------------------------------
    # 1. Clean up key columns and ensure no missing identifiers
    # ------------------------------------------------------------
    df["Province"] = df["Province"].fillna("Unknown")
    df["Location Name"] = df["Location Name"].fillna("Unknown")
    df["Town/City"] = df["Town/City"].fillna("Unknown")

    # ------------------------------------------------------------
    # 2. Validate that sentiment_score column exists
    # ------------------------------------------------------------
    if "sentiment_score" not in df.columns:
        raise ValueError("DataFrame must include 'sentiment_score' column.")

    # ------------------------------------------------------------
    # 3. Convert Review Rating → numeric for safe averaging
    # ------------------------------------------------------------
    df["Review Rating"] = pd.to_numeric(df["Review Rating"], errors="coerce")

    # ------------------------------------------------------------
    # 4. Aggregate province-level metrics
    # ------------------------------------------------------------
    grouped = (
        df.groupby("Province")
        .agg(
            restaurant_count=("Location Name", lambda x: x.nunique()),
            city_count=("Town/City", lambda x: x.nunique()),
            review_count=("Review Text", "count"),
            avg_rating=("Review Rating", "mean"),
            avg_sentiment=("sentiment_score", "mean"),
        )
        .reset_index()
    )

    # Replace NaNs for robustness
    grouped["avg_rating"] = grouped["avg_rating"].fillna(0)
    grouped["avg_sentiment"] = grouped["avg_sentiment"].fillna(0)

    # ------------------------------------------------------------
    # 5. Map sentiment score → RGB color for the folium choropleth
    #
    #    score ∈ [-1,1]
    #      -1 → Red
    #       0 → Yellow
    #       1 → Green
    # ------------------------------------------------------------
    def sentiment_to_rgb(score: float) -> list:
        """
        Convert sentiment value (-1 to +1) into a smooth red→yellow→green gradient.

        Negative sentiment → red  
        Neutral sentiment → yellow  
        Positive sentiment → green
        """
        # Normalize score: [-1,1] → [0,1]
        s = (score + 1) / 2

        if s <= 0.5:
            # Red → Yellow blend
            t = s / 0.5
            r = int(220 + (240 - 220) * t)   # 220 → 240
            g = int(50 + (200 - 50) * t)     # 50  → 200
            b = int(50 + (60 - 50) * t)      # 50  → 60
        else:
            # Yellow → Green blend
            t = (s - 0.5) / 0.5
            r = int(240 + (50 - 240) * t)    # 240 → 50
            g = int(200 + (180 - 200) * t)   # 200 → 180
            b = int(60 + (80 - 60) * t)      # 60  → 80

        return [r, g, b]

    grouped["sentiment_color"] = grouped["avg_sentiment"].apply(sentiment_to_rgb)

    # ------------------------------------------------------------
    # 6. Final rounding for clean display in Streamlit summary cards
    # ------------------------------------------------------------
    grouped["avg_rating"] = grouped["avg_rating"].round(2)
    grouped["avg_sentiment"] = grouped["avg_sentiment"].round(3)

    return grouped
