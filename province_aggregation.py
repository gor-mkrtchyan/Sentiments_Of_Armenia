# province_aggregation.py
import pandas as pd
import numpy as np
from collections import Counter

def compute_province_stats(df):
    """
    Computes aggregated statistics per province.
    Handles numeric conversion for review ratings.
    Supports sentiment in [-1, 0, 1].
    """

    df = df.copy()

    # Clean Province / City / Restaurant names
    df['Province'] = df['Province'].fillna("Unknown")
    df['Location Name'] = df['Location Name'].fillna("Unknown")
    df['Town/City'] = df['Town/City'].fillna("Unknown")

    # -------------------------
    # 1. Ensure sentiment_score exists
    # -------------------------
    if 'sentiment_score' not in df.columns:
        raise ValueError("DataFrame must include 'sentiment_score' column.")

    # -------------------------
    # 2. Convert Review Rating → numeric
    # -------------------------
    df['Review Rating'] = pd.to_numeric(df['Review Rating'], errors='coerce')

    # -------------------------
    # 3. Group-level aggregation
    # -------------------------
    grouped = df.groupby('Province').agg(
        restaurant_count=('Location Name', lambda x: x.nunique()),
        city_count=('Town/City', lambda x: x.nunique()),
        review_count=('Review Text', 'count'),
        avg_rating=('Review Rating', 'mean'),
        avg_sentiment=('sentiment_score', 'mean')
    ).reset_index()

    # Replace NaN ratings with 0 for safety
    grouped['avg_rating'] = grouped['avg_rating'].fillna(0)
    grouped['avg_sentiment'] = grouped['avg_sentiment'].fillna(0)

    # -------------------------
    # 4. Sentiment color mapping [-1 → red → neutral → green]
    # -------------------------
    def sentiment_to_rgb(score):
        """
        score in [-1, 1]
        -1 = red
         0 = yellow
         1 = green
        """

        # Normalize score from [-1,1] → [0,1]
        s = (score + 1) / 2

        if s <= 0.5:
            # red → yellow transition
            t = s / 0.5
            r = int(220 + (240 - 220) * t)   # 220 → 240
            g = int(50 + (200 - 50) * t)     # 50 → 200
            b = int(50 + (60 - 50) * t)      # 50 → 60
        else:
            # yellow → green transition
            t = (s - 0.5) / 0.5
            r = int(240 + (50 - 240) * t)    # 240 → 50
            g = int(200 + (180 - 200) * t)   # 200 → 180
            b = int(60 + (80 - 60) * t)      # 60 → 80

        return [r, g, b]

    grouped['sentiment_color'] = grouped['avg_sentiment'].apply(sentiment_to_rgb)

    # Final rounding
    grouped['avg_rating'] = grouped['avg_rating'].round(2)
    grouped['avg_sentiment'] = grouped['avg_sentiment'].round(3)

    return grouped