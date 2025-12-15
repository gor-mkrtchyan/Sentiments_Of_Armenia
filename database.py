# database.py
"""
Database Access Layer for Armenia Restaurants & Hotels Explorer
--------------------------------------------------------------

This module provides a clean, centralized interface for retrieving
restaurant and hotel review data from a PostgreSQL database.

Responsibilities:
- Load environment variables for secure DB credentials
- Establish a PostgreSQL connection via psycopg2
- Provide two SQL queries:
      RESTAURANT_SQL  → fetch restaurant reviews
      HOTEL_SQL        → fetch hotel reviews
- Expose a single function:
      fetch_reviews(mode="Restaurant" | "Hotel")

The returned data is used by the ML engine, sentiment analysis
pipeline, province aggregation logic, and the Streamlit application.
"""

import os
from dotenv import load_dotenv
import psycopg2
import pandas as pd

# ------------------------------------------------------------
# Load environment variables from .env (local development)
# Deployment platforms (Render, Streamlit Cloud, Railway)
# will inject these via environment settings instead.
# ------------------------------------------------------------
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_SSLMODE = os.getenv("DB_SSLMODE", "require")


# ------------------------------------------------------------
# SQL Queries
# These queries join:
#   - reviews table
#   - locations (Restaurant/Hotel)
#   - geolocation
#   - app_users
#   - restaurant_review_details or hotel_review_details
#
# They return clean, ready-to-use DataFrames for ML + UI.
# ------------------------------------------------------------

RESTAURANT_SQL = """
SELECT
    g.province AS "Province",
    g.town_city AS "Town/City",
    l.location_name AS "Location Name",
    l.location_rating AS "Location Rating",
    l.location_num_reviews AS "Location # of Reviews",
    l.location_tag AS "Location Tag",
    l.location_price_range AS "Location Price Range",
    u.user_name AS "User Name",
    u.user_from AS "User From",
    r.review_rating AS "Review Rating",
    r.review_subject AS "Review Subject",
    r.review_date AS "Review Date",
    r.review_type AS "Type",
    r.review_text AS "Review Text",
    rd.value_rating AS "Restaurant Value Rating",
    rd.service_rating AS "Restaurant Service Rating",
    rd.food_rating AS "Restaurant Food Rating",
    rd.atmosphere_rating AS "Restaurant Atmosphere Rating"
FROM reviews r
JOIN locations l ON r.location_id = l.location_id
JOIN geolocation g ON l.geolocation_id = g.geolocation_id
JOIN app_users u ON r.user_id = u.user_id
LEFT JOIN restaurant_review_details rd ON r.review_id = rd.review_id
WHERE l.location_type = 'Restaurant';
"""

HOTEL_SQL = """
SELECT
    g.province AS "Province",
    g.town_city AS "Town/City",
    l.location_name AS "Location Name",
    l.location_rating AS "Location Rating",
    l.location_num_reviews AS "Location # of Reviews",
    l.location_tag AS "Location Tag",
    l.location_price_range AS "Location Price Range",
    u.user_name AS "User Name",
    u.user_from AS "User From",
    r.review_rating AS "Review Rating",
    r.review_subject AS "Review Subject",
    r.review_date AS "Review Date",
    r.review_type AS "Type",
    r.review_text AS "Review Text",
    hd.value_rating AS "Hotel Value Rating",
    hd.rooms_rating AS "Hotel Rooms Rating",
    hd.location_rating AS "Hotel Location Rating",
    hd.cleanliness_rating AS "Hotel Cleanliness Rating",
    hd.service_rating AS "Hotel Service Rating",
    hd.sleep_quality_rating AS "Hotel Sleep Quality Rating"
FROM reviews r
JOIN locations l ON r.location_id = l.location_id
JOIN geolocation g ON l.geolocation_id = g.geolocation_id
JOIN app_users u ON r.user_id = u.user_id
LEFT JOIN hotel_review_details hd ON r.review_id = hd.review_id
WHERE l.location_type = 'Hotel';
"""


# ------------------------------------------------------------
# Database Connection Helper
# ------------------------------------------------------------
def get_connection():
    """
    Create and return a new PostgreSQL database connection.

    Uses environment variables:
        DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_SSLMODE

    Returns
    -------
    psycopg2.extensions.connection
        Open psycopg2 connection object.
    """
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        sslmode=DB_SSLMODE
    )
    return conn


# ------------------------------------------------------------
# Fetch Review Data (Restaurants OR Hotels)
# ------------------------------------------------------------
def fetch_reviews(mode: str = "Restaurant") -> pd.DataFrame:
    """
    Fetch reviews from the PostgreSQL database.

    Parameters
    ----------
    mode : str
        "Restaurant" or "Hotel".
        Determines which SQL query is executed.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing all joined review records
        relevant to the requested mode.

    Raises
    ------
    ValueError
        If `mode` is not one of the allowed values.
    """
    if mode not in ("Restaurant", "Hotel"):
        raise ValueError("mode must be 'Restaurant' or 'Hotel'")

    sql = RESTAURANT_SQL if mode == "Restaurant" else HOTEL_SQL

    conn = get_connection()
    try:
        df = pd.read_sql(sql, conn)
    finally:
        conn.close()

    return df
