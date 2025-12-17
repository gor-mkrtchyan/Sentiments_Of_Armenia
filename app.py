# =========================================================
# app.py — Armenia Explorer (Restaurants & Hotels)
# Final Version with UI Improvements, ML/Gemini Chatbot,
# Interactive Sentiment Map, Rich Analytics & Recommendation Engine
#
# NOTE:
#   This version contains detailed, professional commentary designed
#   for academic/capstone submission and industry-quality readability.
#   No functional changes have been made—only documentation added.
# =========================================================


import os
import json
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv

# ---------------------------------------------------------
# Internal Modules — Database, ML, NLP, Province Aggregation
# ---------------------------------------------------------
from database import fetch_reviews
import sentiment_model
from province_aggregation import compute_province_stats

# ---------------------------------------------------------
# ARMEN ML ENGINE (Offline Recommendation System)
# ---------------------------------------------------------
# - Semantic embeddings using SentenceTransformer
# - Sentiment, rating, quality computation
# - Keyword extraction, location extraction
# - Hybrid scoring model (semantic + sentiment + rating)
from armen_ml import (
    model,
    build_embeddings,
    compute_sentiment,
    compute_rating,
    compute_hotel_quality,
    extract_keywords_from_query,
    extract_price_from_query,
    extract_city_from_query,
    extract_province_from_query,
    recommend_from_query
)

# ---------------------------------------------------------
# ARMEN GEMINI ENGINE (Conversational Recommendation Mode)
# ---------------------------------------------------------
from armen_gemini import armen_gemini_response


# ---------------------------------------------------------
# Streamlit Initialization & Global Config
# ---------------------------------------------------------
load_dotenv()
st.set_page_config(layout="wide", page_title="Armenia Explorer (Restaurants & Hotels)")

# Path for region boundaries (used by Folium map)
GEOJSON_PATH = os.getenv("ARMENIA_GEOJSON_PATH", "armenia_provinces.geojson")

# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
# Stores:
#   - Chat history for each mode (restaurants / hotels)
#   - Currently selected province
#   - Last map click JSON dump for click-detection
#   - Trigger token for forcing map refresh
st.session_state.setdefault("restaurant_chat_history", [])
st.session_state.setdefault("hotel_chat_history", [])
st.session_state.setdefault("armen_last_query", "")
st.session_state.setdefault("selected_province", None)
st.session_state.setdefault("last_click_raw", "")
st.session_state.setdefault("trigger_rerun_token", 0)


# ---------------------------------------------------------
# Top Explore Mode Toggle (Restaurants vs Hotels)
# ---------------------------------------------------------
# Custom CSS styling for larger controls + consistent layout
st.markdown("""
<style>
.explore-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 18px;
}
.explore-label {
    font-size: 24px !important;
    font-weight: 700 !important;
    color: white !important;
}
.big-radio div[role='radiogroup'] > label {
    font-size: 18px !important;
}
</style>
""", unsafe_allow_html=True)

# UI for switching explore category
st.markdown("<div class='explore-bar'><span class='explore-label'>Explore:</span></div>", unsafe_allow_html=True)

explore_mode = st.radio(
    label="",                       # Hidden label (decorated above)
    options=["🍽️ Restaurants", "🏨 Hotels"],
    index=0,
    horizontal=True,
    key="explore_toggle"
)

# Normalize UI label → internal keyword
explore_mode = "Restaurants" if explore_mode.startswith("🍽️") else "Hotels"


# ---------------------------------------------------------
# Select the correct chat history based on selected mode
# ---------------------------------------------------------
active_history = (
    st.session_state["restaurant_chat_history"]
    if explore_mode == "Restaurants"
    else st.session_state["hotel_chat_history"]
)


# ---------------------------------------------------------
# Data Loaders (cached for fast reloads)
# ---------------------------------------------------------
@st.cache_data
def load_geojson(path):
    """Load GeoJSON file containing province boundaries."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_data(mode="Restaurants"):
    """
    Fetch all reviews from the database.
    Apply sentiment model.
    Compute aggregated province statistics.
    """
    db_mode = "Restaurant" if mode == "Restaurants" else "Hotel"
    df = fetch_reviews(db_mode)
    df = sentiment_model.apply_sentiment(df, "Review Text")
    stats = compute_province_stats(df)
    return df, stats


# ---------------------------------------------------------
# Load Combined Review Dataset + Province Statistics
# ---------------------------------------------------------
df, province_stats = load_data(mode=explore_mode)

# Ensure Review Text exists even if DB returns incomplete columns
if "Review Text" not in df.columns:
    df["Review Text"] = ""

# Create cleaned text for embeddings/sentiment fallback
try:
    df["clean_text"] = df["Review Text"].apply(sentiment_model.clean_text)
except:
    # Fallback regex cleans non-alphabetic characters
    import re
    df["clean_text"] = df["Review Text"].apply(
        lambda t: " ".join(re.sub(r"[^a-zA-Z\s]", "", str(t).lower()).split())
    )


# ---------------------------------------------------------
# Build ML Engine Components for Active Mode
# ---------------------------------------------------------
kind = explore_mode[:-1] if explore_mode.endswith("s") else explore_mode

# Embeddings for each location (semantic search)
embeddings = build_embeddings(df, kind=kind)

# Sentiment dictionary per location
sentiment = compute_sentiment(df)

# Rating dictionary per location
rating = compute_rating(df)

# Hotel quality scores (for hotel mode only)
hotel_quality = compute_hotel_quality(df) if explore_mode == "Hotels" else None

# Unique cities/provinces for location extraction (NLP)
cities_list = df["Town/City"].dropna().unique().tolist() if "Town/City" in df.columns else []
provinces_list = df["Province"].dropna().unique().tolist() if "Province" in df.columns else []


# ---------------------------------------------------------
# Load Armenia Province GeoJSON for Map Rendering
# ---------------------------------------------------------
geojson = load_geojson(GEOJSON_PATH)


# ---------------------------------------------------------
# Sentiment → Color Mapping Utility
# ---------------------------------------------------------
# Converts normalized sentiment score into a perceptually intuitive
# gradient (red → orange → green). Ensures visual alignment with legend.
def sentiment_to_color(score):
    if score is None:
        return (200, 200, 200)

    score = max(0, min(1, float(score)))  # Clamp value

    if score <= 0.5:
        # Negative → Neutral (red → orange)
        t = score / 0.5
        r = int(220 + (255 - 220) * t)
        g = int(60 + (165 - 60) * t)
        b = int(60 + (0 - 60) * t)
    else:
        # Neutral → Positive (orange → green)
        t = (score - 0.5) / 0.5
        r = int(255 + (80 - 255) * t)
        g = int(165 + (200 - 165) * t)
        b = int(0 + (80 - 0) * t)

    return (r, g, b)


# ---------------------------------------------------------
# Attach Computed Province Metrics to GeoJSON Features
# ---------------------------------------------------------
for feat in geojson["features"]:
    props = feat["properties"]
    province = props.get("shapeName", "").strip()

    # Lookup matching province row in aggregated stats
    match = (
        province_stats[province_stats["Province"].str.lower() == province.lower()]
        if "Province" in province_stats.columns else None
    )

    if match is not None and not match.empty:
        r = match.iloc[0]
        avg_sent = float(r.get("avg_sentiment", 0.0))

        # Attach aggregated metadata to feature properties
        props.update({
            "restaurant_count": int(r.get("restaurant_count", 0)),
            "review_count": int(r.get("review_count", 0)),
            "city_count": int(r.get("city_count", 0)),
            "avg_rating": float(r.get("avg_rating", 0.0)),
            "avg_sentiment": avg_sent,
            "sentiment_color": sentiment_to_color(avg_sent)
        })
    else:
        # Default values for provinces with no reviews
        props.update({
            "restaurant_count": 0,
            "review_count": 0,
            "city_count": 0,
            "avg_rating": 0.0,
            "avg_sentiment": 0.0,
            "sentiment_color": (200, 200, 200)
        })


# ---------------------------------------------------------
# Global CSS Styling — Cards, Chat Bubbles, Table Formatting
# ---------------------------------------------------------
st.markdown("""
<style>

.card {
    background: linear-gradient(135deg, #1c1c1c, #0e0e0e);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 0 16px rgba(0,0,0,0.45);
    margin-bottom: 15px;
    color: white;
    text-align: center;
}
.card h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 600;
    color: #eeeeee;
    text-align: center;
    margin-bottom: 6px;
}
.card p {
    margin: 5px 0 0;
    font-size: 30px;
    font-weight: bold;
    color: white;
    text-align: center;
}

/* Chat bubbles styled similar to WhatsApp for better UX */
.user-bubble {
    background: #075E54;
    color: white;
    padding: 10px 14px;
    border-radius: 16px;
    border-bottom-right-radius: 4px;
    margin: 8px 0;
    font-size: 15px;
    max-width: 80%;
    word-wrap: break-word;
}

.armen-bubble {
    background: #262626;
    color: #ffe49e;
    padding: 10px 14px;
    border-radius: 16px;
    border-bottom-left-radius: 4px;
    margin: 8px 0;
    font-size: 15px;
    max-width: 80%;
    word-wrap: break-word;
    margin-left:auto;
    text-align:left;
}

/* Reset button alignment */
.reset-button-container {
    display:flex;
    justify-content:flex-end;
    margin-top:-10px;
    margin-bottom:10px;
}

/* ML table style inside Armen responses */
.armen-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 15px;
    color: #ffe49e;
}
.armen-table th {
    background-color: #333;
    padding: 8px;
    text-align: left;
    border-bottom: 2px solid #555;
}
.armen-table td {
    padding: 8px;
    border-bottom: 1px solid #444;
}
.armen-table tr:hover {
    background-color: #222;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# UI — Page Title & Spacing
# ---------------------------------------------------------
st.title(f"Armenia — {explore_mode} Sentiment Map Explorer")
st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)


# =========================================================
# SUMMARY CARDS (Global Statistics)
# =========================================================
# These metrics summarize the entire dataset (country-wide)
card_col1, card_col2, card_col3, card_col4 = st.columns([1,1,1,1])

total_items = int(province_stats.get("restaurant_count", pd.Series([0])).sum())
total_reviews = int(province_stats.get("review_count", pd.Series([0])).sum())

weighted_sent = (
    (province_stats.get("avg_sentiment", pd.Series([0])) *
     province_stats.get("review_count", pd.Series([1]))).sum()
    / max(total_reviews, 1)
)

avg_rating = round(float(province_stats.get("avg_rating", pd.Series([0])).mean()), 2)

# Render dashboard cards
with card_col1:
    st.markdown(f"""
        <div class="card"><h2>Total {explore_mode}</h2><p>{total_items}</p></div>
    """, unsafe_allow_html=True)

with card_col2:
    st.markdown(f"""
        <div class="card"><h2>Total Reviews</h2><p>{total_reviews}</p></div>
    """, unsafe_allow_html=True)

with card_col3:
    st.markdown(f"""
        <div class="card"><h2>Weighted Sentiment</h2><p>{round(weighted_sent,3)}</p></div>
    """, unsafe_allow_html=True)

with card_col4:
    st.markdown(f"""
        <div class="card"><h2>Average Rating</h2><p>{avg_rating}</p></div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)


# =========================================================
# INTERACTIVE MAP — Province Sentiment Visualization
# =========================================================
st.subheader("Click a province on the map")

# Initialize Folium map (Dark Matter style)
m = folium.Map(
    location=[40.1, 44.5],
    zoom_start=7,
    tiles="CartoDB dark_matter"
)

# Keep map reference for JS-based click detection
m.get_root().script.add_child(folium.Element("window.map = this;"))

# Style provinces using sentiment-based color scale
def style_fn(f):
    r, g, b = f["properties"]["sentiment_color"]
    return {
        "fillColor": f"rgb({r},{g},{b})",
        "color": "white",
        "weight": 1,
        "fillOpacity": 0.87,
    }

# Province boundaries + tooltip + popup
folium.GeoJson(
    geojson,
    style_function=style_fn,
    tooltip=folium.GeoJsonTooltip(
        fields=["shapeName", "city_count", "restaurant_count",
                "review_count", "avg_rating", "avg_sentiment"],
        aliases=["Province", "Cities", f"{explore_mode}",
                 "Reviews", "Avg Rating", "Sentiment"]
    ),
    popup=folium.GeoJsonPopup(fields=["shapeName"])
).add_to(m)

# Custom legend for sentiment scale
legend_html = """
<div style="
    position: absolute;
    top: 10px; right: 10px;
    z-index: 9999;
    background-color: rgba(0,0,0,0.75);
    padding: 18px 20px;
    border-radius: 10px;
    color: white;
    font-size: 15px;
    width: 220px;
    box-shadow: 0 0 12px rgba(0,0,0,0.7);
">
    <b style="font-size: 17px;">Sentiment Scale</b><br>
    <div style="
        margin-top: 10px;
        width: 100%; height: 20px;
        border-radius: 5px;
        background: linear-gradient(to right,
            rgb(220,60,60),
            rgb(255,210,60),
            rgb(80,200,80)
        );
    "></div>
    <div style="display:flex; justify-content:space-between; margin-top: 5px; font-size: 13px;">
        <span>Negative</span><span>Neutral</span><span>Positive</span>
    </div>
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Render map in Streamlit
map_data = st_folium(
    m,
    width=None,
    height=650,
    key=f"map_{st.session_state['trigger_rerun_token']}"
)


# =========================================================
# PROVINCE CLICK DETECTION
# =========================================================
# Extract click event info from Folium → Streamlit → Session state
raw_json = json.dumps(map_data, sort_keys=True)

# Detect new click event by comparing previous serialized map state
if raw_json != st.session_state["last_click_raw"]:
    st.session_state["last_click_raw"] = raw_json

    clicked = None

    # Try extract from drawing layer
    if map_data.get("last_active_drawing"):
        clicked = map_data["last_active_drawing"]["properties"].get("shapeName")

    # Try popup (fallback)
    if not clicked and map_data.get("last_object_clicked_popup"):
        clicked = map_data["last_object_clicked_popup"].replace("Select", "").strip()

    # Try tooltip (last fallback)
    if not clicked and map_data.get("last_object_clicked_tooltip"):
        lines = [
            l.strip() for l in map_data["last_object_clicked_tooltip"].splitlines()
            if l.strip()
        ]
        if len(lines) >= 2:
            clicked = lines[1]

    # Store selected province & trigger map refresh
    if clicked:
        lookup = (
            {p.lower(): p for p in province_stats["Province"].unique()}
            if "Province" in province_stats.columns else {}
        )
        canonical = lookup.get(clicked.lower(), clicked)
        st.session_state["selected_province"] = canonical
        st.session_state["trigger_rerun_token"] += 1
        st.rerun()


# =========================================================
# TOP 5 LOCATIONS (Restaurants / Hotels) for Selected Province
# =========================================================
st.header(f"Top 5 {explore_mode} (Ranked by Rating & Reviews)")

prov = st.session_state.get("selected_province")

if prov:
    st.subheader(f"Province: {prov}")
    if "Province" not in df.columns:
        st.warning(f"No {explore_mode.lower()} found in this province.")
    else:
        # Filter reviews belonging to the selected province
        sel = df[df["Province"].str.lower() == prov.lower()].copy()

        if sel.empty:
            st.warning(f"No {explore_mode.lower()} found in this province.")
        else:
            # Convert relevant numeric fields
            sel["Review Rating"] = pd.to_numeric(sel["Review Rating"], errors="coerce")
            sel["sentiment_score"] = pd.to_numeric(sel["sentiment_score"], errors="coerce")

            # Compute top 5 ranked by rating + review count
            top5 = (
                sel.groupby("Location Name")
                .agg(
                    avg_rating=("Review Rating", "mean"),
                    reviews_count=("Review Text", "count"),
                    avg_sentiment=("sentiment_score", "mean")
                )
                .reset_index()
                .sort_values(["avg_rating", "reviews_count"], ascending=[False, False])
                .head(5)
            )

            top5.index = range(1, len(top5) + 1)

            # Human-friendly column names
            top5 = top5.rename(columns={
                "Location Name": "Location",
                "avg_rating": "Average Rating",
                "reviews_count": "Review Count",
                "avg_sentiment": "Average Sentiment"
            })

            # ----------------------------------------------
            # Render Top 5 Table Using Custom HTML + CSS
            # ----------------------------------------------
            import textwrap

            html = textwrap.dedent("""
            <style>
                .top5-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                    font-size: 15px;
                }
                .top5-table th {
                    padding: 10px;
                    background-color: #222;
                    color: #eee;
                    border-bottom: 2px solid #444;
                    text-align: center;
                }
                .top5-table th:first-child {
                    text-align: left;
                }
                .top5-table td {
                    padding: 8px;
                    border-bottom: 1px solid #333;
                    text-align: center;
                }
                .top5-table td:first-child {
                    text-align: left;
                }
            </style>

            <table class="top5-table">
                <tr>
                    <th>Location</th>
                    <th>Average Rating</th>
                    <th>Review Count</th>
                    <th>Average Sentiment</th>
                </tr>
            """)

            for idx, row in top5.iterrows():
                html += textwrap.dedent(f"""
                <tr>
                    <td>{row['Location']}</td>
                    <td>{row['Average Rating']:.2f}</td>
                    <td>{row['Review Count']}</td>
                    <td>{row['Average Sentiment']:.3f}</td>
                </tr>
                """)

            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)

else:
    st.info("Click a province above to explore its top locations.")


# =========================================================
# ARMEN CHATBOT UI — Dual-Mode Conversational Assistant
# =========================================================
# The chatbot provides:
#   - Semantic ML-based recommendations
#   - Conversational Gemini responses for richer dialogue
#   - Interactive WhatsApp-style chat interface
st.markdown("---")
st.header("🤖 Armen — Your Personal Guide")

# Reset chat history (per mode)
st.markdown("<div class='reset-button-container'>", unsafe_allow_html=True)

if st.button("🗑️ Reset Armen Chat", key="reset_chat_btn"):
    if explore_mode == "Restaurants":
        st.session_state["restaurant_chat_history"] = []
    else:
        st.session_state["hotel_chat_history"] = []
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

# Switch between ML and Gemini for recommendations
mode = st.radio(
    "Choose Armen mode:",
    ("Armen ML (Fast, Offline)", "Armen Gemini (Conversational AI)"),
    index=0,
    horizontal=True
)

# Chat container (history + input)
chat_container = st.container()

with chat_container:

    # Display chat history (bubbles)
    for role, msg in active_history:
        if role == "user":
            st.markdown(
                f"<div class='user-bubble'>🧑‍💬 <strong>You:</strong><br>{msg}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='armen-bubble'>🤖 <strong>Armen:</strong><br>{msg}</div>",
                unsafe_allow_html=True
            )

    # User input field
    user_query = st.text_input(
        f"Tell Armen what you’re looking for in {explore_mode}:",
        key="armen_input"
    )

    # On submit
    if st.button("Ask Armen", key="ask_armen"):
        if user_query.strip() == "":
            st.warning("Please type something.")
        else:
            # Store user message
            active_history.append(("user", user_query))

            # -------------------------------------------------
            # ML Recommendation Pipeline (Fast Offline Mode)
            # -------------------------------------------------
            def build_ml_response(query, top_n=5):
                # Compute recommended locations
                recs = recommend_from_query(
                    query=query,
                    df=df,
                    embeddings=embeddings,
                    sentiment=sentiment,
                    rating=rating,
                    cities=cities_list,
                    provinces=provinces_list,
                    mode=(explore_mode[:-1] if explore_mode.endswith("s") else explore_mode),
                    hotel_quality=hotel_quality,
                    top_n=top_n
                )

                # Handle no match scenarios gracefully
                if not recs:
                    city_hit = extract_city_from_query(query, cities_list)
                    prov_hit = extract_province_from_query(query, provinces_list)

                    if city_hit:
                        return f"Barev jan — I couldn't find any {explore_mode.lower()} in **{city_hit}**."
                    if prov_hit:
                        return f"Barev jan — I couldn't find any {explore_mode.lower()} in the **{prov_hit}** province."
                    return f"Barev jan — I couldn't find any {explore_mode.lower()} matching your request."

                # Build HTML table for results
                rows = []
                for name, score in recs:
                    row = df[df["Location Name"] == name].head(1)
                    if row.empty:
                        continue
                    row = row.iloc[0]

                    # Extract metadata
                    r_town = row.get("Town/City", "Unknown")
                    r_prov = row.get("Province", "Unknown")
                    r_price = row.get("Location Price Range", "Unknown")
                    r_rating = rating.get(name, 0.0)
                    r_sent = sentiment.get(name, 0.0)

                    # Keyword match display
                    keywords = extract_keywords_from_query(query)
                    text_blob = " ".join(
                        df[df["Location Name"] == name]["clean_text"].astype(str).tolist()
                    ).lower()
                    matched_keywords = [k for k in keywords if k in text_blob] if keywords else []
                    matches_str = ", ".join(matched_keywords) if matched_keywords else ""

                    rows.append({
                        "Location": f"{name} ({r_town}, {r_prov})",
                        "Score": round(score, 3),
                        "Matches": matches_str,
                        "Price": r_price,
                        "Rating": round(r_rating, 2),
                        "Sentiment": round(r_sent, 2)
                    })

                # Render table as HTML
                table_df = pd.DataFrame(rows)
                table_html = table_df.to_html(index=False, classes="armen-table")

                return "Here are my top recommendations:<br><br>" + table_html

            # -------------------------------------------------
            # Select ML or Gemini Response Mode
            # -------------------------------------------------
            if mode == "Armen ML (Fast, Offline)":
                with st.spinner("Armen (ML) is thinking..."):
                    ml_reply = build_ml_response(user_query, top_n=5)
                active_history.append(("armen", ml_reply))
                st.rerun()

            else:
                with st.spinner("Armen (Gemini) is thinking..."):
                    try:
                        armen_reply = armen_gemini_response(
                            user_input=user_query,
                            model=model,
                            df=df,
                            embeddings=embeddings,
                            sentiment=sentiment,
                            rating=rating,
                            cities=cities_list,
                            provinces=provinces_list,
                            mode=(explore_mode[:-1] if explore_mode.endswith("s") else explore_mode),
                            hotel_quality=hotel_quality,
                            top_n=5
                        )
                    except Exception as e:
                        armen_reply = (
                            "Sorry jan — Gemini is not available right now. "
                            f"Please try ML mode instead. (Error: {e})"
                        )

                active_history.append(("armen", armen_reply))
                st.rerun()