# =========================================================
# app.py  — MAIN APPLICATION
# Interactive Sentiment Map + Dual ML/Gemini Chatbot
# =========================================================

"""
This Streamlit application powers the Armenia Restaurants & Hotels Explorer.

Major Features:
---------------
1. Interactive folium sentiment map (province-level)
2. Explore toggle: 🍽 Restaurants ↔ 🏨 Hotels
3. Recommendation Engine:
       - ML Hybrid Recommender (offline, fast)
       - Gemini Conversational Armen (LLM)
4. WhatsApp-style chat UI + reset button
5. Dynamic Top-5 rankings
6. Dual datasets (restaurant + hotel)
7. Separate chat histories per mode

This file acts as the final integration layer between:
    - database.py               (SQL data access)
    - sentiment_model.py        (SVM sentiment)
    - province_aggregation.py   (province stats)
    - armen_ml.py               (recommender engine)
    - armen_gemini.py           (LLM conversational layer)
"""

# =========================================================
# Imports
# =========================================================
import os
import json
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv

# ---- DATA / PROCESSING MODULES ----
from database import fetch_reviews
import sentiment_model
from province_aggregation import compute_province_stats

# ---- ML ENGINE (dual-mode) ----
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

# ---- GEMINI CONVERSATIONAL ARMEN ----
from armen_gemini import armen_gemini_response


# =========================================================
# Streamlit App Setup
# =========================================================
load_dotenv()
st.set_page_config(
    layout="wide",
    page_title="Armenia Explorer (Restaurants & Hotels)"
)

# Path to geojson file for province boundaries
GEOJSON_PATH = os.getenv("ARMENIA_GEOJSON_PATH", "armenia_provinces.geojson")


# =========================================================
# Initialize Session State (must happen before any usage)
# =========================================================
st.session_state.setdefault("restaurant_chat_history", [])
st.session_state.setdefault("hotel_chat_history", [])
st.session_state.setdefault("armen_last_query", "")
st.session_state.setdefault("selected_province", None)
st.session_state.setdefault("last_click_raw", "")
st.session_state.setdefault("trigger_rerun_token", 0)


# =========================================================
# EXPLORE TOGGLE — Restaurants vs Hotels
# (Large styling + unified alignment + emojis)
# =========================================================
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

st.markdown(
    "<div class='explore-bar'><span class='explore-label'>Explore:</span></div>",
    unsafe_allow_html=True
)

# Radio toggle with icons
explore_mode = st.radio(
    label="",
    options=["🍽️ Restaurants", "🏨 Hotels"],
    index=0,
    horizontal=True,
    key="explore_toggle"
)

# Normalize for internal logic
explore_mode = "Restaurants" if explore_mode.startswith("🍽️") else "Hotels"


# =========================================================
# Mode-Specific Chat History
# =========================================================
active_history = (
    st.session_state["restaurant_chat_history"]
    if explore_mode == "Restaurants"
    else st.session_state["hotel_chat_history"]
)


# =========================================================
# Cached Loaders
# =========================================================
@st.cache_data
def load_geojson(path):
    """Load province boundary file for folium map."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data(mode="Restaurants"):
    """
    Load and process dataset for current mode:
        Restaurants → SQL: location_type='Restaurant'
        Hotels      → SQL: location_type='Hotel'
    Applies:
        - SVM sentiment scoring
        - Province-level aggregation
    """
    db_mode = "Restaurant" if mode == "Restaurants" else "Hotel"
    df = fetch_reviews(db_mode)
    df = sentiment_model.apply_sentiment(df, "Review Text")
    stats = compute_province_stats(df)
    return df, stats


# =========================================================
# LOAD DATA FOR CURRENT EXPLORE MODE
# =========================================================
df, province_stats = load_data(mode=explore_mode)

# Ensure Review Text exists
if "Review Text" not in df.columns:
    df["Review Text"] = ""

# Clean text creation (spaCy cleaner or fallback)
try:
    df["clean_text"] = df["Review Text"].apply(sentiment_model.clean_text)
except Exception:
    import re
    df["clean_text"] = df["Review Text"].apply(
        lambda t: " ".join(re.sub(r"[^a-zA-Z\s]", "", str(t).lower()).split())
    )


# =========================================================
# Build ML Backend for This Mode
# =========================================================
kind = explore_mode[:-1] if explore_mode.endswith("s") else explore_mode

# Embeddings for all locations
embeddings = build_embeddings(df, kind=kind)

# Location-level sentiment + rating
sentiment = compute_sentiment(df)
rating = compute_rating(df)

# Hotels → calculate weighted quality score
hotel_quality = compute_hotel_quality(df) if explore_mode == "Hotels" else None

# Extract city & province lists
cities_list = df["Town/City"].dropna().unique().tolist() if "Town/City" in df.columns else []
provinces_list = df["Province"].dropna().unique().tolist() if "Province" in df.columns else []


# =========================================================
# Load Province GeoJSON for Map Rendering
# =========================================================
geojson = load_geojson(GEOJSON_PATH)


# =========================================================
# Merge Province Stats into GeoJSON (Map Coloring)
# =========================================================
for feat in geojson["features"]:
    props = feat["properties"]
    province = props.get("shapeName", "").strip()

    # Try matching province stats row
    match = (
        province_stats[province_stats["Province"].str.lower() == province.lower()]
        if "Province" in province_stats.columns
        else None
    )

    if match is not None and not match.empty:
        r = match.iloc[0]
        props.update({
            "restaurant_count": int(r.get("restaurant_count", 0)),
            "review_count": int(r.get("review_count", 0)),
            "city_count": int(r.get("city_count", 0)),
            "avg_rating": float(r.get("avg_rating", 0.0)),
            "avg_sentiment": float(r.get("avg_sentiment", 0.0)),
            "sentiment_color": r.get("sentiment_color", [200, 200, 200]),
        })
    else:
        props.update({
            "restaurant_count": 0,
            "review_count": 0,
            "city_count": 0,
            "avg_rating": 0.0,
            "avg_sentiment": 0.0,
            "sentiment_color": [200, 200, 200],
        })

# =========================================================
# CSS Styling: Summary Cards + WhatsApp Chat Bubbles
# =========================================================
st.markdown("""
<style>
/* Summary Metric Cards */
.card {
    background: linear-gradient(135deg, #1c1c1c, #0e0e0e);
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 0 16px rgba(0,0,0,0.45);
    margin-bottom: 15px;
    color: white;
}
.card h2 {
    margin: 0;
    font-size: 15px;
    color: #cccccc;
}
.card p {
    margin: 5px 0 0;
    font-size: 30px;
    font-weight: bold;
    color: white;
}

/* WhatsApp-Style CHAT BUBBLES */
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
    margin-left: auto;
    text-align: left;
}

.reset-button-container {
    display:flex;
    justify-content:flex-end;
    margin-top:-10px;
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)



# =========================================================
# Layout Split — Map (Left) + Summary Cards (Right)
# =========================================================
st.title(f"Armenia — {explore_mode} Sentiment Map Explorer")
left, right = st.columns([2, 1])



# =========================================================
# MAP UI — Folium Map Rendering
# =========================================================
with left:
    st.subheader("Click a province on the map")

    # Create folium map centered on Armenia
    m = folium.Map(
        location=[40.1, 44.5],
        zoom_start=7,
        tiles="CartoDB dark_matter"
    )

    # Expose JS map object to Streamlit
    m.get_root().script.add_child(folium.Element("window.map = this;"))

    # Coloring for province shapes
    def style_fn(f):
        r, g, b = f["properties"]["sentiment_color"]
        return {
            "fillColor": f"rgb({r},{g},{b})",
            "color": "white",
            "weight": 1,
            "fillOpacity": 0.87,
        }

    # Add geojson layer
    folium.GeoJson(
        geojson,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=[
                "shapeName", "city_count", "restaurant_count",
                "review_count", "avg_rating", "avg_sentiment"
            ],
            aliases=[
                "Province", "Cities", f"{explore_mode}",
                "Reviews", "Avg Rating", "Sentiment"
            ]
        ),
        popup=folium.GeoJsonPopup(fields=["shapeName"])
    ).add_to(m)

    # Legend HTML
    legend_html = """
    <div style="
        position: absolute;
        top: 20px; right: 20px;
        z-index: 9999;
        background-color: rgba(0,0,0,0.65);
        padding: 12px 15px;
        border-radius: 8px;
        color: white;
        font-size: 13px;
        width: 180px;
        box-shadow: 0 0 10px rgba(0,0,0,0.7);
    ">
        <b>Sentiment Scale</b><br>
        <div style="
            margin-top: 8px;
            width: 100%; height: 15px;
            border-radius: 4px;
            background: linear-gradient(to right,
                rgb(220,60,60),
                rgb(255,210,60),
                rgb(80,200,80)
            );">
        </div>
        <div style="display:flex; justify-content:space-between; margin-top: 3px;">
            <span>Negative</span><span>Neutral</span><span>Positive</span>
        </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Render map in Streamlit
    map_data = st_folium(
        m,
        width=900,
        height=650,
        key=f"map_{st.session_state['trigger_rerun_token']}"
    )



# =========================================================
# CLICK DETECTION — Detect province selected by user
# =========================================================
raw_json = json.dumps(map_data, sort_keys=True)

if raw_json != st.session_state["last_click_raw"]:
    st.session_state["last_click_raw"] = raw_json

    clicked = None

    # 1. If active drawing clicked
    if map_data.get("last_active_drawing"):
        clicked = map_data["last_active_drawing"]["properties"].get("shapeName")

    # 2. If popup clicked
    if not clicked and map_data.get("last_object_clicked_popup"):
        clicked = map_data["last_object_clicked_popup"].replace("Select", "").strip()

    # 3. If tooltip clicked
    if not clicked and map_data.get("last_object_clicked_tooltip"):
        lines = [
            l.strip() for l in map_data["last_object_clicked_tooltip"].splitlines()
            if l.strip()
        ]
        if len(lines) >= 2:
            clicked = lines[1]

    # Handle normalized province lookup
    if clicked:
        lookup = (
            {p.lower(): p for p in province_stats["Province"].unique()}
            if "Province" in province_stats.columns else {}
        )
        canonical = lookup.get(clicked.lower(), clicked)

        # Update selected province
        st.session_state["selected_province"] = canonical
        st.session_state["trigger_rerun_token"] += 1
        st.rerun()



# =========================================================
# SUMMARY CARDS — right column
# =========================================================
with right:
    st.header("Summary")

    total_items = int(province_stats.get("restaurant_count", pd.Series([0])).sum())
    total_reviews = int(province_stats.get("review_count", pd.Series([0])).sum())

    weighted_sent = (
        (
            province_stats.get("avg_sentiment", pd.Series([0]))
            * province_stats.get("review_count", pd.Series([1]))
        ).sum() / max(total_reviews, 1)
    )

    avg_rating = round(float(province_stats.get("avg_rating", pd.Series([0])).mean()), 2)

    st.markdown(f"""
    <div class="card"><h2>Total {explore_mode}</h2><p>{total_items}</p></div>
    <div class="card"><h2>Total Reviews</h2><p>{total_reviews}</p></div>
    <div class="card"><h2>Weighted Sentiment</h2><p>{round(weighted_sent,3)}</p></div>
    <div class="card"><h2>Average Rating</h2><p>{avg_rating}</p></div>
    """, unsafe_allow_html=True)



# =========================================================
# TOP 5 PANEL — Restaurants or Hotels
# =========================================================
st.header(f"Top 5 {explore_mode} (Ranked by Rating & Reviews)")

prov = st.session_state.get("selected_province")

if prov:
    st.subheader(f"Province: {prov}")

    if "Province" not in df.columns:
        st.warning(f"No {explore_mode.lower()} found in this province.")
    else:
        sel = df[df["Province"].str.lower() == prov.lower()].copy()

        if sel.empty:
            st.warning(f"No {explore_mode.lower()} found in this province.")
        else:
            # Clean numeric fields
            sel["Review Rating"] = pd.to_numeric(sel["Review Rating"], errors="coerce")
            sel["sentiment_score"] = pd.to_numeric(sel["sentiment_score"], errors="coerce")

            # Top-5 ranking
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

            st.dataframe(
                top5.style.format({
                    "avg_rating": "{:.2f}",
                    "avg_sentiment": "{:.3f}"
                }),
                use_container_width=True
            )

else:
    st.info("Click a province above to explore its top locations.")
# ------------------------------
# Ensure active_history is current
# ------------------------------
active_history = (
    st.session_state["restaurant_chat_history"]
    if explore_mode == "Restaurants"
    else st.session_state["hotel_chat_history"]
)

# ARMEN model selection (ML vs Gemini) — UI placed above chat
mode = st.radio(
    "Choose Armen mode:",
    ("Armen ML (Fast, Offline)", "Armen Gemini (Conversational AI)"),
    index=0,
    horizontal=True,
    key="armen_mode_select"
)

# Chat container (display + input handled below)
st.markdown("---")
st.header("🤖 Armen — Your Personal Guide")

# Reset Chat Button (duplicate-safe — will reset active history)
cols = st.columns([1, 2, 1])
with cols[2]:
    if st.button("🗑️ Reset Armen Chat", key="reset_active_chat"):
        if explore_mode == "Restaurants":
            st.session_state["restaurant_chat_history"] = []
        else:
            st.session_state["hotel_chat_history"] = []
        # Clear text input as well (if present)
        if "armen_input" in st.session_state:
            st.session_state["armen_input"] = ""
        st.experimental_rerun()

# Render current chat history with WhatsApp-like bubbles
chat_display = st.container()
with chat_display:
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

# Input field (kept outside of chat_display to avoid rerendering issues)
user_query = st.text_input(
    f"Tell Armen what you’re looking for in {explore_mode}:",
    key="armen_input"
)

# Helper: Build ML response (uses your armen_ml.recommend_from_query)
def build_ml_response(query: str, top_n: int = 5) -> str:
    """
    Return a plain-text response built from the ML recommender.
    Keeps responses grounded (no hallucination).
    """
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

    if not recs:
        # Try to detect specific city/province in query to give a clear message
        city_hit = extract_city_from_query(query, cities_list)
        prov_hit = extract_province_from_query(query, provinces_list)

        if city_hit:
            return f"Barev jan — I couldn't find any {explore_mode.lower()} in **{city_hit}**."
        if prov_hit:
            return f"Barev jan — I couldn't find any {explore_mode.lower()} in the **{prov_hit}** province."
        return f"Barev jan — I couldn't find any {explore_mode.lower()} matching your request."

    # Compose friendly ranked list
    preamble = "Here are my top recommendations:\n\n"
    parts = [preamble]

    for name, score in recs:
        # get a representative row if available
        row = df[df["Location Name"] == name].head(1)
        if row.empty:
            continue
        row = row.iloc[0]
        town = row.get("Town/City", "Unknown")
        prov = row.get("Province", "Unknown")
        price = row.get("Location Price Range", row.get("location_price_range", "Unknown"))
        r_rating = rating.get(name, 0.0)
        r_sent = sentiment.get(name, 0.0)

        keywords = extract_keywords_from_query(query)
        text_blob = " ".join(df[df["Location Name"] == name]["clean_text"].astype(str).tolist()).lower()
        matched_keywords = [k for k in keywords if k in text_blob] if keywords else []

        reasons = []
        if matched_keywords:
            reasons.append("matches: " + ", ".join(matched_keywords))

        q_price = extract_price_from_query(query)
        if q_price:
            if q_price == price:
                reasons.append(f"price match ({price})")
            else:
                reasons.append(f"price: {price}")
        else:
            if price:
                reasons.append(f"price: {price}")

        reasons.append(f"rating: {r_rating:.2f}")
        reasons.append(f"sentiment: {r_sent:.2f}")

        # hotels: include hotel_quality if available
        if explore_mode == "Hotels" and hotel_quality:
            qval = hotel_quality.get(name, None)
            if qval is not None:
                reasons.append(f"hotel quality: {qval:.2f}")

        parts.append(
            f"🔹 **{name}** — {town}, {prov}\nScore: {round(score,3)}\nWhy: {', '.join(reasons)}\n\n"
        )

    return "".join(parts)


# Button handler: Ask Armen
if st.button("Ask Armen", key="ask_armen_button"):
    if not user_query or user_query.strip() == "":
        st.warning("Please type something.")
    else:
        # Append user message into the proper session history
        if explore_mode == "Restaurants":
            st.session_state["restaurant_chat_history"].append(("user", user_query))
        else:
            st.session_state["hotel_chat_history"].append(("user", user_query))

        # Decide ML vs Gemini
        if mode == "Armen ML (Fast, Offline)":
            with st.spinner("Armen (ML) is thinking..."):
                try:
                    ml_reply = build_ml_response(user_query, top_n=5)
                except Exception as e:
                    ml_reply = f"Sorry jan — ML recommender failed: {e}"

            # Append Armen reply into session history
            if explore_mode == "Restaurants":
                st.session_state["restaurant_chat_history"].append(("armen", ml_reply))
            else:
                st.session_state["hotel_chat_history"].append(("armen", ml_reply))

            # Clear input and rerun to show updated history
            st.session_state["armen_input"] = ""
            st.rerun()

        else:
            # Gemini path (guarded)
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
                        "Sorry jan — Gemini is not available right now. Please try ML mode instead. "
                        f"(Error: {e})"
                    )

            # Append Armen reply into session history
            if explore_mode == "Restaurants":
                st.session_state["restaurant_chat_history"].append(("armen", armen_reply))
            else:
                st.session_state["hotel_chat_history"].append(("armen", armen_reply))

            # Clear input and rerun
            st.session_state["armen_input"] = ""
            st.rerun()