# =========================================================
# app.py  (Updated with Bigger Explore Toggle, Icons, Chat Bubbles, Reset Button)
# =========================================================

import os
import json
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv

# ---- YOUR MODULES ----
from database import fetch_reviews
import sentiment_model
from province_aggregation import compute_province_stats

# ---- ARMEN ML ENGINE (dual-mode) ----
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

# ---- ARMEN GEMINI ----
from armen_gemini import armen_gemini_response


# ---------------------------------------------------------
# Streamlit setup
# ---------------------------------------------------------
load_dotenv()
st.set_page_config(layout="wide", page_title="Armenia Explorer (Restaurants & Hotels)")

GEOJSON_PATH = os.getenv("ARMENIA_GEOJSON_PATH", "armenia_provinces.geojson")

# ---------------------------------------------------------
# Initialize session state keys (must happen before access)
# ---------------------------------------------------------
st.session_state.setdefault("restaurant_chat_history", [])
st.session_state.setdefault("hotel_chat_history", [])
st.session_state.setdefault("armen_last_query", "")
st.session_state.setdefault("selected_province", None)
st.session_state.setdefault("last_click_raw", "")
st.session_state.setdefault("trigger_rerun_token", 0)


# ---------------------------------------------------------
# UI: Explore toggle (top, bigger, unified alignment)
# ---------------------------------------------------------
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

st.markdown("<div class='explore-bar'><span class='explore-label'>Explore:</span></div>", unsafe_allow_html=True)

explore_mode = st.radio(
    label="",
    options=["🍽️ Restaurants", "🏨 Hotels"],
    index=0,
    horizontal=True,
    key="explore_toggle"
)

# Normalize
if explore_mode.startswith("🍽️"):
    explore_mode = "Restaurants"
else:
    explore_mode = "Hotels"


# ---------------------------------------------------------
# chat history based on mode
# ---------------------------------------------------------
active_history = (
    st.session_state["restaurant_chat_history"]
    if explore_mode == "Restaurants"
    else st.session_state["hotel_chat_history"]
)


# ---------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------
@st.cache_data
def load_geojson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_data(mode="Restaurants"):
    db_mode = "Restaurant" if mode == "Restaurants" else "Hotel"
    df = fetch_reviews(db_mode)
    df = sentiment_model.apply_sentiment(df, "Review Text")
    stats = compute_province_stats(df)
    return df, stats


# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
df, province_stats = load_data(mode=explore_mode)

# Defensive: ensure expected columns exist
if "Review Text" not in df.columns:
    df["Review Text"] = ""

# Ensure clean_text exists
try:
    df["clean_text"] = df["Review Text"].apply(sentiment_model.clean_text)
except:
    import re
    df["clean_text"] = df["Review Text"].apply(
        lambda t: " ".join(re.sub(r"[^a-zA-Z\s]", "", str(t).lower()).split())
    )


# ---------------------------------------------------------
# Build ML backend for current mode
# ---------------------------------------------------------
kind = explore_mode[:-1] if explore_mode.endswith("s") else explore_mode

embeddings = build_embeddings(df, kind=kind)
sentiment = compute_sentiment(df)
rating = compute_rating(df)

hotel_quality = compute_hotel_quality(df) if explore_mode == "Hotels" else None

cities_list = df["Town/City"].dropna().unique().tolist() if "Town/City" in df.columns else []
provinces_list = df["Province"].dropna().unique().tolist() if "Province" in df.columns else []


# ---------------------------------------------------------
# Load GeoJSON
# ---------------------------------------------------------
geojson = load_geojson(GEOJSON_PATH)

# ---------------------------------------------------------
# Merge province-level stats
# ---------------------------------------------------------
for feat in geojson["features"]:
    props = feat["properties"]
    province = props.get("shapeName", "").strip()
    match = province_stats[province_stats["Province"].str.lower() == province.lower()] if "Province" in province_stats.columns else None

    if match is not None and not match.empty:
        r = match.iloc[0]
        props.update({
            "restaurant_count": int(r.get("restaurant_count", 0)),
            "review_count": int(r.get("review_count", 0)),
            "city_count": int(r.get("city_count", 0)),
            "avg_rating": float(r.get("avg_rating", 0.0)),
            "avg_sentiment": float(r.get("avg_sentiment", 0.0)),
            "sentiment_color": r.get("sentiment_color", [200,200,200])
        })
    else:
        props.update({
            "restaurant_count": 0,
            "review_count": 0,
            "city_count": 0,
            "avg_rating": 0.0,
            "avg_sentiment": 0.0,
            "sentiment_color": [200,200,200]
        })


# ---------------------------------------------------------
# CSS for Cards + Chat Bubbles (WhatsApp-like)
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

/* WHATSAPP-STYLE CHAT BUBBLES */

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

.reset-button-container {
    display:flex;
    justify-content:flex-end;
    margin-top:-10px;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Layout
# ---------------------------------------------------------
st.title(f"Armenia — {explore_mode} Sentiment Map Explorer")
left, right = st.columns([2, 1])

# ---------------------------------------------------------
# MAP UI
# ---------------------------------------------------------
with left:
    st.subheader("Click a province on the map")

    m = folium.Map(
        location=[40.1, 44.5],
        zoom_start=7,
        tiles="CartoDB dark_matter"
    )

    m.get_root().script.add_child(folium.Element("window.map = this;"))

    def style_fn(f):
        r, g, b = f["properties"]["sentiment_color"]
        return {
            "fillColor": f"rgb({r},{g},{b})",
            "color": "white",
            "weight": 1,
            "fillOpacity": 0.87,
        }

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

    map_data = st_folium(
        m,
        width=900,
        height=650,
        key=f"map_{st.session_state['trigger_rerun_token']}"
    )

# ---------------------------------------------------------
# CLICK DETECTION
# ---------------------------------------------------------
raw_json = json.dumps(map_data, sort_keys=True)

if raw_json != st.session_state["last_click_raw"]:
    st.session_state["last_click_raw"] = raw_json

    clicked = None
    if map_data.get("last_active_drawing"):
        clicked = map_data["last_active_drawing"]["properties"].get("shapeName")

    if not clicked and map_data.get("last_object_clicked_popup"):
        clicked = map_data["last_object_clicked_popup"].replace("Select", "").strip()

    if not clicked and map_data.get("last_object_clicked_tooltip"):
        lines = [
            l.strip() for l in map_data["last_object_clicked_tooltip"].splitlines()
            if l.strip()
        ]
        if len(lines) >= 2:
            clicked = lines[1]

    if clicked:
        lookup = (
            {p.lower(): p for p in province_stats["Province"].unique()}
            if "Province" in province_stats.columns else {}
        )
        canonical = lookup.get(clicked.lower(), clicked)
        st.session_state["selected_province"] = canonical
        st.session_state["trigger_rerun_token"] += 1
        st.rerun()


# ---------------------------------------------------------
# SUMMARY CARDS
# ---------------------------------------------------------
with right:
    st.header("Summary")

    total_items = int(province_stats.get("restaurant_count", pd.Series([0])).sum())
    total_reviews = int(province_stats.get("review_count", pd.Series([0])).sum())
    weighted_sent = (
        (province_stats.get("avg_sentiment", pd.Series([0])) * province_stats.get("review_count", pd.Series([1]))).sum()
        / max(total_reviews, 1)
    )
    avg_rating = round(float(province_stats.get("avg_rating", pd.Series([0])).mean()), 2)

    st.markdown(f"""
        <div class="card"><h2>Total {explore_mode}</h2><p>{total_items}</p></div>
        <div class="card"><h2>Total Reviews</h2><p>{total_reviews}</p></div>
        <div class="card"><h2>Weighted Sentiment</h2><p>{round(weighted_sent,3)}</p></div>
        <div class="card"><h2>Average Rating</h2><p>{avg_rating}</p></div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# TOP 5 (Restaurants or Hotels)
# ---------------------------------------------------------
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
            sel["Review Rating"] = pd.to_numeric(sel["Review Rating"], errors="coerce")
            sel["sentiment_score"] = pd.to_numeric(sel["sentiment_score"], errors="coerce")

            top5 = (
                sel.groupby("Location Name")
                .agg(
                    avg_rating=("Review Rating","mean"),
                    reviews_count=("Review Text","count"),
                    avg_sentiment=("sentiment_score","mean")
                )
                .reset_index()
                .sort_values(["avg_rating","reviews_count"], ascending=[False,False])
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

# =========================================================
#  ARMEN CHATBOT UI — (Dual-Mode + Chat Bubbles + Reset Button)
# =========================================================
st.markdown("---")
st.header("🤖 Armen — Your Personal Guide")

# -----------------------------------------------
# Reset Chat Button (mode-specific)
# -----------------------------------------------
st.markdown(
    "<div class='reset-button-container'>",
    unsafe_allow_html=True
)

if st.button("🗑️ Reset Armen Chat", key="reset_chat_btn"):
    # Reset ONLY the active history
    if explore_mode == "Restaurants":
        st.session_state["restaurant_chat_history"] = []
    else:
        st.session_state["hotel_chat_history"] = []

    # Rerun to clear chat display
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------
# Armen Model Toggle (ML / Gemini)
# -----------------------------------------------
mode = st.radio(
    "Choose Armen mode:",
    ("Armen ML (Fast, Offline)", "Armen Gemini (Conversational AI)"),
    index=0,
    horizontal=True
)


# -----------------------------------------------
# Chat Container
# -----------------------------------------------
chat_container = st.container()

with chat_container:

    # ---------------------------
    # Display chat history
    # ---------------------------
    for role, msg in active_history:
        if role == "user":
            st.markdown(
                f"<div class='user-bubble'>"
                f"🧑‍💬 <strong>You:</strong><br>{msg}</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='armen-bubble'>"
                f"🤖 <strong>Armen:</strong><br>{msg}</div>",
                unsafe_allow_html=True
            )

    # ---------------------------
    # User Input
    # ---------------------------
    user_query = st.text_input(
        f"Tell Armen what you’re looking for in {explore_mode}:",
        key="armen_input"
    )

    # ---------------------------
    # On Send
    # ---------------------------
    if st.button("Ask Armen", key="ask_armen"):
        if user_query.strip() == "":
            st.warning("Please type something.")
        else:
            # add user message
            active_history.append(("user", user_query))

            # ============================
            # ML RESPONSE (FAST, OFFLINE)
            # ============================
            def build_ml_response(query, top_n=5):
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

                # No results
                if not recs:
                    city_hit = extract_city_from_query(query, cities_list)
                    prov_hit = extract_province_from_query(query, provinces_list)

                    if city_hit:
                        return f"Barev jan — I couldn't find any {explore_mode.lower()} in **{city_hit}**."
                    if prov_hit:
                        return f"Barev jan — I couldn't find any {explore_mode.lower()} in the **{prov_hit}** province."

                    return f"Barev jan — I couldn't find any {explore_mode.lower()} matching your request."

                # Normal results
                preamble = "Here are my top recommendations:\n\n"
                parts = [preamble]

                for name, score in recs:
                    row = df[df["Location Name"] == name].head(1)
                    if row.empty:
                        continue
                    row = row.iloc[0]

                    r_town = row.get("Town/City", "Unknown")
                    r_prov = row.get("Province", "Unknown")
                    r_price = row.get("Location Price Range", "Unknown")
                    r_rating = rating.get(name, 0.0)
                    r_sent = sentiment.get(name, 0.0)

                    keywords = extract_keywords_from_query(query)
                    text_blob = " ".join(
                        df[df["Location Name"] == name]["clean_text"].astype(str).tolist()
                    ).lower()
                    matched_keywords = [k for k in keywords if k in text_blob] if keywords else []

                    reasons = []
                    if matched_keywords:
                        reasons.append("matches: " + ", ".join(matched_keywords))

                    q_price = extract_price_from_query(query)
                    if q_price:
                        if q_price == r_price:
                            reasons.append(f"price match ({r_price})")
                        else:
                            reasons.append(f"price: {r_price}")
                    else:
                        if r_price:
                            reasons.append(f"price: {r_price}")

                    reasons.append(f"rating: {r_rating:.2f}")
                    reasons.append(f"sentiment: {r_sent:.2f}")

                    # Hotel-only upgrade
                    if explore_mode == "Hotels" and hotel_quality:
                        quality_value = hotel_quality.get(name, None)
                        if quality_value is not None:
                            reasons.append(f"hotel quality: {quality_value:.2f}")

                    parts.append(
                        f"🔹 **{name}** — {r_town}, {r_prov}\n"
                        f"Score: {round(score, 3)}\n"
                        f"Why: {', '.join(reasons)}\n\n"
                    )

                return "".join(parts)

            # ============================
            # CHOOSE MODE: ML vs GEMINI
            # ============================
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

