# armen_gemini.py
"""
Armen Gemini Conversational Engine
==================================

This module defines the interface between:
    - The ML recommendation engine (armen_ml.py)
    - Google’s Gemini API (generative AI)
    - The Streamlit conversational UI

Responsibilities:
-----------------
1. Extract structured signals from the user query  
   (keywords, city, province, price preference)

2. Run the ML hybrid recommender to obtain **grounded** candidate suggestions.

3. Build a **strict grounding prompt** so Gemini:
      - NEVER hallucinates restaurants/hotels  
      - ONLY uses the ML-provided candidate list  
      - Speaks in Armen’s warm Armenian tone  
      - Uses different personas for Restaurants vs. Hotels

4. Generate a natural-language reply via Gemini.

Gemini NEVER:
-------------
- Invents new restaurants/hotels  
- Invents details not present in ML results  
- Provides unavailable metadata  
- Breaks grounding rules  

Armen’s Personality:
--------------------
Restaurant Mode → Warm, food-loving “Armenian uncle”  
Hotel Mode → Warm but cleanliness/service/location–focused  
"""

import os
from typing import List, Dict, Any
import google.generativeai as genai

# Configure the Gemini API using environment variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Import ML helpers
from armen_ml import (
    recommend_from_query,
    extract_keywords_from_query,
    extract_price_from_query,
    extract_city_from_query,
    extract_province_from_query
)


# ============================================================
#  Main Conversational Function — Armen Gemini Reply
# ============================================================

def armen_gemini_response(
    user_input: str,
    model,                     # embedding model (SentenceTransformer)
    df,
    embeddings: Dict[str, Any],
    sentiment: Dict[str, float],
    rating: Dict[str, float],
    cities: List[str],
    provinces: List[str],
    mode: str = "Restaurant",  # "Restaurant" | "Hotel"
    hotel_quality: Dict[str, float] = None,
    top_n: int = 5
) -> str:
    """
    Generate Armen's conversational response using Gemini 2.5 Flash.

    Steps:
        1. Extract structured query features (keywords, city, province, price)
        2. Run ML recommender to obtain grounded candidates
        3. Build a grounding-aware system prompt + user prompt
        4. Call Gemini and return the formatted natural reply

    Grounding Rules:
        - Gemini MUST only use the ML candidate list
        - No fabricated restaurants, hotels, dishes, attributes, or locations
        - Explanations must reflect ML-calculated scores/logic
        - Tone adjusts based on mode ("Restaurant" vs "Hotel")
    """

    # ------------------------------------------------------------
    # 1. Extract Query Signals (NLU Layer)
    # ------------------------------------------------------------
    keywords = extract_keywords_from_query(user_input)
    price_pref = extract_price_from_query(user_input)
    city = extract_city_from_query(user_input, cities)
    province = extract_province_from_query(user_input, provinces)

    # ------------------------------------------------------------
    # 2. Run ML Recommender (Ground Truth Candidate List)
    # ------------------------------------------------------------
    ml_recs = recommend_from_query(
        query=user_input,
        df=df,
        embeddings=embeddings,
        sentiment=sentiment,
        rating=rating,
        cities=cities,
        provinces=provinces,
        mode=mode,
        hotel_quality=hotel_quality,
        top_n=top_n
    )

    # No candidates found → friendly fallback
    if not ml_recs:
        if city:
            return (
                f"Barev jan — I couldn't find any {mode.lower()}s in **{city}**. "
                f"Would you like me to search nearby places?"
            )
        if province:
            return (
                f"Barev jan — I couldn't find any {mode.lower()}s in the **{province}** province. "
                f"Shall I expand the search?"
            )
        return (
            f"Barev jan — I wasn't able to find matching {mode.lower()}s. "
            "Shall I broaden your search?"
        )

    # Convert ML results to a structured list for grounding
    candidates = [{"name": name, "score": float(score)} for name, score in ml_recs]

    # ------------------------------------------------------------
    # 3. Choose Armen’s Personality (Restaurant vs Hotel)
    # ------------------------------------------------------------

    if mode == "Hotel":
        system_prompt = """
You are Armen, a warm Armenian **hotel expert**, with a slightly hotel-focused tone.

Behavior Rules:
- ONLY use the provided candidate list.
- NO hallucinations — do not invent hotels, details, or locations.
- For each candidate: provide a short descriptor and one matching reason
  (cleanliness, service, location, rooms, sentiment, or query keyword relevance).
- Use warm Armenian phrases like “Barev jan”, “sirun jan” — but moderately.
- Keep responses concise, helpful, and friendly.
"""
    else:
        system_prompt = """
You are Armen, a warm Armenian **restaurant expert**, known for friendly guidance.

Behavior Rules:
- ONLY use the provided candidate list.
- NO invented restaurants, dishes, or locations.
- For each candidate: provide a short descriptor and one reason why it matches
  (keywords, city/province, price, sentiment, rating).
- Use Armenian warmth such as “Barev jan”, “sirun jan”.
- Keep replies concise and helpful.
"""

    # ------------------------------------------------------------
    # 4. Build User Prompt (Grounded Context)
    # ------------------------------------------------------------
    user_prompt = f"""
User request:
{user_input}

Extracted Query Signals:
- City: {city}
- Province: {province}
- Keywords: {keywords}
- Price Preference: {price_pref}
- Mode: {mode}

Candidate List (STRICTLY use these — do NOT invent anything else):
{candidates}

Instructions:
- Start with a friendly greeting.
- Present the top {len(candidates)} recommendations as concise bullet points.
- For each:
      1. Describe it using only available/high-level info.
      2. Provide one clear reason why it matches the user's request
         (keyword → match, city/province → match, price → match/mismatch,
          sentiment → strong positivity, quality → high, etc.)
- No phone numbers, no addresses, no invented attributes.
- If no exact city/province match → politely note and explain.
"""

    # ------------------------------------------------------------
    # 5. Call the Gemini API
    # ------------------------------------------------------------
    llm = genai.GenerativeModel("models/gemini-2.5-flash")

    try:
        response = llm.generate_content(system_prompt + "\n\n" + user_prompt)
        return response.text

    except Exception as e:
        # Last-resort error fallback
        return (
            "Barev jan — Gemini couldn't respond right now. "
            "Try again in a few moments or use Armen ML mode.\n\n"
            f"(Error: {e})"
        )
