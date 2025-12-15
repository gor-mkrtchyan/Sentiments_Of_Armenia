# armen_gemini.py
import os
from typing import List, Dict, Any
import google.generativeai as genai

# configure API key from env
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from armen_ml import (
    recommend_from_query,
    extract_keywords_from_query,
    extract_price_from_query,
    extract_city_from_query,
    extract_province_from_query
)


def armen_gemini_response(
    user_input: str,
    model,  # embedding model (SentenceTransformer) passed for encoding if needed
    df,
    embeddings: Dict[str, Any],
    sentiment: Dict[str, float],
    rating: Dict[str, float],
    cities: List[str],
    provinces: List[str],
    mode: str = "Restaurant",
    hotel_quality: Dict[str, float] = None,
    top_n: int = 5
) -> str:
    """
    Generate a grounded reply using Gemini.
    Uses the ML recommender to provide a candidate list and asks Gemini to format the reply,
    **strictly** using only the candidates (no hallucinations).
    """

    # 1) extract preferences
    keywords = extract_keywords_from_query(user_input)
    price_pref = extract_price_from_query(user_input)
    city = extract_city_from_query(user_input, cities)
    province = extract_province_from_query(user_input, provinces)

    # 2) run ML recommender (use mode)
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

    # 3) fallback if empty
    if not ml_recs:
        if city:
            return f"Barev jan — I couldn't find any {mode.lower()}s in {city}. Would you like me to search nearby cities?"
        if province:
            return f"Barev jan — I couldn't find any {mode.lower()}s in the {province} province. Shall I expand the search?"
        return f"Barev jan — I couldn't find matching {mode.lower()}s. Shall I broaden the search?"

    # 4) build structured candidate info for prompt (only names & scores; no invented facts)
    candidates = [{"name": name, "score": float(score)} for name, score in ml_recs]

    # 5) choose personality based on mode
    if mode == "Hotel":
        system_prompt = """
You are Armen, a warm Armenian hotel expert (slightly hotel-focused tone).
RULES:
- Use only the provided candidate list. Do NOT invent or hallucinate any extra restaurants/hotels, facts, or locations.
- For each candidate, give one short description (based on the available data passed separately) and one clear reason why it matches (city/province, keywords, price, sentiment, rating, hotel-quality).
- Keep the tone warm but slightly more hotel-focused (mention cleanliness, service, rooms, sleep if relevant).
- Use short Armenian warmth phrases like "Barev jan", "sirun jan" sparingly.
- Keep the reply concise and helpful.
"""
    else:
        system_prompt = """
You are Armen, a warm Armenian restaurant expert.
RULES:
- Use only the provided candidate list. Do NOT invent or hallucinate any extra restaurants/hotels, facts, or locations.
- For each candidate, give one short description and one clear reason why it matches (city/province, keywords, price, sentiment, rating).
- Use short Armenian warmth phrases like "Barev jan", "sirun jan".
- Keep the reply concise and helpful.
"""

    # 6) Build user prompt including the candidate list and the extracted preferences
    user_prompt = f"""
User request:
{user_input}

Extracted:
- city: {city}
- province: {province}
- keywords: {keywords}
- price_pref: {price_pref}
- mode: {mode}

Candidates (do NOT invent beyond this list):
{candidates}

Instructions:
- Greet user briefly.
- Present the top {len(candidates)} candidates as short bullets.
- For each candidate: provide one short descriptor line (use the candidate name and available row info if you have it) and one reason why it matches (keyword/city/price/sentiment/rating/quality).
- If there is no exact match to user's city/province, say so and offer to broaden filter.
- Do NOT provide phone numbers, addresses, or other facts not in the provided candidate list or the ML data.
"""

    # 7) Call Gemini
    llm = genai.GenerativeModel("models/gemini-2.5-flash")
    resp = llm.generate_content(system_prompt + "\n\n" + user_prompt)

    # Return the assistant text
    return resp.text
