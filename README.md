# 🇦🇲 Armenia Restaurants & Hotels Explorer  
### *AI-powered sentiment map, smart recommendations, and conversational assistant.*

This project provides an intelligent platform for exploring **Restaurants** and **Hotels** across Armenia.  
It brings together:

- 🗺️ **Interactive Folium Sentiment Map**  
- ⭐ **ML-Based Recommender System**  
- 🤖 **Armen — AI Assistant (ML Mode + Gemini Mode)**  
- 🏨 **Dual datasets:** Restaurants + Hotels  
- 🎨 **WhatsApp-style chat UI**  
- 🧠 **Custom NLP + Sentiment + Embeddings**  
- 🗄️ **PostgreSQL backend**  
- 📊 **Detailed province insights**

This app is ideal for tourism analysis, hospitality intelligence, travel apps, or data-driven recommendation platforms.

---

## ✨ Features

### 🗺️ Interactive Sentiment Map
Visualizes aggregated restaurant/hotel sentiment across Armenian provinces.

- Hover to see details  
- Click a province to filter  
- Color-coded sentiment  
- Displays:
  - Total Restaurants/Hotels  
  - Review Counts  
  - Avg Sentiment  
  - Avg Rating  

---

### 🍽️🏨 Dual Dataset Support  
A top-level toggle lets you seamlessly switch between:

**Explore:** `🍽️ Restaurants` | `🏨 Hotels`

Everything in the app updates:

- Map  
- Stats  
- ML backend  
- Chat personality  
- Top recommendations  
- Data loading  
- Embeddings  
- Sentiment  

---

### 🤖 Armen — Your Personal Guide

Armen has **two personalities**:

---

### 1️⃣ ML Mode (Offline, Fast)
- SentenceTransformer embeddings  
- Keyword-based filters  
- City/Province extraction  
- Price preference detection  
- Sentiment scoring (SVM model)  
- Hybrid scoring:
  - Embedding similarity  
  - Keyword boost  
  - Sentiment  
  - Price match  
  - Restaurant-specific ratings  
  - **Hotel quality score (weighted):**
    - Cleanliness (0.35)  
    - Service (0.25)  
    - Location (0.20)  
    - Rooms (0.10)  
    - Value (0.05)  
    - Sleep Quality (0.05)

---

### 2️⃣ Gemini Mode (Conversational)
- Warm Armenian personality  
- Natural explanations  
- Strict grounding (no hallucination)  
- Restaurant persona  
- Hotel persona  

Example response:

> *“Barev jan! Let me recommend a lovely family-friendly place in Dilijan…”*

---

## 💬 WhatsApp-Style Chat UI
- Rounded chat bubbles  
- Left (User) / Right (Armen)  
- Icons  
- Soft gradients  
- Auto-scroll  
- Per-mode chat memory  
- **“Reset Conversation”** button  

---

## ⭐ Top 5 Restaurants/Hotels Panel
Shows best-rated locations in a selected province based on:

- Avg Review Rating  
- Review Volume  
- Sentiment Score  

---

## 🧠 ML Engine

### Embeddings  
Model: **all-MiniLM-L6-v2**

Separate embeddings are built for:

- Restaurants  
- Hotels  

---

### Sentiment Analysis  
Custom SVM classifier outputs:

- +1 → Positive  
- 0 → Neutral  
- −1 → Negative  

Used for:

- Province stats  
- Recommender  
- Chat explanations  

---

# 🗄️ Database Structure

### Restaurants Dataset
- Location Name  
- Province  
- Town/City  
- Review Text  
- Review Rating  
- Subratings:
  - Value, Service, Food, Atmosphere  
- Price Range  
- User Information  

### Hotels Dataset
- Location Name  
- Province  
- Town/City  
- Review Text  
- Review Rating  
- Subratings:
  - Cleanliness  
  - Service  
  - Location  
  - Rooms  
  - Value  
  - Sleep Quality  

---

# 🔧 Installation

## 1. Clone repo

```bash
git clone https://github.com/gor-mkrtchyan/Sentiments_Of_Armenia.git
cd Sentiments_Of_Armenia
```

## 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Create .env file

```bash
DB_HOST=your-db-host
DB_USER=your-db-user
DB_PASSWORD=your-password
DB_NAME=your-db-name
DB_PORT=5432
DB_SSLMODE=require

GEMINI_API_KEY=your-google-cloud-key
ARMENIA_GEOJSON_PATH=armenia_provinces.geojson
```

## 5. ▶️ Run the App

```bash
streamlit run app.py
```
