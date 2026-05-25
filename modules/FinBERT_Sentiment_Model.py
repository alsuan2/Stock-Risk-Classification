import streamlit as st
from transformers import pipeline
import pandas as pd

# Loads and caches the FinBERT sentiment pipeline.
@st.cache_resource
def load_finbert():
    return pipeline("text-classification", model="ProsusAI/finbert", truncation=True, max_length=512, top_k=None)

# Classifies headlines in dataframe and returns results of dominant sentiment, and a 0–1 risk score.
def classify_news(df):
    if df.empty:
        return pd.DataFrame(columns=["date", "headline", "sentiment", "confidence"]), "neutral", 0.5

    finbert = load_finbert()
    df = df.copy()
    df["date"] = pd.to_datetime(df["datetime"], unit="s").dt.date
    df["headline"] = df["headline"].fillna("").str.strip()

    # Returns the top sentiment label, its confidence, and a positive-minus-negative lean score.
    def get_sentiment(text):
        if not text:
            return "neutral", 0.0, 0.0
        results = finbert(text)[0]
        score_map = {r["label"].lower(): r["score"] for r in results}
        top_label = max(score_map, key=score_map.get)
        top_conf = score_map[top_label]
        lean = score_map.get("positive", 0.0) - score_map.get("negative", 0.0)
        return top_label, top_conf, lean

    df[["sentiment", "confidence", "_lean"]] = df["headline"].apply(
        lambda x: pd.Series(get_sentiment(x))
    )

    avg_lean = df["_lean"].mean()

    if avg_lean > 0.05:
        dominant_sentiment = "positive"
    elif avg_lean < -0.05:
        dominant_sentiment = "negative"
    else:
        dominant_sentiment = "neutral"

    risk_value = round(max(0.0, min(1.0, (avg_lean + 1) / 2)), 4)
    df = df.drop(columns=["_lean"])

    return df, dominant_sentiment, risk_value
