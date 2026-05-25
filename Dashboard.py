import streamlit as st
import pandas as pd
import finnhub
import datetime
import plotly.graph_objects as go
import os
from modules import LSTM_Forecasting_Model
from modules.FinBERT_Sentiment_Model import classify_news
from modules.LightGBM_Fundamentals_Model import get_fundamentals, classify_stock

st.set_page_config(page_title="Stock Insights", layout="wide", initial_sidebar_state="expanded")

st.sidebar.header("Stock Ticker")
symbol = st.sidebar.text_input("Select Stock", "AAPL")
st.markdown(f"""
    <h1 style="text-align: center; color: #58a6ff; margin-bottom: 20px;">
        Stock Insights: {symbol}
    </h1>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        
        .stApp {
            background-color: #0d1117;
            font-family: 'JetBrains Mono', monospace;
        }
        
        h1, h2, h3 {
            color: #58a6ff !important;
            border-bottom: 1px solid #30363d;
            padding-bottom: 5px;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.6rem !important;
            color: #00ff41 !important;
        }
        
        div[data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {
            background: #161b22;
            border: 1px solid #30363d;
            padding: 15px;
            border-radius: 4px;
        }

        section[data-testid="stSidebar"] {
            background-color: #010409;
            border-right: 1px solid #30363d;
        }
    </style>
""", unsafe_allow_html=True)

df = pd.DataFrame()
risk_value = 0.5
dominant_sentiment = "neutral"
classification = "Neutral"
fundamentals_score = 50

api_key = st.secrets["finnhub_api_key"]
finnhub_client = finnhub.Client(api_key=api_key)

# Fetches the last 30 days of company news from Finnhub, cached for 30 minutes.
@st.cache_data(ttl=1800)
def get_news(symbol):
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=30)
    return finnhub_client.company_news(
        symbol,
        _from=str(start_date),
        to=str(today)
    )

# Runs FinBERT sentiment classification on the news dataframe.
@st.cache_data(show_spinner=False)
def get_sentiment(df):
    return classify_news(df)

# Loads the LSTM model and scaler from disk, training them first if they don't exist.
@st.cache_resource
def load_or_train_lstm(symbol):
    model_path = f"models/{symbol}_lstm_model.keras"
    scaler_path = f"models/{symbol}_scaler.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return LSTM_Forecasting_Model.if_not_exists(symbol)
    else:
        return LSTM_Forecasting_Model.load_lstm(symbol)

if symbol:
    LSTM_model, scaler = load_or_train_lstm(symbol)

# Returns historical price data and future price predictions for the given symbol.
@st.cache_data(show_spinner=False)
def get_lstm_predictions(symbol, _model, _scaler):
    return LSTM_Forecasting_Model.predict_future(symbol, _model, _scaler)

data, predictions = get_lstm_predictions(symbol, LSTM_model, scaler)

historical = data[['Close']].copy()
historical.columns = ['Price']

future_dates = pd.date_range(
    start=historical.index[-1] + pd.Timedelta(days=1),
    periods=len(predictions)
)

fundamentals_dict = get_fundamentals(symbol)
classification, fundamentals_score, _, returned_sector = classify_stock(symbol)

# Combines LSTM, FinBERT, and LightGBM scores into a single weighted risk label and score.
def compute_overall_risk(predictions, risk_value, classification, fundamentals_score):
    lstm_future_return = (predictions[-1] - predictions[0]) / predictions[0]
    lstm_score = 0.5 + (lstm_future_return * 0.5)

    sentiment_score = risk_value

    fundamentals_norm = fundamentals_score / 100 if fundamentals_score is not None else 0.5

    if classification == "Outperform":
        class_score = 1.0
    elif classification == "Underperform":
        class_score = 0.0
    else:
        class_score = 0.5

    lgb_score = (class_score * 0.3) + (fundamentals_norm * 0.7)

    weight_lstm = 0.2
    weight_sentiment = 0.3
    weight_lgb = 0.5

    combined_score = (
        lstm_score * weight_lstm +
        sentiment_score * weight_sentiment +
        lgb_score * weight_lgb
    )

    if combined_score >= 0.65:
        overall_label = "Likely Low Risk"
    elif combined_score >= 0.40:
        overall_label = "Moderate Risk"
    else:
        overall_label = "Likely High Risk"

    return overall_label, combined_score

news = get_news(symbol)
df = pd.DataFrame(news)
if not df.empty:
    df, dominant_sentiment, risk_value = get_sentiment(df)

overall_label, combined_score = compute_overall_risk(predictions, risk_value, classification, fundamentals_score)

col_fund, col_news = st.columns([2, 1])

with col_fund:
        st.subheader(f"{symbol} Price & Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical.index, y=historical['Price'], name="HISTORICAL", line=dict(color='#8b949e')))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, name="LSTM_FORECAST", line=dict(color='#00ff41', dash='dot')))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Fundamental Metrics")
        f_df = pd.DataFrame.from_dict(fundamentals_dict, orient="index", columns=["VALUE"])
        st.dataframe(f_df, use_container_width=True, height=300)

        st.subheader("Fundamental Analysis")
        comp_data = {
            "Metric": ["Performance against S&P 500", "Fundamental Score", "Sector"],
            "Value": [f"{classification}", f"{fundamentals_score}/100", returned_sector]
        }
        st.table(pd.DataFrame(comp_data))

        st.subheader("Overall Risk")
        if combined_score >= 0.65:
            risk_color_label = "#00C853"
        elif combined_score >= 0.40:
            risk_color_label = "#FFAB00"
        else:
            risk_color_label = "#D50000"
        st.markdown(f"""
        <div style="
            background:{risk_color_label};
            border-radius:14px;
            height:55px;
            width:220px;
            display:flex;
            justify-content:center;
            align-items:center;
            color:white;
            font-size:17px;
            font-weight:700;
            letter-spacing:1.5px;
            box-shadow:0 6px 18px rgba(0,0,0,0.35);
            margin-top:10px;
        ">
            {overall_label.upper()}
        </div>
    """, unsafe_allow_html=True)
        st.caption(f"Combined score: {combined_score:.2f} — LSTM, FinBERT & LightGBM.")

with col_news:
        st.subheader("Sentiment Classification")
        fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_value,
        number={'font': {'size': 20, 'color': 'white'}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 0},
            'bar': {'color': "white"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 0.33], 'color': '#D50000'},
                {'range': [0.33, 0.66], 'color': '#FFAB00'},
                {'range': [0.66, 1], 'color': '#00C853'}],}))

        fig_gauge.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        color_map = {
             "negative": "#D50000",
             "neutral": "#FFAB00",
             "positive": "#00C853"
}
        sentiment_label = dominant_sentiment.lower()

        risk_color = color_map.get(sentiment_label, "#9E9E9E")
        risk_html = f"""<div style="display:flex;justify-content:center;align-items:center;width:100%;margin-top:20px;margin-bottom:20px;">
        <div style="
        background:{risk_color};
        border-radius:14px;
        height:55px;
        width:200px;
        display:flex;
        justify-content:center;
        align-items:center;
        text-align:center;
        color:white;
        font-size:17px;
        font-weight:700;
        letter-spacing:1.5px;
        line-height:1;
        box-shadow:0 6px 18px rgba(0,0,0,0.35);
        ">
        {sentiment_label.upper()}
        </div>
        </div>
        """
        st.markdown(risk_html, unsafe_allow_html=True)

        st.subheader("Recent News")
        if not df.empty:
            for i, row in df.head(8).iterrows():
                s_color = "#00ff41" if row['sentiment'] == "positive" else "#d50000" if row['sentiment'] == "negative" else "#ffab00"
                st.markdown(f"""
                <div style="margin-bottom:10px; border-bottom:1px solid #30363d; padding-bottom:8px;">
                    <small style="color:#8b949e;">{row['date']}</small><br>
                    <span style="font-size:0.85rem; font-weight:bold;">{row['headline'][:80]}...</span><br>
                    <span style="color:{s_color}; font-size:0.7rem;">[{row['sentiment']}]</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.write("No recent news found for this ticker")

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"]::after {
            content: "NOT FINANCIAL ADVICE";
            position: fixed;
            bottom: 0px;
            left: 0px;
            background: #0d1117;
            border-top: 1px solid #ff3b3b;
            border-right: 1px solid #ff3b3b;
            color: #ff6b6b;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 1.2px;
            padding: 6px 14px;
            border-radius: 0px 6px 0px 0px;
            z-index: 99999;
        }
        [data-testid="stHeader"]::after {
            content: "NOT FINANCIAL ADVICE";
            position: fixed;
            top: 38px;
            right: 110px;
            background: #0d1117;
            border: 1px solid #ff3b3b;
            color: #ff6b6b;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 1.2px;
            padding: 5px 10px;
            border-radius: 6px;
            z-index: 99999;
        }
    </style>
""", unsafe_allow_html=True)
