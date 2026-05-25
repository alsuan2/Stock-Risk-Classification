import pandas as pd
import os
import requests

FILE = "dataset/sp500_tickers.csv"
URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# Fetches S&P 500 tickers from Wikipedia, falling back to cache on failure.
def fetch_sp500_tickers() -> list:
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        response = requests.get(URL, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(response.text)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception as e:
        print(f"Failed to fetch tickers from Wikipedia: {e}")
        return load_cached_tickers() or []

# Returns cached tickers from disk, or None if no cache exists.
def load_cached_tickers() -> list | None:
    if os.path.exists(FILE):
        df = pd.read_csv(FILE)
        return df["Symbol"].tolist()
    return None

# Saves tickers to a local CSV cache.
def save_cache(tickers: list):
    os.makedirs("dataset", exist_ok=True)
    df = pd.DataFrame({
        "Symbol": tickers,
    })
    df.to_csv(FILE, index=False)

# Returns cached tickers if available, otherwise fetches and caches fresh ones.
def get_sp500_tickers(force_refresh: bool = False) -> list:
    if not force_refresh:
        cached = load_cached_tickers()
        if cached:
            return cached
    tickers = fetch_sp500_tickers()
    save_cache(tickers)
    return tickers

if __name__ == "__main__":
    tickers = get_sp500_tickers()
    print(f"Loaded {len(tickers)} S&P 500 tickers: {tickers[:10]}...")
