import yfinance as yf
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
import os
from modules.sp500_dataset import get_sp500_tickers

warnings.filterwarnings("ignore")
DATA_PATH = "dataset/fundamentals.csv"
MODEL = "lightgbm_model.pkl"

# Fetches live fundamental metrics and sector for a ticker based off user input.
def get_fundamentals(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info

    fundamentals = {
        "de_ratio": info.get("debtToEquity"),
        "current_ratio": info.get("currentRatio"),
        "roe": info.get("returnOnEquity"),
        "gross_margin": info.get("grossMargins"),
        "profit_margin": info.get("profitMargins"),
        "revenue_growth": info.get("revenueGrowth"),
        "eps_growth": info.get("earningsQuarterlyGrowth"),
        "market_cap": info.get("marketCap"),
        "sector": info.get("sector")
    }
    return fundamentals

# Builds labelled training rows from quarterly financials and forward price returns vs S&P 500.
def get_label_rows(symbol, forward_days=180):
    try:
        ticker = yf.Ticker(symbol)

        bs  = ticker.quarterly_balance_sheet
        inc = ticker.quarterly_income_stmt
        info = ticker.info
        sector = info.get("sector")

        if bs is None or inc is None or bs.empty or inc.empty:
            return []

        price = yf.download(symbol, period="4y", interval="1d", progress=False)["Close"].dropna()
        sp500 = yf.download("^GSPC",  period="4y", interval="1d", progress=False)["Close"].dropna()

        if len(price) < forward_days + 30 or len(sp500) < forward_days + 30:
            return []

        rows = []
        for report_date in bs.columns[:4]:
            try:
                rd = pd.Timestamp(report_date)

                price_idx = price.index.searchsorted(rd)
                if price_idx + forward_days >= len(price):
                    continue

                entry_price = float(price.iloc[price_idx])
                exit_price  = float(price.iloc[price_idx + forward_days])

                sp500_idx   = sp500.index.searchsorted(rd)
                if sp500_idx + forward_days >= len(sp500):
                    continue

                sp500_entry = float(sp500.iloc[sp500_idx])
                sp500_exit  = float(sp500.iloc[sp500_idx + forward_days])

                stock_return = (exit_price  - entry_price)  / entry_price
                sp500_return = (sp500_exit  - sp500_entry)  / sp500_entry
                label = 1 if stock_return > sp500_return else 0

                def safe(df, key):
                    try:
                        val = df.loc[key, report_date] if key in df.index else np.nan
                        return float(val) if not pd.isna(val) else np.nan
                    except Exception:
                        return np.nan

                total_debt   = safe(bs,  "Total Debt")
                equity       = safe(bs,  "Stockholders Equity")
                current_assets = safe(bs, "Current Assets")
                current_liab   = safe(bs, "Current Liabilities")
                net_income   = safe(inc, "Net Income")
                gross_profit = safe(inc, "Gross Profit")
                total_rev    = safe(inc, "Total Revenue")
                market_cap   = info.get("marketCap")

                de_ratio       = total_debt / equity       if equity       and equity != 0       else np.nan
                current_ratio  = current_assets / current_liab if current_liab and current_liab != 0 else np.nan
                roe            = net_income / equity       if equity       and equity != 0       else np.nan
                gross_margin   = gross_profit / total_rev  if total_rev    and total_rev != 0    else np.nan
                profit_margin  = net_income  / total_rev   if total_rev    and total_rev != 0    else np.nan

                row = {
                    "de_ratio":       de_ratio,
                    "current_ratio":  current_ratio,
                    "roe":            roe,
                    "gross_margin":   gross_margin,
                    "profit_margin":  profit_margin,
                    "revenue_growth": info.get("revenueGrowth"),
                    "eps_growth":     info.get("earningsQuarterlyGrowth"),
                    "market_cap":     market_cap,
                    "sector":         sector,
                    "label":          label,
                    "symbol":         symbol,
                }

                fund_keys = ["de_ratio", "current_ratio", "roe", "gross_margin", "profit_margin",
                             "revenue_growth", "eps_growth", "market_cap"]
                if any(pd.isna(row.get(k)) for k in fund_keys):
                    continue

                rows.append(row)

            except Exception:
                continue

        return rows

    except Exception as e:
        print(f"{symbol} training rows failed: {e}")
        return []

# Computes mean fundamental metrics grouped by sector.
def get_sector_avg_fundamentals(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        if col not in ["sector", "symbol", "label"]:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    return df_clean.groupby("sector")[numeric_cols].mean()

# computes fundamental score (0–100) through a stock's fundamentals relative to its sector average.
def get_fundamentals_score(stock_fund, sector_avg_row):
    metrics = ["de_ratio", "current_ratio", "roe", "gross_margin",
               "profit_margin", "revenue_growth", "eps_growth", "market_cap"]

    score = 0
    total = 0

    for m in metrics:
        stock_value = stock_fund.get(m)
        sector_value = sector_avg_row[m] if m in sector_avg_row else None

        if stock_value is None or sector_value is None:
            continue
        if pd.isna(stock_value) or pd.isna(sector_value):
            continue

        total += 1

        if m == "de_ratio":
            if stock_value < sector_value:
                score += 1
        else:
            if stock_value > sector_value:
                score += 1

    return round((score / total) * 100, 2) if total > 0 else np.nan

# Loads the trained model and returns Outperform or Underperform label, fundamentals score, symbol, and sector.
def classify_stock(symbol):
    if not os.path.exists(MODEL):
        return "Model not trained", None, symbol, None

    model = joblib.load(MODEL)

    df = pd.read_csv(DATA_PATH)
    sector_avg = df.groupby("sector").mean(numeric_only=True)

    stock_fund = get_fundamentals(symbol)
    sector = stock_fund.get("sector")

    if sector not in sector_avg.index:
        return "Insufficient Data", None, symbol, sector

    sector_row = sector_avg.loc[sector]
    stock_fund["fundamentals_score"] = get_fundamentals_score(stock_fund, sector_row)

    numeric_features = {
        k: pd.to_numeric(stock_fund.get(k), errors="coerce")
        for k in [
            "de_ratio", "current_ratio", "roe", "gross_margin",
            "profit_margin", "revenue_growth", "eps_growth",
            "market_cap", "fundamentals_score"
        ]
    }

    X = pd.DataFrame([numeric_features])

    pred = model.predict(X)[0]
    label = "Outperform" if pred == 1 else "Underperform"

    return label, numeric_features["fundamentals_score"], symbol, sector


if __name__ == "__main__":
    tickers = get_sp500_tickers()
    rows = []

    for symbol in tickers:
        label = get_label_rows(symbol, forward_days=180)
        rows.extend(label)

    df = pd.DataFrame(rows)

    sector_avg = get_sector_avg_fundamentals(df)
    df["fundamentals_score"] = df.apply(
        lambda x: get_fundamentals_score(
            x, sector_avg.loc[x["sector"]] if x["sector"] in sector_avg.index else {}
        ), axis=1
    )

    os.makedirs("dataset", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    feature_cols = [
        "de_ratio", "current_ratio", "roe", "gross_margin",
        "profit_margin", "revenue_growth", "eps_growth",
        "market_cap", "fundamentals_score",
    ]

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = df["label"]

    mask = X.notna().all(axis=1)
    X, y = X[mask], y[mask]

    for col in X.columns:
        lo, hi = X[col].quantile(0.01), X[col].quantile(0.99)
        X[col] = X[col].clip(lo, hi)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # LightGBM Architecture

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=20,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=0.3,
        class_weight="balanced",
        random_state=42,
    )

    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")

    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    joblib.dump(model, MODEL)

    #Evaluation Metrics

    results = [
        {"metric": "CV Accuracy (5-Fold)", "value": round(cv_scores.mean(), 4)},
        {"metric": "CV Std Deviation",     "value": round(cv_scores.std(),  4)},
        {"metric": "Test Accuracy",        "value": round(accuracy_score(y_test, y_pred),          4)},
        {"metric": "Precision — Underperform (0)", "value": round(report_dict["0"]["precision"], 4)},
        {"metric": "Recall    — Underperform (0)", "value": round(report_dict["0"]["recall"],    4)},
        {"metric": "F1-Score  — Underperform (0)", "value": round(report_dict["0"]["f1-score"],  4)},
        {"metric": "Support   — Underperform (0)", "value": int(report_dict["0"]["support"])},
        {"metric": "Precision — Outperform  (1)", "value": round(report_dict["1"]["precision"], 4)},
        {"metric": "Recall    — Outperform  (1)", "value": round(report_dict["1"]["recall"],    4)},
        {"metric": "F1-Score  — Outperform  (1)", "value": round(report_dict["1"]["f1-score"],  4)},
        {"metric": "Support   — Outperform  (1)", "value": int(report_dict["1"]["support"])},
    ]
    pd.DataFrame(results).to_csv("dataset/evaluation_results.csv", index=False)
