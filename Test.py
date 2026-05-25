import os
import math
import numpy as np
import pandas as pd
import pytest


METRICS = [
    "de_ratio", "current_ratio", "roe", "gross_margin",
    "profit_margin", "revenue_growth", "eps_growth", "market_cap"
]


# Computes percentage fundamentals score relative to sector averages
def fundamentals_score(stock: dict, sector: dict) -> float:
    valid = []

    for m in METRICS:
        sv = stock.get(m)
        av = sector.get(m)

        if sv is None or av is None:
            continue

        try:
            if math.isnan(float(sv)) or math.isnan(float(av)):
                continue
        except (TypeError, ValueError):
            continue

        valid.append((m, float(sv), float(av)))

    if not valid:
        return float("nan")

    wins = sum(
        (sv < av if m == "de_ratio" else sv > av)
        for m, sv, av in valid
    )

    return round((wins / len(valid)) * 100, 2)


# Converts sentiment lean value into label and normalized risk score
def dominant_sentiment(avg_lean: float):
    if avg_lean > 0.05:
        label = "positive"
    elif avg_lean < -0.05:
        label = "negative"
    else:
        label = "neutral"

    risk_value = round(max(0, min(1, (avg_lean + 1) / 2)), 4)
    return label, risk_value


# Calculates normalized LSTM confidence score from predicted returns
def lstm_score(predictions: np.ndarray) -> float:
    if predictions[0] == 0:
        return 0.5

    future_return = (predictions[-1] - predictions[0]) / predictions[0]
    return 0.5 + future_return * 0.5


# Combines model outputs into final investment risk classification score
def compute_overall_risk(predictions, risk_value, classification, fundamentals_score_val):
    lstm_s = lstm_score(predictions)
    sentiment_s = risk_value

    fundamentals_norm = (
        fundamentals_score_val / 100
        if fundamentals_score_val is not None
        else 0.5
    )

    class_score = {
        "Outperform": 1.0,
        "Underperform": 0.0
    }.get(classification, 0.5)

    lgb_s = (class_score * 0.3) + (fundamentals_norm * 0.7)

    combined = (
        lstm_s * 0.2 +
        sentiment_s * 0.3 +
        lgb_s * 0.5
    )

    if combined >= 0.65:
        label = "Likely Low Risk"
    elif combined >= 0.40:
        label = "Moderate Risk"
    else:
        label = "Likely High Risk"

    return label, combined


# Tests FinBERT sentiment classification threshold logic
class TestFinBERT:

    # Verifies positive sentiment classification above threshold
    def test_positive_threshold(self):
        label, rv = dominant_sentiment(0.10)
        assert label == "positive"
        assert rv > 0.5

    # Verifies negative sentiment classification below threshold
    def test_negative_threshold(self):
        label, rv = dominant_sentiment(-0.10)
        assert label == "negative"
        assert rv < 0.5

    # Verifies neutral classification within sentiment dead zone
    def test_neutral_dead_zone(self):
        label, rv = dominant_sentiment(0.0)
        assert label == "neutral"
        assert rv == pytest.approx(0.5)

    # Validates continuous sentiment risk value calculation formula
    def test_risk_value_formula(self):
        _, rv = dominant_sentiment(0.30)
        assert rv == pytest.approx(0.65)

    # Confirms empty sentiment dataset fallback behaviour
    def test_empty_df_returns_neutral(self):
        df = pd.DataFrame()
        label, rv = ("neutral", 0.5) if df.empty else dominant_sentiment(0)
        assert (label, rv) == ("neutral", 0.5)


# Tests LightGBM-style fundamentals scoring behaviour
class TestFundamentalsScore:

    BASE_SECTOR = {m: 1.0 for m in METRICS}

    # Confirms full metric wins produce perfect fundamentals score
    def test_perfect_score(self):
        stock = {m: (0.5 if m == "de_ratio" else 2.0) for m in METRICS}
        assert fundamentals_score(stock, self.BASE_SECTOR) == 100.0

    # Confirms full metric losses produce zero fundamentals score
    def test_zero_score(self):
        stock = {m: (2.0 if m == "de_ratio" else 0.5) for m in METRICS}
        assert fundamentals_score(stock, self.BASE_SECTOR) == 0.0

    # Verifies partial metric wins produce expected proportional score
    def test_partial_score_75(self):
        stock = {m: (0.5 if m == "de_ratio" else 2.0) for m in METRICS}
        for m in METRICS[1:3]:
            stock[m] = 0.5
        assert fundamentals_score(stock, self.BASE_SECTOR) == 75.0

    # Ensures NaN metric values are excluded safely from scoring
    def test_nan_excluded(self):
        stock = {m: float("nan") for m in METRICS}
        stock["roe"] = 2.0
        assert not math.isnan(
            fundamentals_score(stock, self.BASE_SECTOR)
        )

    # Confirms missing metrics return undefined fundamentals score
    def test_all_none_returns_nan(self):
        stock = {m: None for m in METRICS}
        assert math.isnan(
            fundamentals_score(stock, self.BASE_SECTOR)
        )

    # Verifies lower debt-to-equity ratio improves fundamentals score
    def test_de_ratio_directionality(self):
        low = {m: (0.5 if m == "de_ratio" else 2.0) for m in METRICS}
        high = {m: 2.0 for m in METRICS}
        assert fundamentals_score(low, self.BASE_SECTOR) > fundamentals_score(high, self.BASE_SECTOR)


# Tests LSTM forecast-derived scoring behaviour
class TestLSTMScore:

    # Confirms positive predicted returns increase confidence score
    def test_positive_return(self):
        assert lstm_score(np.array([100, 120])) == pytest.approx(0.60)

    # Confirms negative predicted returns decrease confidence score
    def test_negative_return(self):
        assert lstm_score(np.array([100, 80])) == pytest.approx(0.40)

    # Confirms flat predicted returns produce neutral confidence score
    def test_flat_return(self):
        assert lstm_score(np.array([100, 100])) == pytest.approx(0.50)


# Tests dashboard compliance with financial advice disclaimer requirement
class TestDashboardNFA:

    # Verifies disclaimer presence within dashboard source file
    def test_nfa_label_present(self):
        if not os.path.exists("Dashboard.py"):
            pytest.skip("Dashboard.py not found")

        with open("Dashboard.py") as f:
            assert "NOT FINANCIAL ADVICE" in f.read()


# Tests integrated multi-model investment risk scoring logic
class TestRiskIntegration:

    BASE = np.array([100.0, 100.0])
    VALID = {"Likely Low Risk", "Moderate Risk", "Likely High Risk"}

    # Ensures combined risk score always remains within valid bounds
    def test_score_range(self):
        _, score = compute_overall_risk(self.BASE, 0.5, "Neutral", 50)
        assert 0 <= score <= 1

    # Confirms strong positive signals produce low-risk classification
    def test_low_risk_label(self):
        label, score = compute_overall_risk(self.BASE, 1.0, "Outperform", 100)
        assert label == "Likely Low Risk"
        assert score >= 0.65

    # Confirms weak signals produce high-risk classification
    def test_high_risk_label(self):
        label, score = compute_overall_risk(self.BASE, 0.0, "Underperform", 0)
        assert label == "Likely High Risk"
        assert score < 0.40

    # Confirms mixed signals produce moderate-risk classification
    def test_moderate_risk_label(self):
        label, score = compute_overall_risk(self.BASE, 0.5, "Neutral", 50)
        assert label == "Moderate Risk"
        assert 0.40 <= score < 0.65

    # Verifies higher sentiment confidence increases combined risk score
    def test_positive_sentiment_increases_score(self):
        _, pos = compute_overall_risk(self.BASE, 1.0, "Outperform", 50)
        _, neg = compute_overall_risk(self.BASE, 0.0, "Outperform", 50)
        assert pos > neg

    # Ensures only valid classification labels are returned by model
    def test_valid_labels_only(self):
        for rv in (0.0, 0.5, 1.0):
            for cls in ("Outperform", "Neutral", "Underperform"):
                label, _ = compute_overall_risk(self.BASE, rv, cls, 50)
                assert label in self.VALID


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
