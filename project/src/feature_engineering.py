import numpy as np
import pandas as pd


def build_credibility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["source_verified"] = (df.get("resource_type", "").astype(str).str.len() > 0).astype(int)
    out["duplicate_content"] = df["text"].astype(str).duplicated(keep=False).astype(int)
    sentiment = df.get("sentiment", pd.Series(np.zeros(len(df)), index=df.index))
    out["sentiment_panic"] = np.clip(np.abs(sentiment.fillna(0)), 0, 1)
    diversity = df.groupby("text")["crisis_type"].transform("nunique") if "crisis_type" in df.columns else 1
    out["cross_source_match"] = np.clip(1 / np.maximum(diversity, 1), 0, 1)
    out["prediction_confidence"] = np.clip(1 - (df.get("severity_score", 0) / 10.0), 0, 1)
    out["urgency_mismatch"] = (
        (df.get("severity_score", 0) >= 4) & (out["prediction_confidence"] > 0.7)
    ).astype(int)
    return out


def build_yearly_features(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    if "timestamp" not in dfx.columns:
        dfx["timestamp"] = pd.Timestamp("today")
    dfx["year"] = pd.to_datetime(dfx["timestamp"], errors="coerce").dt.year.fillna(pd.Timestamp("today").year).astype(int)
    yearly = dfx.groupby("year", as_index=False).agg(
        disaster_count=("tweet_id", "count"),
        avg_severity=("severity_score", "mean"),
    )
    yearly["fatalities_proxy"] = yearly["disaster_count"] * yearly["avg_severity"]
    yearly["region_risk_score"] = yearly["avg_severity"] * np.log1p(yearly["disaster_count"])
    return yearly
