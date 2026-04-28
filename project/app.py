from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.preprocessing import ensure_dataframe_schema, parse_text_input
from src.utils import load_artifact

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT.parent / "public_crisis_dataset.csv"

st.set_page_config(page_title="Disaster Intelligence", layout="wide")
st.markdown(
    """
<style>
.stApp { background-color: #0b1220; color: #dbeafe; }
[data-testid="stSidebar"] { background-color: #0f172a; }
</style>
""",
    unsafe_allow_html=True,
)


def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.DataFrame(columns=["text", "crisis_type", "severity_score"])
    return ensure_dataframe_schema(df)


def load_models():
    out = {}
    for key, file_name in {
        "clf": "disaster_classifier_pipeline.joblib",
        "forecast": "forecast_bundle.joblib",
        "cred": "credibility_model.joblib",
    }.items():
        path = MODELS_DIR / file_name
        out[key] = load_artifact(path) if path.exists() else None
    return out


def _run_training_from_app() -> bool:
    cmd = [sys.executable, str(ROOT / "train.py"), "--config", str(ROOT / "config.yaml")]
    with st.spinner("Training models... this may take a few minutes."):
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if proc.returncode == 0:
        st.success("Model training completed. Reloading artifacts.")
        return True
    st.error("Training failed. Check details below.")
    st.code(proc.stderr or proc.stdout or "No output captured.")
    return False


def _missing_artifact_ui():
    st.warning("Required model artifacts are missing.")
    st.caption("Click below to train all models and enable Prediction/Credibility/Forecast pages.")
    if st.button("Train Models Now", type="primary"):
        if _run_training_from_app():
            st.rerun()


def page_home(df: pd.DataFrame):
    st.title("Credibility-Aware Multisource Disaster Intelligence System")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Reports", len(df))
    c2.metric("Unique Crisis Types", int(df["crisis_type"].nunique()))
    c3.metric("Avg Severity", f"{df['severity_score'].mean():.2f}")


def page_upload_predict(models):
    st.header("Upload CSV/Text")
    clf = models["clf"]
    if clf is None:
        _missing_artifact_ui()
        return
    text = st.text_area("Real-time text prediction")
    if text.strip():
        one = parse_text_input(text)
        pred = clf.predict(one["text"])[0]
        conf = float(clf.predict_proba(one["text"]).max())
        st.success(f"Prediction: {pred} | Confidence: {conf:.3f}")

    csv_file = st.file_uploader("Batch CSV prediction", type=["csv"])
    if csv_file is not None:
        bdf = pd.read_csv(csv_file)
        bdf = ensure_dataframe_schema(bdf)
        bdf["predicted_crisis"] = clf.predict(bdf["text"])
        bdf["prediction_confidence"] = clf.predict_proba(bdf["text"]).max(axis=1)
        st.dataframe(bdf.head(20), use_container_width=True)
        st.download_button("Download predictions", bdf.to_csv(index=False), "predictions.csv", "text/csv")


def page_disaster_prediction(df: pd.DataFrame, models):
    st.header("Disaster Prediction")
    clf = models["clf"]
    if clf is None:
        _missing_artifact_ui()
        return
    sample = st.selectbox("Choose sample text", df["text"].astype(str).head(500).tolist())
    if st.button("Predict class"):
        pred_label = clf.predict([sample])[0]
        proba = clf.predict_proba([sample])[0]
        confidence = float(np.max(proba))

        c1, c2 = st.columns([1.2, 1.0])
        c1.markdown(
            f"""
            <div style="padding:14px 16px;border-radius:10px;background:#13203a;border:1px solid #223b67;">
                <div style="font-size:0.85rem;color:#93c5fd;margin-bottom:6px;">Predicted Crisis Type</div>
                <div style="font-size:1.25rem;font-weight:700;color:#f8fafc;">{pred_label.replace('_', ' ').title()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c2.metric("Confidence", f"{confidence * 100:.2f}%")
        st.progress(min(max(confidence, 0.0), 1.0), text=f"Model confidence: {confidence:.3f}")

        if hasattr(clf, "label_encoder"):
            top_idx = np.argsort(proba)[::-1][:3]
            top_labels = [str(clf.label_encoder.classes_[i]) for i in top_idx]
            top_scores = [float(proba[i]) for i in top_idx]
            top_df = pd.DataFrame({"Crisis Type": top_labels, "Probability": top_scores})
            st.caption("Top predictions")
            st.dataframe(
                top_df.style.format({"Probability": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )


def page_credibility(models):
    st.header("Credibility Score")
    cred = models["cred"]
    if cred is None:
        _missing_artifact_ui()
        return
    source_verified = st.checkbox("Source verified", True)
    duplicate_content = st.checkbox("Duplicate content", False)
    sentiment_panic = st.slider("Sentiment panic", 0.0, 1.0, 0.4)
    cross_source_match = st.slider("Cross-source match", 0.0, 1.0, 0.6)
    prediction_confidence = st.slider("Prediction confidence", 0.0, 1.0, 0.7)
    urgency_mismatch = st.checkbox("Urgency mismatch", False)
    x = pd.DataFrame(
        [{
            "source_verified": int(source_verified),
            "duplicate_content": int(duplicate_content),
            "sentiment_panic": sentiment_panic,
            "cross_source_match": cross_source_match,
            "prediction_confidence": prediction_confidence,
            "urgency_mismatch": int(urgency_mismatch),
        }]
    )
    probs = cred.predict_proba(x)[0]
    label = cred.predict(x)[0]
    score = float(np.dot(probs, np.array([25, 60, 95])[: len(probs)]))
    st.metric("Credibility Score", f"{score:.1f}/100")
    st.write(f"Level: **{label}**")


def page_forecast(models):
    st.header("Forecast Dashboard")
    bundle = models["forecast"]
    if bundle is None:
        _missing_artifact_ui()
        return
    future = bundle["future_forecast"]
    st.plotly_chart(px.line(future, x="year", y="predicted_fatalities", title="Next 5-year fatalities forecast"), use_container_width=True)


def page_analytics(df: pd.DataFrame):
    st.header("Visual Analytics")
    selected_crisis = st.multiselect(
        "Filter crisis type",
        options=sorted(df["crisis_type"].dropna().unique().tolist()),
        default=sorted(df["crisis_type"].dropna().unique().tolist()),
    )
    sev_min, sev_max = st.slider("Severity range", 0.0, 5.0, (0.0, 5.0), step=0.1)
    filtered = df[df["crisis_type"].isin(selected_crisis)].copy()
    filtered = filtered[(filtered["severity_score"] >= sev_min) & (filtered["severity_score"] <= sev_max)]

    if filtered.empty:
        st.info("No rows match current filters.")
        return

    dist = filtered["crisis_type"].value_counts().reset_index()
    dist.columns = ["crisis_type", "count"]
    st.plotly_chart(px.bar(dist, x="crisis_type", y="count"), use_container_width=True)

    if {"latitude", "longitude"}.issubset(filtered.columns):
        mdf = filtered.dropna(subset=["latitude", "longitude"]).copy()
        if mdf.empty:
            st.info("No valid coordinates after filtering.")
            return

        lat_min, lat_max = float(mdf["latitude"].min()), float(mdf["latitude"].max())
        lon_min, lon_max = float(mdf["longitude"].min()), float(mdf["longitude"].max())
        st.caption(
            f"Coverage: lat {lat_min:.2f} to {lat_max:.2f}, lon {lon_min:.2f} to {lon_max:.2f}. "
            "Current dataset is geographically concentrated, so points cluster in one region."
        )

        map_mode = st.radio("Map mode", ["Geo Scatter", "Density Heatmap"], horizontal=True)
        if map_mode == "Geo Scatter":
            hover_cols = [c for c in ["tweet_id", "severity_score", "resource_type"] if c in mdf.columns]
            fig = px.scatter_geo(
                mdf,
                lat="latitude",
                lon="longitude",
                color="crisis_type",
                size="severity_score",
                hover_data=hover_cols,
                projection="natural earth",
                title="Interactive Global Disaster Points",
            )
            fig.update_layout(height=560)
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.density_mapbox(
                mdf,
                lat="latitude",
                lon="longitude",
                z="severity_score",
                radius=14,
                center={"lat": float(mdf["latitude"].mean()), "lon": float(mdf["longitude"].mean())},
                zoom=2,
                map_style="carto-darkmatter",
                title="Disaster Intensity Heatmap",
            )
            fig.update_layout(height=560)
            st.plotly_chart(fig, use_container_width=True)


def page_metrics():
    st.header("Model Metrics")
    metrics_file = MODELS_DIR / "metrics.csv"
    if metrics_file.exists():
        mdf = pd.read_csv(metrics_file)
        st.dataframe(mdf, use_container_width=True)
        st.plotly_chart(px.bar(mdf, x="model", y="f1", title="F1 score comparison"), use_container_width=True)
    else:
        st.info("No metrics file found.")


def main():
    df = load_data()
    models = load_models()
    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Upload CSV/Text",
            "Disaster Prediction",
            "Credibility Score",
            "Forecast Dashboard",
            "Visual Analytics",
            "Model Metrics",
        ],
    )
    if page == "Home":
        page_home(df)
    elif page == "Upload CSV/Text":
        page_upload_predict(models)
    elif page == "Disaster Prediction":
        page_disaster_prediction(df, models)
    elif page == "Credibility Score":
        page_credibility(models)
    elif page == "Forecast Dashboard":
        page_forecast(models)
    elif page == "Visual Analytics":
        page_analytics(df)
    elif page == "Model Metrics":
        page_metrics()


if __name__ == "__main__":
    main()
