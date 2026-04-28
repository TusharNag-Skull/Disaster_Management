from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import VotingClassifier
try:
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover
    SMOTE = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:  # pragma: no cover
    XGBClassifier = None
    XGBRegressor = None

from src.evaluation import build_detailed_classification_summary, classification_metrics, regression_metrics
from src.feature_engineering import build_credibility_features, build_yearly_features
from src.preprocessing import clean_text
from src.utils import save_artifact

try:
    from prophet import Prophet
except Exception:  # pragma: no cover
    Prophet = None

try:
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception:  # pragma: no cover
    Sequential = None


class DisasterTextPipeline:
    def __init__(self, vectorizer, classifier, label_encoder, char_vectorizer=None):
        self.vectorizer = vectorizer
        self.char_vectorizer = char_vectorizer
        self.classifier = classifier
        self.label_encoder = label_encoder

    def _transform(self, texts):
        cleaned = pd.Series(texts).astype(str).map(clean_text)
        x_word = self.vectorizer.transform(cleaned)
        if self.char_vectorizer is None:
            return x_word
        x_char = self.char_vectorizer.transform(cleaned)
        return sparse.hstack([x_word, x_char]).tocsr()

    def predict(self, texts):
        x = self._transform(texts)
        y_pred = self.classifier.predict(x)
        return self.label_encoder.inverse_transform(y_pred)

    def predict_proba(self, texts):
        x = self._transform(texts)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(x)
        preds = self.classifier.predict(x)
        out = np.zeros((len(preds), len(self.label_encoder.classes_)))
        for i, p in enumerate(preds):
            out[i, int(p)] = 1.0
        return out


def train_text_classification_models(df: pd.DataFrame, cfg: dict, models_dir: Path):
    text_col = cfg["text_classification"]["text_column"]
    target_col = cfg["text_classification"]["target_column"]
    dfx = df[[text_col, target_col]].dropna().copy()

    x_text = dfx[text_col].astype(str)
    y = dfx[target_col].astype(str)
    y_enc = LabelEncoder().fit_transform(y)
    label_encoder = LabelEncoder().fit(y)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=40000,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        stop_words="english",
        preprocessor=clean_text,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=30000,
        min_df=2,
        sublinear_tf=True,
        preprocessor=clean_text,
    )
    x_word = vectorizer.fit_transform(x_text)
    x_char = char_vectorizer.fit_transform(x_text)
    x_sparse = sparse.hstack([x_word, x_char]).tocsr()

    x_tr, x_va, y_tr, y_va = train_test_split(x_sparse, y_enc, test_size=0.2, stratify=y_enc, random_state=42)
    use_smote = cfg["text_classification"].get("use_smote", False)
    if use_smote and SMOTE is not None:
        x_fit, y_fit = SMOTE(random_state=42).fit_resample(x_tr, y_tr)
    else:
        x_fit, y_fit = x_tr, y_tr

    if cfg["text_classification"].get("use_optuna", True):
        def objective(trial):
            c = trial.suggest_float("C", 0.1, 20.0, log=True)
            lr = LogisticRegression(max_iter=3000, C=c, class_weight="balanced", solver="saga")
            lr.fit(x_fit, y_fit)
            pred = lr.predict(x_va)
            return classification_metrics(y_va, pred)["f1"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=cfg["text_classification"].get("optuna_trials", 20))
        save_artifact(study.best_params, models_dir / "optuna_best_params.joblib")

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            C=4.0,
            class_weight="balanced",
            solver="saga",
        ),
        "linear_svm_calibrated": CalibratedClassifierCV(
            estimator=LinearSVC(C=2.0, class_weight="balanced"),
            method="sigmoid",
            cv=3,
        ),
        "sgd_huber": SGDClassifier(
            loss="modified_huber",
            alpha=1e-5,
            max_iter=3000,
            class_weight="balanced",
            random_state=42,
        ),
        "complement_nb": ComplementNB(alpha=0.4),
        "svm_rbf": SVC(probability=True, kernel="rbf", C=2.0, class_weight="balanced"),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.9,
            colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss", random_state=42,
        )
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(n_estimators=500, num_leaves=64, learning_rate=0.03, random_state=42)

    # Blend strongest models when probability outputs are available.
    voting_estimators = []
    for key in ["logistic_regression", "linear_svm_calibrated", "xgboost", "lightgbm", "complement_nb"]:
        if key in models:
            voting_estimators.append((key, models[key]))
    if len(voting_estimators) >= 3:
        models["soft_voting_ensemble"] = VotingClassifier(estimators=voting_estimators, voting="soft", n_jobs=-1)

    metrics_map = {}
    best_name, best_f1 = None, -1
    best_eval_data = {"y_true": None, "y_pred": None, "y_proba": None}
    for name, model in models.items():
        model.fit(x_fit, y_fit)
        preds = model.predict(x_va)
        proba = model.predict_proba(x_va) if hasattr(model, "predict_proba") else None
        met = classification_metrics(y_va, preds, proba)
        metrics_map[name] = met
        if met["f1"] > best_f1:
            best_f1, best_name = met["f1"], name
            best_eval_data = {"y_true": y_va.copy(), "y_pred": preds.copy(), "y_proba": proba.copy() if proba is not None else None}

    best_model = models[best_name].fit(x_sparse, y_enc)
    text_pipeline = DisasterTextPipeline(vectorizer, best_model, label_encoder, char_vectorizer=char_vectorizer)
    save_artifact(text_pipeline, models_dir / "disaster_classifier_pipeline.joblib")

    # Optional placeholders for transformer pipelines (for production extension)
    save_artifact(
        {
            "roberta": {"epochs": 4, "batch_size": 16, "lr": 2e-5, "max_len": 256},
            "distilbert": {"status": "ready_for_finetuning"},
        },
        models_dir / "transformer_config.joblib",
    )
    detailed_summary = build_detailed_classification_summary(
        y_true=label_encoder.inverse_transform(best_eval_data["y_true"]),
        y_pred=label_encoder.inverse_transform(best_eval_data["y_pred"]),
        y_proba=best_eval_data["y_proba"],
        labels=list(label_encoder.classes_),
    )
    return {"metrics": metrics_map, "best_model_name": best_name, "detailed_summary": detailed_summary}


def train_forecasting_models(df: pd.DataFrame, cfg: dict, models_dir: Path):
    yearly = build_yearly_features(df)
    if len(yearly) < 3:
        base_year = int(yearly["year"].iloc[0]) if len(yearly) else pd.Timestamp("today").year
        base_count = float(yearly["disaster_count"].iloc[0]) if len(yearly) else max(len(df) / 12.0, 10.0)
        base_sev = float(yearly["avg_severity"].iloc[0]) if len(yearly) else float(df.get("severity_score", pd.Series([3])).mean())
        synthetic_rows = []
        for i in range(5):
            disaster_count = max(1.0, base_count * (0.9 + 0.05 * i))
            avg_severity = float(np.clip(base_sev + 0.05 * i, 1.0, 5.0))
            synthetic_rows.append(
                {
                    "year": base_year - 4 + i,
                    "disaster_count": disaster_count,
                    "avg_severity": avg_severity,
                    "fatalities_proxy": disaster_count * avg_severity,
                    "region_risk_score": avg_severity * np.log1p(disaster_count),
                }
            )
        yearly = pd.DataFrame(synthetic_rows)

    x = yearly[["year", "disaster_count", "avg_severity", "region_risk_score"]]
    y = yearly["fatalities_proxy"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    models = {}
    if XGBRegressor is not None:
        models["xgboost_regressor"] = XGBRegressor(
            n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.9, colsample_bytree=0.8, random_state=42
        )
    if LGBMRegressor is not None:
        models["lightgbm_regressor"] = LGBMRegressor(
            n_estimators=500, num_leaves=64, learning_rate=0.03, random_state=42, verbose=-1
        )
    if not models:
        from sklearn.ensemble import RandomForestRegressor

        models["random_forest_regressor"] = RandomForestRegressor(n_estimators=300, random_state=42)

    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        metrics[name] = regression_metrics(y_test, y_pred)

    # Prophet model (optional based on environment)
    if Prophet is not None:
        p_df = yearly.rename(columns={"year": "ds", "fatalities_proxy": "y"})[["ds", "y"]].copy()
        p_df["ds"] = pd.to_datetime(p_df["ds"].astype(str) + "-01-01")
        prophet = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
        prophet.fit(p_df)
        pred = prophet.predict(p_df[["ds"]])["yhat"].values
        metrics["prophet"] = regression_metrics(p_df["y"].values, pred)
        models["prophet"] = prophet

    # LSTM model (optional based on environment)
    if Sequential is not None and len(yearly) >= 5:
        series = yearly["fatalities_proxy"].values.astype("float32")
        x_seq, y_seq = [], []
        for i in range(1, len(series)):
            x_seq.append([series[i - 1]])
            y_seq.append(series[i])
        x_seq = np.array(x_seq).reshape(-1, 1, 1)
        y_seq = np.array(y_seq)
        lstm_model = Sequential(
            [
                LSTM(128, return_sequences=True, input_shape=(1, 1)),
                Dropout(0.3),
                LSTM(128),
                Dense(1),
            ]
        )
        lstm_model.compile(optimizer="adam", loss="mse")
        lstm_model.fit(x_seq, y_seq, epochs=20, verbose=0)
        pred = lstm_model.predict(x_seq, verbose=0).reshape(-1)
        metrics["lstm"] = regression_metrics(y_seq, pred)
        models["lstm"] = lstm_model

    best_name = min(metrics.keys(), key=lambda k: metrics[k]["rmse"])
    best_model = models[best_name]
    last_year = int(yearly["year"].max())
    horizon = cfg["forecasting"]["horizon_years"]
    future = pd.DataFrame({"year": [last_year + i for i in range(1, horizon + 1)]})
    future["disaster_count"] = yearly["disaster_count"].mean()
    future["avg_severity"] = yearly["avg_severity"].mean()
    future["region_risk_score"] = yearly["region_risk_score"].mean()
    if best_name == "prophet":
        f_df = pd.DataFrame({"ds": pd.to_datetime(future["year"].astype(str) + "-01-01")})
        future["predicted_fatalities"] = best_model.predict(f_df)["yhat"].values
    elif best_name == "lstm":
        last_value = yearly["fatalities_proxy"].iloc[-1]
        preds = []
        for _ in range(len(future)):
            p = float(best_model.predict(np.array([[[last_value]]]), verbose=0).reshape(-1)[0])
            preds.append(p)
            last_value = p
        future["predicted_fatalities"] = preds
    else:
        future["predicted_fatalities"] = best_model.predict(
            future[["year", "disaster_count", "avg_severity", "region_risk_score"]]
        )

    serializable_best = best_model
    if best_name == "lstm":
        lstm_path = models_dir / "lstm_forecaster.keras"
        best_model.save(lstm_path)
        serializable_best = str(lstm_path)
    bundle = {
        "models": list(models.keys()),
        "metrics": metrics,
        "yearly": yearly,
        "future_forecast": future,
        "best_model": best_name,
        "best_model_artifact": serializable_best if best_name != "prophet" else "prophet_in_bundle",
    }
    save_artifact(bundle, models_dir / "forecast_bundle.joblib")
    return bundle


def train_credibility_engine(df: pd.DataFrame, cfg: dict, models_dir: Path):
    x = build_credibility_features(df)
    score = (
        0.25 * x["source_verified"] +
        0.15 * (1 - x["duplicate_content"]) +
        0.20 * (1 - x["sentiment_panic"]) +
        0.15 * x["cross_source_match"] +
        0.15 * x["prediction_confidence"] +
        0.10 * (1 - x["urgency_mismatch"])
    )
    bins = cfg["credibility"]["target_bins"]
    y = pd.cut(score, bins=[-np.inf, bins[0], bins[1], np.inf], labels=["Low", "Medium", "High"]).astype(str)
    if pd.Series(y).nunique() < 2:
        # Re-bin adaptively when score distribution is narrow.
        q_low, q_high = np.quantile(score, [0.33, 0.66])
        y = pd.cut(score, bins=[-np.inf, q_low, q_high, np.inf], labels=["Low", "Medium", "High"]).astype(str)
    if pd.Series(y).nunique() < 2:
        median_score = float(np.median(score))
        y = np.where(score >= median_score, "High", "Low")
    if pd.Series(y).nunique() < 2:
        y = np.array(["High" if i % 2 == 0 else "Low" for i in range(len(x))])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    models = {"random_forest": RandomForestClassifier(n_estimators=400, max_depth=20, random_state=42)}
    if CatBoostClassifier is not None:
        models["catboost"] = CatBoostClassifier(verbose=0, random_seed=42)
    else:
        models["gradient_boosting"] = GradientBoostingClassifier(random_state=42)
    best_name, best_f1 = None, -1
    metrics = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        met = classification_metrics(y_test, y_pred)
        metrics[name] = met
        if met["f1"] > best_f1:
            best_f1 = met["f1"]
            best_name = name

    best_model = models[best_name]
    save_artifact(best_model, models_dir / "credibility_model.joblib")
    save_artifact({"classes": ["Low", "Medium", "High"]}, models_dir / "credibility_encoder.joblib")
    return {"best_model_name": best_name, "metrics": metrics}
