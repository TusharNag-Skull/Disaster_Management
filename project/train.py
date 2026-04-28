import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation import format_detailed_classification_summary, save_metrics
from src.model_training import train_credibility_engine, train_forecasting_models, train_text_classification_models
from src.preprocessing import ensure_dataframe_schema
from src.utils import setup_logging


def run_training(config_path: str) -> None:
    root = Path(__file__).resolve().parent
    setup_logging(root / "logs")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_path = Path(cfg["data"]["dataset_path"])
    if not data_path.is_absolute():
        data_path = root.parent / data_path

    df = pd.read_csv(data_path)
    df = ensure_dataframe_schema(df)
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    text_out = train_text_classification_models(df, cfg, models_dir)
    forecast_out = train_forecasting_models(df, cfg, models_dir)
    cred_out = train_credibility_engine(df, cfg, models_dir)

    def save_metrics_with_fallback(metrics_map: dict, filename: str) -> None:
        target = models_dir / filename
        try:
            save_metrics(metrics_map, target)
        except PermissionError:
            fallback = models_dir / f"{target.stem}_latest{target.suffix}"
            save_metrics(metrics_map, fallback)
            print(f"Warning: '{target.name}' is locked, wrote metrics to '{fallback.name}' instead.")

    save_metrics_with_fallback(text_out["metrics"], "metrics.csv")
    save_metrics_with_fallback(forecast_out["metrics"], "forecast_metrics.csv")
    save_metrics_with_fallback(cred_out["metrics"], "credibility_metrics.csv")
    print("Training complete.")
    print("Best text model:", text_out["best_model_name"])
    print("Best forecast model:", forecast_out["best_model"])
    print("Best credibility model:", cred_out["best_model_name"])
    print()
    print(format_detailed_classification_summary(text_out["detailed_summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train disaster intelligence models")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_training(args.config)
