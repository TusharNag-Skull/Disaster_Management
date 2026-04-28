import argparse
from pathlib import Path

import pandas as pd

from src.preprocessing import ensure_dataframe_schema
from src.utils import load_artifact, validate_required_columns


def predict_batch(input_path: str, output_path: str, model_path: str) -> None:
    model = load_artifact(Path(model_path))
    df = pd.read_csv(input_path)
    df = ensure_dataframe_schema(df)
    validate_required_columns(df, ["text"])

    df["predicted_crisis"] = model.predict(df["text"])
    df["prediction_confidence"] = model.predict_proba(df["text"]).max(axis=1)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaster intelligence batch prediction")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model", default="models/disaster_classifier_pipeline.joblib")
    args = parser.parse_args()
    predict_batch(args.input, args.output, args.model)
