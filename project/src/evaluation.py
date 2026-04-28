from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, y_proba=None):
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_proba is not None:
        try:
            result["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except Exception:
            result["roc_auc"] = None
    else:
        result["roc_auc"] = None
    return result


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def save_metrics(metrics_map: dict, path: Path) -> None:
    rows = []
    for model_name, values in metrics_map.items():
        row = {"model": model_name}
        row.update(values)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_detailed_classification_summary(y_true, y_pred, y_proba=None, labels=None) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted({*np.unique(y_true), *np.unique(y_pred)})
    labels = list(labels)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    conf = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class accuracy here is class recall from confusion matrix.
    per_class_accuracy = {}
    for idx, label in enumerate(labels):
        total = conf[idx, :].sum()
        per_class_accuracy[label] = float(conf[idx, idx] / total) if total else 0.0

    summary = {
        "overall_accuracy": float(report["accuracy"]),
        "macro_avg": {
            "precision": float(report["macro avg"]["precision"]),
            "recall": float(report["macro avg"]["recall"]),
            "f1_score": float(report["macro avg"]["f1-score"]),
        },
        "weighted_avg": {
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1_score": float(report["weighted avg"]["f1-score"]),
        },
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1_score": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
                "accuracy": float(per_class_accuracy[label]),
            }
            for label in labels
        },
        "uncertainty": None,
    }

    if y_proba is not None and len(y_proba):
        y_proba = np.asarray(y_proba)
        max_conf = np.clip(np.max(y_proba, axis=1), 0.0, 1.0)
        uncertainty = 1.0 - max_conf
        correct_mask = y_true == y_pred
        summary["uncertainty"] = {
            "mean_prediction_confidence": float(max_conf.mean()),
            "mean_uncertainty": float(uncertainty.mean()),
            "std_uncertainty": float(uncertainty.std()),
            "mean_uncertainty_correct_predictions": float(uncertainty[correct_mask].mean()) if np.any(correct_mask) else None,
            "mean_uncertainty_incorrect_predictions": float(uncertainty[~correct_mask].mean()) if np.any(~correct_mask) else None,
        }
        if (
            summary["uncertainty"]["mean_uncertainty_correct_predictions"] is not None
            and summary["uncertainty"]["mean_uncertainty_incorrect_predictions"] is not None
        ):
            summary["uncertainty"]["uncertainty_discrimination"] = (
                summary["uncertainty"]["mean_uncertainty_incorrect_predictions"]
                - summary["uncertainty"]["mean_uncertainty_correct_predictions"]
            )

    return summary


def format_detailed_classification_summary(summary: dict, title: str = "Detailed Results Summary") -> str:
    lines = [title, "-" * max(len(title), 28), ""]
    lines.append("Final Metrics:")
    lines.append(f"Overall Accuracy: {summary['overall_accuracy']:.4f}")
    lines.append(
        "Macro Avg (P/R/F1): "
        f"{summary['macro_avg']['precision']:.4f} / "
        f"{summary['macro_avg']['recall']:.4f} / "
        f"{summary['macro_avg']['f1_score']:.4f}"
    )
    lines.append(
        "Weighted Avg (P/R/F1): "
        f"{summary['weighted_avg']['precision']:.4f} / "
        f"{summary['weighted_avg']['recall']:.4f} / "
        f"{summary['weighted_avg']['f1_score']:.4f}"
    )
    lines.append("")
    lines.append("Per-class Metrics:")
    for label, values in summary["per_class"].items():
        lines.append(
            f"  {label}: precision={values['precision']:.4f}, "
            f"recall={values['recall']:.4f}, f1={values['f1_score']:.4f}, "
            f"accuracy={values['accuracy']:.4f}, support={values['support']}"
        )

    uncertainty = summary.get("uncertainty")
    if uncertainty:
        lines.append("")
        lines.append("Uncertainty Analysis:")
        lines.append(f"  Mean Prediction Confidence: {uncertainty['mean_prediction_confidence']:.4f}")
        lines.append(f"  Mean Uncertainty: {uncertainty['mean_uncertainty']:.4f}")
        lines.append(f"  Std Uncertainty: {uncertainty['std_uncertainty']:.4f}")
        if uncertainty.get("mean_uncertainty_correct_predictions") is not None:
            lines.append(
                "  Mean uncertainty (correct predictions): "
                f"{uncertainty['mean_uncertainty_correct_predictions']:.4f}"
            )
        if uncertainty.get("mean_uncertainty_incorrect_predictions") is not None:
            lines.append(
                "  Mean uncertainty (incorrect predictions): "
                f"{uncertainty['mean_uncertainty_incorrect_predictions']:.4f}"
            )
        if uncertainty.get("uncertainty_discrimination") is not None:
            lines.append(f"  Uncertainty discrimination: {uncertainty['uncertainty_discrimination']:.4f}")

    return "\n".join(lines)
