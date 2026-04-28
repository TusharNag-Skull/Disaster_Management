# Credibility-Aware Multisource Disaster Intelligence System

Production-ready Python AI project for end-to-end disaster intelligence:
- NLP disaster text classification with imbalance handling
- Disaster trend and fatalities forecasting
- Credibility scoring engine (0-100 + Low/Medium/High)
- Streamlit dashboard with analytics and SHAP explainability

## Structure

```text
project/
|-- app.py
|-- train.py
|-- predict.py
|-- requirements.txt
|-- config.yaml
|-- src/
|   |-- preprocessing.py
|   |-- feature_engineering.py
|   |-- model_training.py
|   |-- evaluation.py
|   |-- utils.py
|-- models/
|-- notebooks/
|-- tests/
```

## Setup

```bash
pip install -r project/requirements.txt
```

## Train

```bash
python project/train.py --config project/config.yaml
```

## Predict

```bash
python project/predict.py --input public_crisis_dataset.csv --output predictions.csv
```

## Streamlit Dashboard

```bash
streamlit run project/app.py
```

## Notes

- The provided notebook path is configurable in `project/config.yaml`.
- If notebook-derived time-series features are unavailable, the training pipeline uses yearly aggregates from CSV.
- Transformer configs for RoBERTa/DistilBERT are included as production extension hooks.
