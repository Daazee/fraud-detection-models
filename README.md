# An Explainable Machine Learning-Based Framework for Fraud Detection in Digital Banking Transfers.

### MSc Final Year Project - Azeez Adedayo Adebayo

---

## Overview

This repository implements an explainable machine-learning pipeline for detecting fraudulent inter-account banking transfers. It evaluates supervised classifiers (Logistic Regression, Random Forest, XGBoost) and an unsupervised anomaly detector (Isolation Forest) and explores a hybrid approach that fuses the best supervised model with Isolation Forest scores.

---

## Key Features

- Configurable data ingestion (CSV/Excel)
- Preprocessing and feature engineering (winsorisation, log-transform, velocity features)
- Class-imbalance handling using SMOTE inside the training pipeline (no leakage)
- Supervised models: Logistic Regression, Random Forest, XGBoost
- Unsupervised model: Isolation Forest
- Hybrid model combining supervised predictions and anomaly scores (in development)

---

## Dataset

The primary dataset used is the IBM AMLSim example transactions dataset (sender-receiver structure). The raw CSVs are in `data/raw/`. A preprocessed CSV is stored under `data/processed/` once preprocessing completes.

Basic stats (approx):
- Total records: 1,048,575
- Fraudulent records: ~1,329 (≈0.13%)

---

## Project Structure

```
fraud-detection-models/
├── data/
│   ├── raw/                      # raw CSVs
│   └── processed/                # preprocessed CSVs (after running preprocessing)
├── notebooks/                    # EDA and quick-run notebook
├── src/
│   ├── ingestion/loaders.py      # data loaders
│   ├── models/
│   │   ├── evaluate.py           # evaluation utilities
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── isolation_forest.py
│   └── utils/preprocessing.py
├── main.py                       # pipeline entrypoint (preprocess/train/evaluate)
├── .env.example                  # example env variables
└── README.md
```

---

## Setup

1. Clone the project

```bash
git clone https://github.com/Daazee/fraud-detection-models.git
cd fraud-detection-models
```

2. Set up a virtual environment and install dependencies

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

3. Create a `.env` file in the project root. See `.env.example` for required variables.

4. See the notebook for a quick walkthrough: [Evaluation notebook](notebooks/ibm_bank_trans_01_eda.ipynb)

---

## Running (preprocessing → train/evaluate)

1. Run preprocessing to clean data and produce `data/processed/transactions_processed.csv`:

```bash
python main.py --preprocess
# Run preprocessing for data cleanup and feature engineering before training.
```

2. Train and evaluate models using the preprocessed data:

```bash
python main.py
# Use preprocessed data to train and evaluate models
```

---

## Evaluation & Metrics

Evaluation utilities live in `src/models/evaluate.py`. Open that file for details on computed metrics and plotting helpers.

- Primary metrics: PR-AUC (preferred for extreme class imbalance), F1-score
- Other metrics: Precision, Recall, ROC-AUC, FPR, FNR

Quick reference to evaluation file: [Evaluation utilities](src/models/evaluate.py)

---

## Models Implemented

- Logistic Regression (`src/models/logistic_regression.py`)
- Random Forest (`src/models/random_forest.py`) — currently the best performer
- XGBoost (`src/models/xgboost_model.py`)
- Isolation Forest (`src/models/isolation_forest.py`) — unsupervised anomaly detection
- Hybrid model: in development — intended to combine Random Forest (or best supervised model) and Isolation Forest anomaly scores

---

## Reproducibility Notes

- All supervised models use `random_state=42` for reproducibility.
- Training-only SMOTE is applied within the pipeline to avoid test leakage.
- If using a different Python interpreter or platform, re-create the virtual environment and re-install `requirements.txt`.

---

## Troubleshooting & Tips

- If `.\.venv\Scripts\activate` fails, switch from powershell to command prompt.
- If `main.py --preprocess` fails, check that the raw CSVs exist in `data/raw/` and that `.env` paths are correct.
- For memory-heavy operations (XGBoost training), consider increasing available RAM or using a smaller sample during experimentation.

---

## Next Steps (Roadmap)

1. Finalise hybrid model architecture and evaluation
2. Add SHAP explainability visualisations (`shap.TreeExplainer`) and summary reports

---

## Contact

This repository accompanies an ongoing MSc final-year project at the University of Hertfordshire. For questions, contact the author: Azeez Adedayo Adebayo.

---


