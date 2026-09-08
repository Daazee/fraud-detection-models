# An Explainable Machine Learning-Based Framework for Fraud Detection in Digital Banking Transfers.

### MSc Final Year Project - Azeez Adedayo Adebayo

---

## Overview

This repository implements an explainable machine-learning pipeline for detecting fraudulent inter-account banking transfers. It evaluates supervised classifiers (Logistic Regression, Random Forest, XGBoost) against a most-frequent dummy baseline, an unsupervised anomaly detector (Isolation Forest), and a **hybrid** stage that fuses the best supervised model with Isolation Forest anomaly scores. The best supervised model is then interpreted with **SHAP**. All modelling uses a **chronological (temporal) train/validation/test split** so that models are only ever asked to predict fraud in a period later than the one they were trained on.

---

## Key Features

- Configurable data ingestion (CSV/Excel) via `src/ingestion/loaders.py`, driven by `.env` paths
- Transaction-to-account merge that attaches sender and receiver account attributes to each transfer (`src/utils/preprocessing.py`)
- Preprocessing and feature engineering: chronological sort, velocity/count features, correlation pruning, fraud-aware outlier winsorisation, log-transform of skewed features, one-hot encoding
- **Temporal split** (no shuffling): first 80% of time-ordered rows for train+validation, last 20% held out for test; the train+validation block is itself split 80/20 chronologically
- Class-imbalance handling with SMOTE applied **inside** the training pipeline only (no leakage into validation/test)
- Supervised models: Logistic Regression, Random Forest, XGBoost, plus a most-frequent `DummyClassifier` baseline
- Unsupervised model: Isolation Forest, with `contamination` set to the observed training-set fraud rate
- **Hybrid model**: `WeightedAverageHybrid` (`src/models/weighted_hybrid_model.py`) fuses the best supervised model's fraud probability with the sign-flipped, min-max-normalised Isolation Forest score; the blend weight `w` is tuned for validation PR-AUC and the decision threshold for validation F1
- Additional fusion strategies prototyped in the hybrid notebook: feature-level fusion, cascade (uncertainty-band) fusion, threshold OR-rule, and soft voting
- **Explainability**: SHAP `TreeExplainer` over the Random Forest (`src/models/shap_interpretation.py`) - global mean-|SHAP| ranking, beeswarm/summary plots, and per-record waterfall plots
- Complementarity analysis quantifying how much fraud each of Random Forest and Isolation Forest catches that the other misses

---

## Dataset

The project uses two raw files from an IBM AMLSim-style transaction simulation, both in `data/raw/`:

| File | Grain | Key columns |
|------|-------|-------------|
| `ibm_bank_transactions.csv` | one row per transfer | `TX_ID`, `SENDER_ACCOUNT_ID`, `RECEIVER_ACCOUNT_ID`, `TX_TYPE`, `TX_AMOUNT`, `TIMESTAMP`, `IS_FRAUD`, `ALERT_ID` |
| `ibm_bank_accounts.csv` | one row per account | `ACCOUNT_ID`, `CUSTOMER_ID`, `INIT_BALANCE`, `COUNTRY`, `ACCOUNT_TYPE`, `IS_FRAUD`, `TX_BEHAVIOR_ID` |

`wrangle_data()` inner-joins the transactions to the accounts table twice - once on the sender, once on the receiver - and writes the modelling table to `data/processed/merged_transactions_accounts_processed.csv`.

Basic stats:
- Total transactions: 1,048,575
- Fraudulent transactions: 1,329 (≈0.13% overall; ≈0.1331% in the temporal training block)
- Modelling features after preprocessing: `TX_AMOUNT`, `SENDER_TX_BEHAVIOR_ID`, `RECEIVER_INIT_BALANCE`, `RECEIVER_TX_BEHAVIOR_ID`, `SENDER_STEP_COUNT`, `RECEIVER_STEP_COUNT`, `SENDER_RECEIVER_PAIR_COUNT`, `SENDER_ACCOUNT_ID`, `RECEIVER_ACCOUNT_ID` (target: `IS_FRAUD`)

`TX_ID`, `ALERT_ID` (label-leaking), `TX_TYPE` (constant) and `TIMESTAMP` (used only to order rows and derive velocity features) are dropped during preprocessing.

---

## Project Structure

```
fraud-detection-models/
├── data/
│   ├── raw/                                     # ibm_bank_transactions.csv, ibm_bank_accounts.csv
│   └── processed/                               # merged_transactions_accounts_processed.csv (built by --preprocess)
├── notebooks/
│   ├── ibm_bank_accounts_01_eda.ipynb           # EDA of the accounts file
│   ├── ibm_bank_trans_01_eda.ipynb              # EDA of the transactions file (imbalance, timestamps, features)
│   ├── ibm_bank_trans_02_merge_features.ipynb   # prototypes the sender/receiver merge + feature set
│   ├── ibm_bank_trans_03_model_training.ipynb   # supervised + Isolation Forest training and validation comparison
│   ├── ibm_bank_trans_04_hybrid_models.ipynb    # five fusion strategies
│   ├── ibm_bank_trans_05_complementarity_analysis.ipynb  # RF vs Isolation Forest missed-fraud overlap
│   └── ibm_bank_trans_06_shap_explainability.ipynb       # SHAP interpretation of the Random Forest
├── src/
│   ├── ingestion/loaders.py                     # CSV/Excel loader, resolves paths from .env
│   ├── models/
│   │   ├── evaluate.py                          # metrics, threshold tuning, model-comparison helpers
│   │   ├── logistic_regression.py               # SMOTE + StandardScaler + LogisticRegression pipeline
│   │   ├── random_forest.py                     # SMOTE + RandomForestClassifier pipeline (best supervised model)
│   │   ├── xgboost_model.py                     # SMOTE + XGBClassifier pipeline
│   │   ├── isolation_forest.py                  # unsupervised IsolationForest (contamination = train fraud rate)
│   │   ├── weighted_hybrid_model.py             # WeightedAverageHybrid: supervised + Isolation Forest score fusion
│   │   └── shap_interpretation.py               # SHAP TreeExplainer helpers + stratified test sampling
│   ├── utils/preprocessing.py                   # wrangle_data(): merge, feature engineering, cleaning
│   └── main.py                                  # pipeline entrypoint (preprocess / train / evaluate / hybrid)
├── reports/                                     # written report (Interim/Final Project Report - IPR, FPR) - PDF/DOCX
├── requirements.txt
├── .env.example                                 # example env variables
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

3. Create a `.env` file in the project root. See `.env.example`:

```
transactions_file_path="data/raw/ibm_bank_transactions.csv"
accounts_file_path="data/raw/ibm_bank_accounts.csv"
```

4. For a quick walkthrough, work through the notebooks in numeric order (EDA → merge → model training → hybrid → complementarity → SHAP).

---

## Running (preprocessing → train / evaluate / hybrid)

1. Build `data/processed/merged_transactions_accounts_processed.csv` (merge + feature engineering + cleaning):

```bash
python src\main.py --preprocess
```

2. Train and evaluate all models on the temporal split. `main.py` runs preprocessing automatically if the processed file is missing:

```bash
python src\main.py
```

`python src\main.py` (no flag) does the following in order:
- loads the processed data and builds the chronological train / validation / test split
- trains Logistic Regression, Random Forest, XGBoost, Isolation Forest and a most-frequent Dummy baseline on the training block
- compares them on the **validation** block and selects the best classifier by combined PR-AUC + F1 rank (`identify_best_model`)
- trains `WeightedAverageHybrid` on the validation block (tunes the fusion weight and threshold there)
- runs a single, final evaluation of the best classifier, Isolation Forest and the hybrid on the **held-out test** block (`print_final_results`)

---

## Temporal Split (important for reproducibility)

Every modelling script and notebook uses the **same** deterministic, shuffle-free split, computed after sorting rows by `TIMESTAMP`:

| Block | Rows | Definition |
|-------|------|------------|
| Train | 671,088 | first 64% of time-ordered rows (first 80% of the train+validation block) |
| Validation | 167,772 | next 16% (last 20% of the train+validation block) |
| Test | 209,715 | last 20% of time-ordered rows, held out until the final evaluation |

Weight/threshold tuning, model selection and SHAP threshold selection are all done on **validation** only. The test block is scored once per model at the end. `IsolationForest(contamination=...)` is set to the training-block fraud rate (0.001331).

---

## Evaluation & Metrics

Evaluation utilities live in `src/models/evaluate.py`:

- `evaluate(model, X, y, name, threshold=None)` - probability/decision scoring for classifiers, with an optional fixed (validation-tuned) threshold
- `evaluate_anomaly(model, X, y, name)` - maps Isolation Forest's `-1/1` output to `1/0` and negates `decision_function` so higher = more suspicious
- `find_best_threshold(y, score, beta)` - sweeps the precision-recall curve for the F-beta-optimal threshold (fit on validation)
- `find_anomaly_optimal_threshold(y, score)` - F1-optimal threshold on already-inverted anomaly scores
- `identify_best_model(results)` / `print_final_results(results)` - model-comparison tables ranked by PR-AUC + F1

Metrics reported for every model: Precision, Recall, F1-score, FPR, FNR, ROC-AUC, PR-AUC, plus the confusion matrix. **PR-AUC is the primary metric** (preferred under extreme class imbalance); FNR (missed fraud) is the secondary priority.

---

## Models Implemented

| Model | File | Notes |
|-------|------|-------|
| Logistic Regression | `src/models/logistic_regression.py` | SMOTE → StandardScaler → `LogisticRegression(max_iter=1000)` |
| Random Forest | `src/models/random_forest.py` | SMOTE → `RandomForestClassifier(n_estimators=100)` - **best supervised performer**, anchor for the hybrid |
| XGBoost | `src/models/xgboost_model.py` | SMOTE → `XGBClassifier(eval_metric="logloss")` - retained as a substitute when minimising missed fraud is prioritised |
| Isolation Forest | `src/models/isolation_forest.py` | unsupervised; `contamination` = training fraud rate; used as a complementary anomaly signal |
| Hybrid - Weighted Average | `src/models/weighted_hybrid_model.py` | `final = w·clf_norm + (1−w)·iso_norm`; `w` swept on [0, 1] step 0.05 for validation PR-AUC, threshold swept for validation F1 |

The hybrid notebook (`ibm_bank_trans_04_hybrid_models.ipynb`) additionally prototypes four more fusion strategies and selects the best on validation before the single test evaluation:

- **Feature-Level Fusion** - append the inverted Isolation Forest score as an extra Random Forest input feature
- **Cascade** - Random Forest decides by default; Isolation Forest is consulted only for records whose normalised RF score falls in an uncertain band (band tuned on validation PR-AUC)
- **Threshold OR-Rule** - flag as fraud if *either* model exceeds its own F1-tuned threshold
- **Soft Voting** - equal-weight (0.5/0.5) average of the normalised scores, thresholded on validation F1

---

## Explainability (SHAP)

`src/models/shap_interpretation.py` provides:

- `sample_for_shap(X_test, y_test, sample_size=5000)` - stratified sub-sample of the test block (`random_state=42`)
- `explain_model(model, X_shap)` - `shap.TreeExplainer` with `feature_perturbation="tree_path_dependent"` and no background dataset, run on the tree estimator extracted from the pipeline (SMOTE is a train-only step and is bypassed)

`ibm_bank_trans_06_shap_explainability.ipynb` uses these to produce the global mean-|SHAP| feature ranking (fraud class), beeswarm/summary plots, per-record waterfall plots for true/false positives and negatives, and FPR/FNR on the SHAP sample at the validation-tuned threshold.

---

## Reproducing the Report Tables and Figures

The table/figure numbers below follow the written report's Results chapter; if the numbering shifts in a later draft, match on the caption/description. All results assume the processed CSV has been built (`python src\main.py --preprocess`) and the temporal split described above.

| Report item | Produced by | How it is generated |
|-------------|-------------|---------------------|
| **Table 4.1** - Model performance comparison (Logistic Regression, Random Forest, XGBoost, Isolation Forest, Dummy baseline) | `notebooks/ibm_bank_trans_03_model_training.ipynb` (validation comparison cell, `identify_best_model`) and `python src\main.py` (validation `── Model Comparison ──` block) | Each model is trained on the training block and scored on the validation block via `evaluate` / `evaluate_anomaly`; metrics are precision, recall, F1, FPR, FNR, ROC-AUC, PR-AUC |
| **Table 4.2** - Hybrid fusion strategy comparison on validation (Weighted Average, Feature-Level Fusion, Cascade, Threshold OR-Rule, Soft Voting) | `notebooks/ibm_bank_trans_04_hybrid_models.ipynb` (`validation_results` / `identify_best_model` cells) | All five strategies wrap the same Random Forest + Isolation Forest and are scored on the validation block with the same metrics; the best by PR-AUC + F1 rank is carried forward |
| **Table 4.3** - Final held-out test-set results (best supervised model vs Isolation Forest vs best hybrid) | `notebooks/ibm_bank_trans_04_hybrid_models.ipynb` (`final_test_results` / `print_final_results` cell) and `python src\main.py` (`=== Final evaluation on held-out X_test ===` block) | One-time scoring of the selected models on the test block |
| **Table 4.4** - SHAP global feature importance (mean absolute SHAP value per feature, fraud class) | `notebooks/ibm_bank_trans_06_shap_explainability.ipynb` (`mean_abs_shap` cell) | `explain_model` on the Random Forest over the 5,000-row stratified test sample; features ranked by `np.abs(shap_values[:, :, 1]).mean(axis=0)` |
| Complementarity analysis (fraction of each model's missed fraud that the other model catches) | `notebooks/ibm_bank_trans_05_complementarity_analysis.ipynb` | Random Forest (threshold 0.40) and Isolation Forest (F1-optimal threshold) predictions on the test block are cross-tabulated against the true label; `rf_misses` / `iso_misses` overlap fractions are reported |
| SHAP beeswarm / summary / waterfall figures | `notebooks/ibm_bank_trans_06_shap_explainability.ipynb` (`shap.plots.beeswarm`, `shap.summary_plot`, `shap.plots.waterfall` cells) | Same SHAP values as Table 4.4 |
| Class-imbalance / EDA figures | `notebooks/ibm_bank_trans_01_eda.ipynb`, `notebooks/ibm_bank_accounts_01_eda.ipynb` | Direct plots on the raw files |

---

## Reproducibility Notes

- All supervised models and the Isolation Forest use `random_state=42`; SMOTE uses `random_state=42`.
- The train/validation/test split is a fixed chronological cut (no shuffling, no `train_test_split` randomness) - see the Temporal Split section for exact row counts.
- SMOTE runs **inside** the imbalanced-learn `Pipeline`, so it only ever resamples the training fold and never touches validation or test data.
- Isolation Forest `contamination` is pinned to the training-block fraud rate (`0.001331`).
- Hybrid fusion weight, all hybrid decision thresholds, and the SHAP classification threshold are tuned on the validation block only; the test block is scored once.
- The SHAP sample is a stratified 5,000-row draw from the test block with `random_state=42`.
- If using a different Python interpreter or platform, re-create the virtual environment and re-install `requirements.txt`.

---

## Troubleshooting & Tips

- If `.\.venv\Scripts\activate` fails, switch from PowerShell to Command Prompt.
- If you still get an error after creating the `.env` file with the correct paths, close and reopen the IDE so the new environment variables are picked up.
- If `python src\main.py --preprocess` fails, check that both raw CSVs exist in `data/raw/` and that the `.env` paths resolve.
- For memory-heavy operations (XGBoost training, SMOTE on the full training block), consider increasing available RAM or working on a smaller sample during experimentation.
- Notebooks expect to be run from the `notebooks/` directory and add the project root to `sys.path`; run them in numeric order the first time so the processed CSV exists.

---

## Contact

This repository accompanies an MSc final-year project at the University of Hertfordshire. For questions, contact the author: Azeez Adedayo Adebayo.

---
