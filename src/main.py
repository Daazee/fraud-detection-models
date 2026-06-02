import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from utils.preprocessing import wrangle_data
from models.logistic_regression import train as train_lr
from models.random_forest import train as train_rf
from models.xgboost_model import train as train_xgb
from models.isolation_forest import train as train_if
from models.evaluate import evaluate, evaluate_anomaly, identify_best_model

PROCESSED = Path(__file__).parents[1] / "data" / "processed" / "transactions_processed.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing and overwrite the processed data file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.preprocess or not PROCESSED.exists():
        print("Running preprocessing...")
        wrangle_data(save=True)

    df = pd.read_csv(PROCESSED)
    X = df.drop(columns=["IS_FRAUD"])
    y = df["IS_FRAUD"]
    print(f"Loaded: {X.shape[0]:,} rows, {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")
    print(f"Fraud rate — train: {y_train.mean():.4%}  test: {y_test.mean():.4%}\n")

    print("Training Logistic Regression...")
    lr = train_lr(X_train, y_train)

    print("Training Random Forest...")
    rf = train_rf(X_train, y_train)

    print("Training XGBoost...")
    xgb = train_xgb(X_train, y_train)

    print("Training Isolation Forest...")
    iso = train_if(X_train)

  #Evaluation
    results = []
    results.append(evaluate(lr,  X_test, y_test, "Logistic Regression"))
    results.append(evaluate(rf,  X_test, y_test, "Random Forest"))
    results.append(evaluate(xgb, X_test, y_test, "XGBoost"))
    results.append(evaluate_anomaly(iso, X_test, y_test, "Isolation Forest"))

    identify_best_model(results)


if __name__ == "__main__":
    main()
