import argparse
import pandas as pd
from pathlib import Path
from sklearn.dummy import DummyClassifier
from utils.preprocessing import wrangle_data
from models.logistic_regression import train as train_logistic_regression
from models.random_forest import train as train_random_forest
from models.xgboost_model import train as train_xgboost
from models.isolation_forest import train as train_isolation_forest
from models.evaluate import evaluate, evaluate_anomaly, find_anomaly_optimal_threshold, find_best_threshold, identify_best_model
from models.weighted_hybrid_model import train as train_weighted_average_hybrid_model
from models.cascade_hybrid_model import train as train_cascade_hybrid_model 

PROCESSED = Path(__file__).parents[1] / "data" / "processed" / "merged_transactions_accounts_processed.csv"


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

   
    cutoff = int(len(X) * 0.8)
    X_train_full, y_train_full = X.iloc[: cutoff], y.iloc[:cutoff]
    X_test, y_test =  X.iloc[cutoff: ], y.iloc[cutoff:]

    cutoff_train_val = int(len(X_train_full) * 0.8)
    X_train, y_train = X_train_full.iloc[:cutoff_train_val], y_train_full.iloc[:cutoff_train_val]
    X_validation, y_validation = X_train_full.iloc[cutoff_train_val:], y_train_full.iloc[cutoff_train_val:]


    print(f"Train: {X_train.shape[0]:,}  |  Validation: {X_validation.shape[0]:,}  | Test: {X_test.shape[0]:,}")
    print(f"Fraud rate — train: {y_train.mean():.4%} validation: {y_validation.mean():.4%} test: {y_test.mean():.4%}\n")

    print("Training Logistic Regression...")
    logistic_regression_pipeline = train_logistic_regression(X_train, y_train)

    print("Training Random Forest...")
    random_forest_pipeline = train_random_forest(X_train, y_train)

    print("Training XGBoost...")
    xgboost_pipeline = train_xgboost(X_train, y_train)

    print("Training Isolation Forest...")
    isolation_forest_model = train_isolation_forest(X_train)

    print("Training Dummy Baseline...")
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)

    # predicting probabilities and scores for validation set
    logistic_regression_val_probs = logistic_regression_pipeline.predict_proba(X_validation)[:, 1]
    random_forest_val_probs = random_forest_pipeline.predict_proba(X_validation)[:, 1]
    xgboost_val_probs = xgboost_pipeline.predict_proba(X_validation)[:, 1]
    isolation_forest_val_scores = -isolation_forest_model.decision_function(X_validation)

    # Find best thresholds for classifiers and Isolation Forest
    lr_threshold_result = find_best_threshold(y_validation, logistic_regression_val_probs, beta=1.0)
    rf_threshold_result = find_best_threshold(y_validation, random_forest_val_probs, beta=1.0)
    xgb_threshold_result = find_best_threshold(y_validation, xgboost_val_probs, beta=1.0)
    optimal_iso_threshold = find_anomaly_optimal_threshold(y_validation, isolation_forest_val_scores)

    print("\n── Tuned decision thresholds (from validation) ──────────")
    print(f"  Logistic Regression : {lr_threshold_result['threshold']:.4f}")
    print(f"  Random Forest       : {rf_threshold_result['threshold']:.4f}")
    print(f"  XGBoost             : {xgb_threshold_result['threshold']:.4f}")
    print(f"  Isolation Forest    : {optimal_iso_threshold:.4f}")


    # Validation set Evaluation
    print("\n=== Evaluation on held-out X_validation ===")
    validation_results = []
    validation_results.append(evaluate(dummy, X_validation, y_validation, "Dummy (Baseline)"))
    validation_results.append(evaluate(logistic_regression_pipeline, X_validation, y_validation, "Logistic Regression", lr_threshold_result["threshold"]))
    validation_results.append(evaluate(random_forest_pipeline, X_validation, y_validation, "Random Forest", rf_threshold_result["threshold"]))
    validation_results.append(evaluate(xgboost_pipeline, X_validation, y_validation, "XGBoost", xgb_threshold_result["threshold"]))
    validation_results.append(evaluate_anomaly(isolation_forest_model, X_validation, y_validation, "Isolation Forest", optimal_iso_threshold))

    overall_best_classifier_model_name = identify_best_model(validation_results, sort_by_performance=True)

    # Identify best classifier for Hybrid model 
    best_classifier_model = (
                logistic_regression_pipeline if overall_best_classifier_model_name == "Logistic Regression" 
                else random_forest_pipeline if overall_best_classifier_model_name == "Random Forest" else 
                xgboost_pipeline if overall_best_classifier_model_name == "XGBoost" else None
    )

    # get best threshold for the best classifier
    best_threshold = (
        lr_threshold_result["threshold"] if overall_best_classifier_model_name == "Logistic Regression" 
        else rf_threshold_result["threshold"] if overall_best_classifier_model_name == "Random Forest" 
        else xgb_threshold_result["threshold"] if overall_best_classifier_model_name == "XGBoost" 
        else None
    )
 
    print("Fitting Hybrid Models...\n")
    print("Fitting Weighted Average Hybrid Model...")
    weighted_avg_model = train_weighted_average_hybrid_model(best_classifier_model, isolation_forest_model, X_validation, y_validation)

    print("Fitting Cascade Hybrid Model Band...")
    cascade_model = train_cascade_hybrid_model(best_classifier_model, isolation_forest_model, X_validation, y_validation)
    
    # --- Final, one-time evaluation on X_test ---
    print("\n=== Final evaluation on held-out X_test ===")
    final_results = []
    final_results.append(evaluate(best_classifier_model, X_test, y_test, overall_best_classifier_model_name, best_threshold))
    final_results.append(evaluate(logistic_regression_pipeline, X_test, y_test, "Logistic Regression", lr_threshold_result["threshold"]))
    final_results.append(evaluate(xgboost_pipeline, X_test, y_test, "XGBoost", xgb_threshold_result["threshold"]))
    final_results.append(evaluate_anomaly(isolation_forest_model, X_test, y_test, "Isolation Forest", optimal_iso_threshold))
    final_results.append(evaluate(weighted_avg_model, X_test, y_test, "Hybrid (Weighted Average)"))
    final_results.append(evaluate(cascade_model, X_test, y_test, "Hybrid (Cascade Hybrid)"))

    identify_best_model(final_results, sort_by_performance=True)
if __name__ == "__main__":
    main()
