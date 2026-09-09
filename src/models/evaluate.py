import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
)


def find_best_threshold(y_true, y_score, beta: float = 1.0) -> dict:
    """
    Sweep classification thresholds and return the one that maximizes F-beta.

    beta=1.0 -> F1 (precision/recall weighted equally).
    beta>1.0 -> weights recall higher, appropriate for fraud where missing a
                fraud case (FN) is usually costlier than a false alarm (FP).

    Fit this on a VALIDATION set (never the test set) - the returned
    threshold should then be applied as a fixed constant when scoring
    held-out test data, e.g.:

        y_pred = (y_score_test >= result["threshold"]).astype(int)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    precision, recall = precision[:-1], recall[:-1]  # drop the threshold=inf point

    beta_sq = beta ** 2
    f_scores = (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall + 1e-12)

    best_idx = np.nanargmax(f_scores)
    return {
        "threshold": round(float(thresholds[best_idx]), 4),
        "f_score":   round(float(f_scores[best_idx]), 4),
        "precision": round(float(precision[best_idx]), 4),
        "recall":    round(float(recall[best_idx]), 4),
        "beta":      beta,
    }

def find_anomaly_optimal_threshold(y_val, y_score, beta: float = 1.0):
    # note that y_score should already be inverted (higher = more anomalous = more likely fraud)
    # Precision/recall across thresholds on the ALREADY-inverted scale
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_score)

    # F-beta across those thresholds (note: len(thresholds) == len(precisions) - 1).
    # beta=1 -> F1; beta>1 -> weights recall higher, appropriate for fraud where a
    # missed fraud (FN) is costlier than a false alarm (FP).
    beta_sq = beta ** 2
    f_scores = (1 + beta_sq) * (precisions * recalls) / (beta_sq * precisions + recalls + 1e-12)

    # Guard against the last index, which has no corresponding threshold
    f_scores_for_argmax = f_scores[:-1]
    best_idx = np.argmax(f_scores_for_argmax)

    optimal_iso_threshold = thresholds[best_idx]
    return round(float(optimal_iso_threshold), 4)


def _compute_metrics(y_test, y_pred, y_score, model_name: str) -> dict:
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "model":     model_name,
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "fpr":       round(fpr, 4),
        "fnr":       round(fnr, 4),
        "roc_auc":   round(roc_auc_score(y_test, y_score), 4),
        "pr_auc":    round(average_precision_score(y_test, y_score), 4),
    }


def _print_results(metrics: dict, y_test, y_pred) -> None:
    print(f"\n{'=' * 58}")
    print(f"  {metrics['model']}")
    print(f"{'=' * 58}")
    print(f"  Precision          : {metrics['precision']:.4f}")
    print(f"  Recall             : {metrics['recall']:.4f}")
    print(f"  F1-Score           : {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC             : {metrics['pr_auc']:.4f}  <- primary metric")
    print(f"  False Positive Rate: {metrics['fpr']:.4f}  (legit flagged as fraud)")
    print(f"  False Negative Rate: {metrics['fnr']:.4f}  (fraud missed)")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted Legit  Predicted Fraud")
    print(f"  Actual Legit     {conf_matrix[0,0]:>8}         {conf_matrix[0,1]:>8}")
    print(f"  Actual Fraud     {conf_matrix[1,0]:>8}         {conf_matrix[1,1]:>8}")


def evaluate(model, X_test, y_test, model_name: str = "Model", threshold: float | None = None) -> dict:
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    if threshold is not None:
        # Use a threshold tuned on a validation set (e.g. via find_best_threshold)
        # instead of sklearn's default 0.5 cutoff.
        if y_score is None:
            raise ValueError("model has neither predict_proba nor decision_function; cannot apply a custom threshold")
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        if y_score is None:
            y_score = y_pred.astype(float)

    metrics = _compute_metrics(y_test, y_pred, y_score, model_name)
    #_print_results(metrics, y_test, y_pred)
    return metrics


def evaluate_anomaly(model, X_test, y_test, model_name, optimal_score: float | None = None) -> dict:
    # Negate: higher score = more anomalous = higher fraud probability
    y_score = -model.decision_function(X_test)

    if optimal_score is not None:
        # Threshold tuned on the validation set (via find_anomaly_optimal_threshold),
        # applied on the same negated scale precision_recall_curve used: y_score >= t
        y_pred = (y_score >= optimal_score).astype(int)
    else:
        # Default boundary from the model's contamination setting.
        # -1 = anomaly (fraud=1), 1 = normal (legit=0)
        y_pred = np.where(model.predict(X_test) == -1, 1, 0)

    metrics = _compute_metrics(y_test, y_pred, y_score, model_name)
    #_print_results(metrics, y_test, y_pred)
    return metrics


def identify_best_model(results: list[dict], sort_by_performance: bool = False) -> str:
    df = pd.DataFrame(results).set_index("model")

    best_pr = df["pr_auc"].idxmax()
    best_f1 = df["f1_score"].idxmax()

    df["pr_auc_and_f1_score"] = (df["pr_auc"] + df["f1_score"]) / 2

    if sort_by_performance:
        df = df.sort_values(by=["pr_auc_and_f1_score"], ascending=False)

    print("\n\n── Model Comparison ------------")
    print(df[["precision", "recall", "f1_score", "fpr", "fnr", "roc_auc", "pr_auc"]].to_string())

    print("\n── Best Model ───────────────────────────────────────────")
    print(f"  By PR-AUC   : {best_pr:<25} ({df.loc[best_pr,  'pr_auc']:.4f})")
    print(f"  By F1-Score : {best_f1:<25} ({df.loc[best_f1,  'f1_score']:.4f})")

    combined_best = df["pr_auc_and_f1_score"].idxmax()
    print(f"\nOverall best (PR-AUC + F1 average): {combined_best}  ({df.loc[combined_best, 'pr_auc_and_f1_score']:.5f})")

    return combined_best
