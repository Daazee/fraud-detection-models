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
        "threshold": float(thresholds[best_idx]),
        "f_score":   float(f_scores[best_idx]),
        "precision": float(precision[best_idx]),
        "recall":    float(recall[best_idx]),
        "beta":      beta,
    }


def _compute_metrics(y_test, y_pred, y_score, model_name: str) -> dict:
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

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
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted Legit  Predicted Fraud")
    print(f"  Actual Legit     {cm[0,0]:>8}         {cm[0,1]:>8}")
    print(f"  Actual Fraud     {cm[1,0]:>8}         {cm[1,1]:>8}")


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


def evaluate_anomaly(model, X_test, y_test, model_name) -> dict:
    # -1 = anomaly (fraud=1), 1 = normal (legit=0)
    y_pred = np.where(model.predict(X_test) == -1, 1, 0)
    # Negate: higher score = more anomalous = higher fraud probability
    y_score = -model.decision_function(X_test)

    metrics = _compute_metrics(y_test, y_pred, y_score, model_name)
    #_print_results(metrics, y_test, y_pred)
    return metrics


def identify_best_model(results: list[dict]) -> None:
    df = pd.DataFrame(results).set_index("model")

    best_pr  = df["pr_auc"].idxmax()
    best_f1  = df["f1_score"].idxmax()

    print("\n\n── Model Comparison ─────────────────────────────────────")
    print(df[["precision", "recall", "f1_score", "fpr", "fnr", "roc_auc", "pr_auc"]].to_string())

    print("\n── Best Model ───────────────────────────────────────────")
    print(f"  By PR-AUC   : {best_pr:<25} ({df.loc[best_pr,  'pr_auc']:.4f})")
    print(f"  By F1-Score : {best_f1:<25} ({df.loc[best_f1,  'f1_score']:.4f})")

    # Overall winner: ranks by both metrics combined
    df["rank"] = df["pr_auc"].rank(ascending=False) + df["f1_score"].rank(ascending=False)
    overall_best = df["rank"].idxmin()
    print(f"\n  Overall best (PR-AUC + F1): {overall_best}")
    return overall_best

def print_final_results(results: list[dict]) -> None:
    df = pd.DataFrame(results).set_index("model")

    final_best_pr  = df["pr_auc"].idxmax()
    final_best_f1  = df["f1_score"].idxmax()

    print("\n\n── Model Comparison ─────────────────────────────────────")
    print(df[["precision", "recall", "f1_score", "fpr", "fnr", "roc_auc", "pr_auc"]].to_string())

    print("\n── Best Model ───────────────────────────────────────────")
    print(f"  By PR-AUC   : {final_best_pr:<25} ({df.loc[final_best_pr,  'pr_auc']:.4f})")
    print(f"  By F1-Score : {final_best_f1:<25} ({df.loc[final_best_f1,  'f1_score']:.4f})")

    # Overall winner: ranks by both metrics combined
    df["rank"] = df["pr_auc"].rank(ascending=False) + df["f1_score"].rank(ascending=False)
    final_best = df["rank"].idxmin()
    print(f"\n  Final best (PR-AUC + F1): {final_best}")

