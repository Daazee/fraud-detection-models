import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)


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
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
    print(f"  ROC-AUC            : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC             : {metrics['pr_auc']:.4f}  ← primary metric")
    print(f"  False Positive Rate: {metrics['fpr']:.4f}  (legit flagged as fraud)")
    print(f"  False Negative Rate: {metrics['fnr']:.4f}  (fraud missed)")
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"               Predicted Legit  Predicted Fraud")
    print(f"  Actual Legit     {cm[0,0]:>8}         {cm[0,1]:>8}")
    print(f"  Actual Fraud     {cm[1,0]:>8}         {cm[1,1]:>8}")


def evaluate(model, X_test, y_test, model_name: str = "Model") -> dict:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred.astype(float)

    metrics = _compute_metrics(y_test, y_pred, y_score, model_name)
    _print_results(metrics, y_test, y_pred)
    return metrics


def evaluate_anomaly(model, X_test, y_test, model_name: str = "Isolation Forest") -> dict:
    # -1 = anomaly (fraud=1), 1 = normal (legit=0)
    y_pred = np.where(model.predict(X_test) == -1, 1, 0)
    # Negate: higher score = more anomalous = higher fraud probability
    y_score = -model.decision_function(X_test)

    metrics = _compute_metrics(y_test, y_pred, y_score, model_name)
    _print_results(metrics, y_test, y_pred)
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
