from sklearn.model_selection import train_test_split
import shap
import pandas as pd

shap.initjs()


def sample_for_shap(X_test, y_test, sample_size=5000):
    if sample_size >= len(X_test):
        # nothing to subsample - use the full test set as-is
        return X_test, y_test

    _, X_shap, _, y_shap = train_test_split(X_test, y_test,
        test_size=sample_size,
        stratify=y_test,
        random_state=42,
    )
    return X_shap, y_shap

def build_shap_explainer(model, X_train, y_train, n_background=300):
    """
    Build a TreeExplainer configured with a background sample stratified from
    X_train/y_train, guaranteeing minority-class (fraud) representation.

    Uses feature_perturbation="interventional" rather than the default
    tree_path_dependent. This corrects a specific distortion: because SMOTE
    oversamples the minority class internally during pipeline.fit(), the default
    perturbation mode derives its baseline (E[f(X)]) from the trees' internal,
    SMOTE-rebalanced leaf statistics, anchoring explanations to an artificial
    ~50/50 prior rather than the true ~0.13% fraud rate. A background this size
    drawn purely proportionally to the true fraud rate would be expected to
    contain zero fraud observations, so the fraud/legit split is stratified
    explicitly rather than left to chance.
    
    Parameters
    ----------
    model : fitted tree-based estimator
    X_train, y_train : used to build the stratified background sample
    n_background : background sample size (default 300)

    Returns
    -------
    shap.TreeExplainer, configured and ready to call
    """
    fraud_rows = X_train[y_train == 1]
    legit_rows = X_train[y_train == 0]

    n_fraud = max(1, round(n_background * y_train.mean()))
    n_legit = n_background - n_fraud

    shap_background_sample = pd.concat([
        fraud_rows.sample(n=n_fraud, random_state=42),
        legit_rows.sample(n=n_legit, random_state=42),
    ]).sample(frac=1, random_state=42)

    masker = shap.maskers.Independent(shap_background_sample, max_samples=len(shap_background_sample))
    return shap.TreeExplainer(model, data=masker, feature_perturbation="interventional")


def explain_model(explainer, X_shap, check_additivity=False):
    """
    Compute SHAP values for X_shap using a pre-built TreeExplainer (see
    build_shap_explainer).

    check_additivity defaults to False: interventional perturbation is an
    approximation (unlike tree_path_dependent's exact reconstruction), and can
    legitimately fail SHAP's additivity check on extreme-probability predictions
    (e.g. f(x) very close to 0 or 1). This is a known limitation, not a data error.

    Returns
    -------
    shap_values : shap.Explanation, indexable per-record for waterfall/force
                  plots, usable as-is for beeswarm/bar/summary plots
    """
    return explainer(X_shap, check_additivity=check_additivity)