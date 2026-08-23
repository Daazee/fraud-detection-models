from sklearn.model_selection import train_test_split
import shap

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

def explain_model(model, X_shap, feature_names=None, check_additivity=True):
    """
    Compute SHAP values for X_shap using a TreeExplainer on `model`.

    `model` is the fitted tree-based estimator itself (e.g. RandomForestClassifier),
    already extracted from the pipeline by the caller, so SHAP doesn't have to deal
    with non-estimator steps like SMOTE, which only apply during training.

    Uses feature_perturbation="tree_path_dependent" with no background dataset -
    avoids a known additivity-check failure on extreme-probability predictions,
    and keeps explanations anchored to the test period only (see module docstring).

    Parameters
    ----------
    model : fitted tree-based estimator
    X_shap : sample to explain (see sample_for_shap)
    feature_names : optional list/Index of feature names
    check_additivity : bool, default True. Set False only if a failure is
        confirmed to be a negligible floating-point artifact - note it if you do.

    Returns
    -------
    shap_values : shap.Explanation, indexable per-record for waterfall/force
                  plots, usable as-is for beeswarm/bar/summary plots
    """
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent",
        feature_names=feature_names,
    )
    return explainer(X_shap, check_additivity=check_additivity)