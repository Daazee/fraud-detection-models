import numpy as np
from sklearn.metrics import average_precision_score, f1_score


class WeightedAverageHybrid:
    """
    Hybrid model that fuses a supervised classifier's predict_proba with an Isolation Forest's
    decision_function via weighted averaging of normalised scores.

    final_score = weight * clf_norm + (1 - weight) * iso_norm

    Both scores are min-max scaled to [0, 1] using ranges fit on validation
    data, and the Isolation Forest score is sign-flipped so that, like the
    classifier's fraud probability, higher = more suspicious.
    """

    def __init__(self, classifier, isolation_forest):
        self.classifier = classifier
        self.isolation_forest = isolation_forest
        self.weight_ = None
        self._clf_min = self._clf_max = None
        self._iso_min = self._iso_max = None

    def _raw_scores(self, X):
        clf_score = self.classifier.predict_proba(X)[:, 1]
        # IsolationForest.decision_function: higher = more normal, so flip sign
        iso_score = -self.isolation_forest.decision_function(X)
        return clf_score, iso_score

    def _normalise(self, clf_score, iso_score, fit_ranges=False):
        if fit_ranges:
            self._clf_min = clf_score.min()
            self._clf_max = clf_score.max()
            self._iso_min = iso_score.min()
            self._iso_max = iso_score.max()

        clf_norm = (clf_score - self._clf_min) / (self._clf_max - self._clf_min + 1e-12)
        iso_norm = (iso_score - self._iso_min) / (self._iso_max - self._iso_min + 1e-12)
        return clf_norm, iso_norm

    def fit_weight(self, X_val, y_val, w_grid=None, thresholds=None):
        """
        Dynamically select w by sweeping candidate values and keeping
        whichever maximises PR-AUC on X_val. Also fits the min-max
        normalisation ranges used by predict_proba/decision_function below.
        Assumes self.classifier and self.isolation_forest are already fitted
        on a training set that does NOT overlap with X_val.

        Given the best weight, select the decision threshold that maximises F1 score.
        """
        if w_grid is None:
            w_grid = np.arange(0.0, 1.01, 0.05)

        if thresholds is None:
            thresholds = np.arange(0.0, 1.01, 0.05)

        clf_score, iso_score = self._raw_scores(X_val)

        #normalise scores to [0, 1] via min-max scaling using ranges fit on validation data
        clf_norm, iso_norm = self._normalise(clf_score, iso_score, fit_ranges=True)

        best_weight, best_pr_auc = None, -np.inf
        history = []
        for w in w_grid:
            fused = w * clf_norm + (1 - w) * iso_norm
            pr_auc = average_precision_score(y_val, fused)
            history.append((round(float(w), 2), pr_auc))
            if pr_auc > best_pr_auc:
                best_pr_auc, best_weight = pr_auc, w

        self.weight_ = best_weight
        self.val_pr_auc_ = best_pr_auc
        self.w_search_history_ = history

        # Now that the best weight has been selected, we can also select the decision threshold that maximises F1 score.
        fused_at_best_weight = best_weight * clf_norm + (1 - best_weight) * iso_norm
        best_threshold, best_f1 = None, -np.inf
        for t in thresholds:
            y_pred = (fused_at_best_weight >= t).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, t

        self.best_threshold_ = best_threshold
        self.best_f1_ = best_f1

        return self

    def predict_proba(self, X):
        if self.weight_ is None or self.best_threshold_ is None:
            raise RuntimeError("Call fit_weight(X_val, y_val) before predict_proba.")
        clf_score, iso_score = self._raw_scores(X)
        clf_norm, iso_norm = self._normalise(clf_score, iso_score, fit_ranges=False)
        fused = self.weight_ * clf_norm + (1 - self.weight_) * iso_norm
        return np.column_stack([1 - fused, fused])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.best_threshold_).astype(int)


def train(classifier, isolation_forest, X_val, y_val, w_grid=None) -> WeightedAverageHybrid:
    # Both classifier and isolation_forest must already be fitted on same training set
    hybrid = WeightedAverageHybrid(classifier, isolation_forest)
    hybrid.fit_weight(X_val, y_val, w_grid=w_grid)
    print(f"Selected weight = {hybrid.weight_:.2f}  (validation PR-AUC = {hybrid.val_pr_auc_:.4f})")
    return hybrid
