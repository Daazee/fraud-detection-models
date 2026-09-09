import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

ROOT = os.path.abspath("..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from sklearn.metrics import average_precision_score, f1_score

class CascadeHybrid():
    """
    Cascade / two-stage fusion: Random Forest makes the decision by
    default; Isolation Forest is only consulted for samples whose
    RF-normalised score falls inside an uncertain band. Inherits score
    extraction and normalisation from WeightedAverageHybrid.
    """

    def __init__(self, classifier, isolation_forest, band_candidates=None):
        self.classifier = classifier
        self.isolation_forest = isolation_forest
        self.band_candidates = band_candidates or [
            (0.40, 0.60),
            (0.35, 0.65),
            (0.45, 0.55),
            (0.30, 0.70),
        ]
        
        self._clf_min = self._clf_max = None
        self._iso_min = self._iso_max = None
        self.best_lower_ = None
        self.best_upper_ = None
        self.best_pr_auc_ = None
        self.cascade_history_ = None
        self.best_threshold_ = None
        self.best_f1_ = None

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
    
    def fit_band(self, X_val, y_val, thresholds=None):
        """
        Select the uncertain-score band that maximises PR-AUC on the
        validation set. Sets best_lower_, best_upper_, best_pr_auc_,
        and cascade_history_ as fitted attributes.

        Given the best band, also select the decision threshold on the
        fused score that maximises F1 on the validation set, mirroring
        WeightedAverageHybrid. Sets best_threshold_ and best_f1_.
        """
        if thresholds is None:
            thresholds = np.arange(0.0, 1.01, 0.05)

        clf_score, iso_score = self._raw_scores(X_val)
        clf_norm, iso_norm = self._normalise(clf_score, iso_score, fit_ranges=True)
        # normalise scores to [0, 1] via min-max scaling using ranges fit on validation data

        best_lower, best_upper, best_pr_auc = None, None, -np.inf
        history = []

        for lower, upper in self.band_candidates:
            fused = clf_norm.copy()
            uncertain_mask = (clf_norm >= lower) & (clf_norm <= upper)
            fused[uncertain_mask] = iso_norm[uncertain_mask]

            pr_auc = average_precision_score(y_val, fused)
            history.append({
                "band": (round(float(lower), 2), round(float(upper), 2)),
                "pr_auc": pr_auc,
                "n_escalated": int(uncertain_mask.sum()),
            })

            if pr_auc > best_pr_auc:
                best_pr_auc, best_lower, best_upper = pr_auc, lower, upper

        self.best_lower_ = best_lower
        self.best_upper_ = best_upper
        self.best_pr_auc_ = best_pr_auc
        self.cascade_history_ = history

        # Now that the best band is selected, tune the decision threshold on
        # the fused score (F1) instead of the previously hard-coded 0.6.
        fused_at_best_band = clf_norm.copy()
        uncertain_mask = (clf_norm >= best_lower) & (clf_norm <= best_upper)
        fused_at_best_band[uncertain_mask] = iso_norm[uncertain_mask]

        best_threshold, best_f1 = None, -np.inf
        for t in thresholds:
            y_pred = (fused_at_best_band >= t).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, t

        self.best_threshold_ = round(float(best_threshold), 4)
        self.best_f1_ = best_f1

        return self

    def transform_uncertain_predictions(self, X):
        """
        Apply the fitted cascade band to new data (e.g. test set).
        Reuses the min-max ranges fit during fit() — does not refit
        normalisation ranges on X, to avoid test-set leakage.
        """
        if self.best_lower_ is None:
            raise ValueError("CascadeHybrid instance is not fitted yet. Call 'fit' first.")

        clf_score, iso_score = self._raw_scores(X)
        clf_norm, iso_norm = self._normalise(clf_score, iso_score, fit_ranges=False)

        fused = clf_norm.copy()
        uncertain_mask = (clf_norm >= self.best_lower_) & (clf_norm <= self.best_upper_)
        fused[uncertain_mask] = iso_norm[uncertain_mask]

        return fused

    def predict_proba(self, X):
        """
        sklearn-compatible wrapper around transform(), so this class
        can be evaluated with the same evaluate() function used for
        the other models. Returns shape (n_samples, 2): [P(legit), P(fraud)].
        """
        fraud_score = self.transform_uncertain_predictions(X)
        return np.column_stack([1 - fraud_score, fraud_score])

    def predict(self, X):
        # Use the threshold tuned on the validation set in fit_band (F1),
        # consistent with the other hybrids. >= matches the comparison
        # used during that threshold sweep.
        if self.best_threshold_ is None:
            raise ValueError("CascadeHybrid instance is not fitted yet. Call 'fit_band' first.")
        fraud_score = self.transform_uncertain_predictions(X)
        return (fraud_score >= self.best_threshold_).astype(int)

def train(classifier, isolation_forest, X_val, y_val) -> CascadeHybrid:
        cascade_hybrid = CascadeHybrid(classifier, isolation_forest)
        cascade_hybrid.fit_band(X_val, y_val)
        print(f"Cascade band = ({cascade_hybrid.best_lower_}, {cascade_hybrid.best_upper_}), threshold = {cascade_hybrid.best_threshold_}, val PR-AUC = {cascade_hybrid.best_pr_auc_:.4f}\n") 
        return cascade_hybrid
