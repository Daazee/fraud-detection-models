from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)),
    ])


def train(X_train, y_train) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline
