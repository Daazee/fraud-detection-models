from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ])


def train(X_train, y_train) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline
