from sklearn.ensemble import IsolationForest

CONTAMINATION = 0.0013  # observed fraud rate of 0.13%


def build_model() -> IsolationForest:
    return IsolationForest(
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )


def train(X_train) -> IsolationForest:
    # Unsupervised — no y_train; learns the distribution of normal transactions
    model = build_model()
    model.fit(X_train)
    return model
