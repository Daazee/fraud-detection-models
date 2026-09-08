from sklearn.ensemble import IsolationForest

CONTAMINATION = 0.001331  # observed fraud rate of 0.1331% from the training data (i.e., 0.001331)


def build_model() -> IsolationForest:
    return IsolationForest(
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )


def train(X_train) -> IsolationForest:
    # Unsupervised model. There is no y_train; learns the distribution of normal transactions
    model = build_model()
    model.fit(X_train)
    return model
