import shap
shap.initjs()


def explain_model(model, X_train, X_test, feature_names):
    explainer = shap.Explainer(model, X_train, feature_names=feature_names) #should shap object be created with X_train dataset
    shap_values = explainer(X_test)
    return shap_values