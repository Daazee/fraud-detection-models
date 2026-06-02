import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler

from ingestion.loaders import load_file

PROJECT_ROOT = Path(__file__).parents[2]

TARGET = "IS_FRAUD"
ID_COLS = ["TX_ID"]
LEAKY_COLS = ["ALERT_ID"]
CORR_THRESHOLD = 0.8
SKEW_THRESHOLD = 1.0


def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ID_COLS + LEAKY_COLS if c in df.columns]
    return df.drop(columns=cols)


def drop_correlated_features(
    df: pd.DataFrame,
    target: str = TARGET,
    threshold: float = CORR_THRESHOLD,
) -> pd.DataFrame:
    num_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    corr_df = df[num_cols].copy()
    corr_df[target] = df[target].astype(int)

    corr_matrix = corr_df.corr()
    target_corr = corr_matrix[target].abs().drop(target)

    to_drop = set()
    for i in range(len(num_cols)):
        for j in range(i + 1, len(num_cols)):
            a, b = num_cols[i], num_cols[j]
            if abs(corr_matrix.loc[a, b]) >= threshold:
                drop = a if target_corr[a] < target_corr[b] else b
                to_drop.add(drop)

    return df.drop(columns=list(to_drop))


def _outlier_fraud_correlated(col: pd.Series, fraud: pd.Series) -> bool:
    Q1, Q3 = col.quantile(0.25), col.quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0:
        return False
    is_outlier = (col < Q1 - 1.5 * IQR) | (col > Q3 + 1.5 * IQR)
    if is_outlier.sum() == 0:
        return False
    ct = pd.crosstab(
        is_outlier.map({True: "Outlier", False: "Normal"}),
        fraud.map({True: "Fraud", False: "Legit"}),
    )
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return False
    _, p, _, _ = chi2_contingency(ct)
    return (p < 0.05) and (fraud[is_outlier].mean() > fraud.mean())


def treat_outliers(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    df = df.copy()
    features = [c for c in df.select_dtypes(include="number").columns if c != target]
    for feat in features:
        if not _outlier_fraud_correlated(df[feat], df[target]):
            lo, hi = df[feat].quantile(0.01), df[feat].quantile(0.99)
            df[feat] = df[feat].clip(lower=lo, upper=hi)
    return df


def apply_log_transform(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    df = df.copy()
    features = [c for c in df.select_dtypes(include="number").columns if c != target]
    for feat in features:
        if abs(df[feat].skew()) > SKEW_THRESHOLD:
            df[feat] = np.log1p(df[feat])
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if not cat_cols:
        return df
    return pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)


def encode_target(df: pd.DataFrame, target: str = TARGET) -> pd.DataFrame:
    df = df.copy()
    df[target] = df[target].astype(int)
    return df

def wrangle_data(save: bool = True) -> pd.DataFrame:
    df = load_file()
    df = drop_columns(df)
    df = drop_correlated_features(df)
    df = treat_outliers(df)
    df = apply_log_transform(df)
    df = encode_categoricals(df)
    df = encode_target(df)

    if save:
        out = PROJECT_ROOT / "data" / "processed" / "transactions_processed.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)

    return df
