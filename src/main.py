import argparse
import pandas as pd
from pathlib import Path

from utils.preprocessing import wrangle_data

PROCESSED = Path(__file__).parents[1] / "data" / "processed" / "transactions_processed.csv"


def parse_args():
    parser = argparse.ArgumentParser(description="Fraud Detection Pipeline")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run preprocessing and overwrite the processed data file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.preprocess or not PROCESSED.exists():
        print("Running preprocessing...")
        wrangle_data(save=True)

    df = pd.read_csv(PROCESSED)
    X = df.drop(columns=["IS_FRAUD"])
    y = df["IS_FRAUD"]

    print(f"Loaded: {X.shape[0]:,} rows, {X.shape[1]} features")
    # train_model(X, y)


if __name__ == "__main__":
    main()