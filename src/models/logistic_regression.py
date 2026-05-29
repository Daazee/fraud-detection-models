from utils.common import wrangle_data
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()



def train_model():
    file_path = os.getenv("file_path")
    if not file_path:
        raise ValueError("file_path is not set in the environment variables.")
    transaction_dataframe = wrangle_data(file_path)
    print(transaction_dataframe.head())
    # Placeholder for model training logic
    # This function would contain the actual implementation for training the logistic regression model
    pass
#determine the base line accuracy
#baseline_accuracy = transaction_dataframe[""].value_counts(normalize=True).max()

# design the model
