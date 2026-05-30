from ingestion.loaders import load_file
import pandas as pd



def explore_data():
    transaction_dataframe = load_file()
    print(transaction_dataframe.head())
    # Placeholder for data exploration logic
    # This function would contain the actual implementation for exploring the data
    pass