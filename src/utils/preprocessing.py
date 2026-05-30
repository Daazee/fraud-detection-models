from ingestion.loaders import load_file
import pandas as pd

def wrangle_data(file_path):
    transactions_dataframe = load_file()
    print(transactions_dataframe.head())

    
    columns_to_drop = []
    #drop leaky columns
    leaky_columns = ["ALERT_ID"]
    columns_to_drop.append(leaky_columns)


    #transactions_dataframe.drop(columns= columns_to_drop, inplace=True)
    return transactions_dataframe