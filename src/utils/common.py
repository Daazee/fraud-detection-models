import pandas as pd
def wrangle_data(file_path):
    # Placeholder for data wrangling logic
    # This function would contain the actual implementation for cleaning and transforming the data
    transactions_dataframe = pd.read_csv(file_path)
    
    columns_to_drop = []
    #drop leaky columns
    leaky_columns = ["ALERT_ID"]
    columns_to_drop.append(leaky_columns)


    #transactions_dataframe.drop(columns= columns_to_drop, inplace=True)
    return transactions_dataframe