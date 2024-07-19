# utils.py

import pandas as pd

def preprocess_years_in_job(X):
    X = X.copy()
    # Ensure 'Years in current job' is of type string
    X['Years in current job'] = X['Years in current job'].astype(str)
    # Handle possible NaN values and empty strings
    X['Years in current job'] = X['Years in current job'].fillna('')
    # Extract numeric values from the strings
    X['Years in current job'] = X['Years in current job'].str.extract(r'(\d+)').astype(float)
    return X