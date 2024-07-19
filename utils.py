# utils.py

import pandas as pd

def preprocess_years_in_job(X):
    X = X.copy()
    X['Years in current job'] = X['Years in current job'].str.extract(r'(\d+)').astype(float)
    return X
