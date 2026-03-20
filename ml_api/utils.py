import pandas as pd

def preprocess_years_in_job(X):
    """Extract numeric value from 'Years in current job' strings like '4 years', '10+ years'."""
    X = X.copy()
    X['Years in current job'] = (
        X['Years in current job']
        .astype(str)
        .str.extract(r'(\d+)')
        .astype(float)
    )
    return X