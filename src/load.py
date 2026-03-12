import pandas as pd
import numpy as np


def load_and_inspect(filepath: str, encoding: str | None = None):
    """Load a CSV and print basic inspection output.

    If no encoding is provided, this tries UTF-8 first and falls back to cp1252.
    """
    try:
        df = pd.read_csv(filepath, encoding=encoding)
    except UnicodeDecodeError:
        # Falls back for CSVs written with a Windows/Latin1 encoding.
        df = pd.read_csv(filepath, encoding="cp1252")

    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Target distribution:\n{df['SalePrice'].describe()}")
    return df


def clean_data(df):
    # Drop columns with >40% missing
    thresh = len(df) * 0.6
    df = df.dropna(axis=1, thresh=thresh)

    # Fill numeric missing with median
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical missing with mode
    cat_cols = df.select_dtypes(include="object").columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # Log-transform skewed target
    df["SalePrice"] = np.log1p(df["SalePrice"])
    return df


if __name__ == "__main__":
    import traceback

    try:
        load_and_inspect("src/ames.csv")
    except Exception:
        traceback.print_exc()
        raise
