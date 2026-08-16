import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


def get_dataset_info(df):
    """
    Generates df.info() like structural dictionary for frontend missing values table.
    """
    info = []
    total_rows = len(df)

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        non_null_count = total_rows - null_count
        dtype_str = str(df[col].dtype)
        unique_count = int(df[col].nunique(dropna=True))

        is_categorical = (
            dtype_str in ["object", "category", "bool"] or 
            (unique_count < 20 and total_rows > 0)
        )

        info.append(
            {
                "column": col,
                "non_null": non_null_count,
                "null_count": null_count,
                "dtype": dtype_str,
                "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
                "is_categorical": is_categorical
            }
        )

    return info


def modify_column_type(df, col, new_type):
    """
    Converts data type of a column. Throws ValueError if conversion fails.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' does not exist.")

    try:
        if new_type == "datetime64[ns]":
            df[col] = pd.to_datetime(df[col])
        else:
            df[col] = df[col].astype(new_type)
        return df
    except Exception as e:
        raise ValueError(
            f"Cannot convert column '{col}' to {new_type}. Details: {str(e)}"
        )


def handle_missing_values(df, col, strategy):
    """
    Handles missing values based on selected column and action strategy.
    """
    if col not in df.columns:
        return df

    if strategy == "drop_col":
        return df.drop(columns=[col])
    elif strategy == "drop":
        df = df.dropna(subset=[col])
    elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mean())
    elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    elif strategy == "mode":
        mode_val = df[col].mode()
        if not mode_val.empty:
            df[col] = df[col].fillna(mode_val[0])
    elif strategy == "ffill":
        df[col] = df[col].ffill()
    elif strategy == "bfill":
        df[col] = df[col].bfill()

    return df


def inspect_uniqueness(df, col):
    """
    Returns unique values and frequency counts for a column.
    """
    if col not in df.columns:
        return {"error": "Column not found"}

    val_counts = df[col].value_counts(dropna=False).head(50).to_dict()
    unique_list = [
        {"value": str(k), "count": int(v)} for k, v in val_counts.items()
    ]

    return {
        "unique_count": int(df[col].nunique(dropna=False)),
        "total_rows": len(df),
        "values": unique_list,
    }


def find_and_replace(df, col, find_val, replace_val):
    """
    Replaces exact string/value occurrences inside a specified column.
    """
    if col not in df.columns:
        return df

    df[col] = df[col].astype(str).replace(find_val, replace_val)
    return df


def encode_column(df, col, method):
    """
    Applies Label Encoding or One-Hot Encoding.
    """
    if col not in df.columns:
        return df

    if method == "label":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    elif method == "onehot":
        df = pd.get_dummies(df, columns=[col], drop_first=True)

    return df


def handle_outliers_iqr(df, col):
    """
    Trims numeric outliers outside 1.5 * IQR boundaries.
    """
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return df

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]


def scale_feature(df, col, method):
    """
    Scales a numeric column using MinMax or StandardScaler.
    """
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return df

    vals = df[[col]].values
    if method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    df[col] = scaler.fit_transform(vals)
    return df


def clean_text_column(df, col):
    """
    Normalizes string column (lowercase, trim, punctuation removal).
    """
    if col not in df.columns:
        return df

    df[col] = (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
    )
    return df


def preprocess_dataset(df, options=None):
    """
    Backward-compatibility wrapper for whole-dataset preprocessing.
    """
    if options is None:
        options = {}
    
    missing_strategy = options.get("missing_strategy", "drop")
    for col in df.columns:
        df = handle_missing_values(df, col, missing_strategy)

    return df

# marketMindAI/models/preprocessing.py

def clean_data(df):
    # Standardize column names to lowercase with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('.', '_')
    
    # Map common review column aliases to standard 'review_text'
    review_aliases = ['reviews_text', 'review', 'comments', 'comment', 'feedback', 'text']
    for alias in review_aliases:
        if alias in df.columns and 'review_text' not in df.columns:
            df.rename(columns={alias: 'review_text'}, inplace=True)
            
    # Map rating aliases if needed
    rating_aliases = ['reviews_rating', 'rating', 'stars', 'score']
    for alias in rating_aliases:
        if alias in df.columns and 'rating' not in df.columns:
            df.rename(columns={alias: 'rating'}, inplace=True)

    return df