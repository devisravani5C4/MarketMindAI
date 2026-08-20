import pandas as pd
import re

def detect_segmentation_columns(df: pd.DataFrame):
    """
    Detects candidate columns for Customer ID, Date, and Amount using keyword matching
    and data-type fallback inspection.
    """
    cols = df.columns.tolist()
    cols_lower = {col.lower(): col for col in cols}

    # 1. Customer ID Candidates
    id_keywords = ["customer", "cust", "user", "client", "member", "account", "phone", "email"]
    detected_id = None
    
    # Direct keyword search
    for kw in id_keywords:
        matched = [c for c in cols if kw in c.lower()]
        if matched:
            # Prefer ID/NAME over address or contact names
            id_matches = [m for m in matched if any(x in m.lower() for x in ["id", "code", "number", "num", "name"])]
            detected_id = id_matches[0] if id_matches else matched[0]
            break

    # 2. Transaction Date Candidates
    date_keywords = ["date", "time", "timestamp", "dt", "day"]
    detected_date = None
    
    for kw in date_keywords:
        matched = [c for c in cols if kw in c.lower()]
        if matched:
            detected_date = matched[0]
            break

    # Fallback for date: Check column data types for datetime strings
    if not detected_date:
        for col in cols:
            if df[col].dtype == 'datetime64[ns]':
                detected_date = col
                break
            # Check sample strings for date format
            sample = df[col].dropna().astype(str).head(10)
            if sample.str.contains(r'\d{2,4}[-/.]\d{1,2}[-/.]\d{1,2}').any():
                detected_date = col
                break

    # 3. Order Amount Candidates
    amount_keywords = ["sales", "amount", "total", "price", "revenue", "spend", "cost", "value"]
    detected_amount = None
    
    for kw in amount_keywords:
        matched = [c for c in cols if kw in c.lower()]
        if matched:
            detected_amount = matched[0]
            break

    # Summary Check
    is_suitable = bool(detected_id and detected_date and detected_amount)

    return {
        "is_suitable": is_suitable,
        "mapping": {
            "id_col": detected_id,
            "date_col": detected_date,
            "amount_col": detected_amount
        },
        "missing_requirements": [
            req for req, val in [
                ("Customer Identification Column (e.g. CUSTOMERNAME)", detected_id),
                ("Transaction Date Column (e.g. ORDERDATE)", detected_date),
                ("Transaction Amount Column (e.g. SALES)", detected_amount)
            ] if not val
        ]
    }