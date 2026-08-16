import re
import math
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure VADER lexicon is cached and available
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


def check_review_suitability(df: pd.DataFrame) -> dict:
    """
    Checks if the dataset contains text columns suitable for sentiment analysis.
    Beyond "does a candidate column exist", this also validates that the
    detected column actually holds substantive free text (not mostly empty,
    not just short repeated labels/codes) so the UI can give a precise reason
    when a dataset is rejected instead of a generic one.
    """
    missing_requirements = []

    if df is None or df.empty:
        return {
            "is_suitable": False,
            "missing_requirements": ["Dataset is empty"],
            "mapping": {"review_col": "", "product_col": "", "rating_col": ""}
        }

    mapping = auto_detect_review_columns(df)
    review_col = mapping.get("review_col")

    if not review_col:
        text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        if not text_cols:
            missing_requirements.append(
                "No text columns found in dataset (sentiment analysis needs a column of written feedback)"
            )
        else:
            missing_requirements.append(
                "No column with substantive free-form text was found -- only short labels, "
                "codes, or mostly-empty text columns are present"
            )
        return {
            "is_suitable": False,
            "missing_requirements": missing_requirements,
            "mapping": mapping
        }

    return {
        "is_suitable": True,
        "missing_requirements": [],
        "mapping": mapping
    }


def auto_detect_review_columns(df: pd.DataFrame) -> dict:
    """
    Scored fuzzy detection for the Review Text, Product, and Rating columns.

    Rather than taking the first column whose name loosely matches a keyword
    (which can misfire -- e.g. "review_id" matching before "review_content",
    or a product "description" column being mistaken for customer feedback),
    every candidate column is scored on both its NAME and its CONTENT, and the
    best-scoring candidate wins. Content signals (average word count, ratio of
    unique values, non-null coverage) are what separate real free-text review
    fields from short/repetitive/ID-like columns that merely share a keyword.
    """
    empty_mapping = {"review_col": "", "product_col": "", "rating_col": ""}
    if df is None or df.empty:
        return empty_mapping

    # Strong keywords are unambiguous signals of customer feedback text.
    # Weak keywords are generic and can just as easily belong to a product
    # description, a log message, etc. -- they only count as a tiebreaker/
    # fallback, never enough on their own to beat a column with no name match
    # but much better content.
    STRONG_REVIEW_KEYWORDS = ['review', 'feedback', 'comment', 'opinion', 'testimonial']
    WEAK_REVIEW_KEYWORDS = ['text', 'description', 'message', 'remark', 'summary', 'content', 'body', 'notes']

    # Column-name fragments that mean "this is an identifier/metadata field,
    # not free text", even if it also happens to contain a review keyword
    # (e.g. "review_id", "review_date", "review_link").
    ID_LIKE_KEYWORDS = ['id', 'sku', 'code', 'date', 'time', 'link', 'url', 'image', 'img', 'asin', 'no', 'num', 'key']

    RATING_KEYWORDS = ['rating', 'stars', 'star', 'score']

    MIN_AVG_WORDS = 3          # below this, it's a label/tag, not review prose
    MIN_NON_NULL_RATIO = 0.05  # column that's almost entirely empty is unusable

    def tokens(col):
        # Split into whole words on snake_case/kebab-case/space/camelCase
        # boundaries so keywords are matched as WHOLE WORDS, not raw
        # substrings -- e.g. "Unnamed: 0" must NOT match keyword "name" just
        # because the letters "n-a-m-e" happen to appear inside "unnamed".
        s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', str(col))
        return [p.lower() for p in re.split(r'[^a-zA-Z0-9]+', s) if p]

    def has_kw(col_tokens, keywords):
        return any(kw in col_tokens for kw in keywords)

    def is_id_like(col_tokens):
        return has_kw(col_tokens, ID_LIKE_KEYWORDS)

    def content_stats(col):
        s = df[col].dropna().astype(str)
        non_null_ratio = len(s) / len(df) if len(df) else 0
        if len(s) == 0:
            return {"avg_words": 0.0, "avg_len": 0.0, "uniqueness": 0.0, "non_null_ratio": 0.0}
        avg_words = s.str.split().str.len().mean()
        avg_len = s.str.len().mean()
        uniqueness = s.nunique() / len(s)
        return {
            "avg_words": 0.0 if pd.isna(avg_words) else avg_words,
            "avg_len": 0.0 if pd.isna(avg_len) else avg_len,
            "uniqueness": uniqueness,
            "non_null_ratio": non_null_ratio,
        }

    text_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    # --- Review column ---
    scored_candidates = []
    for col in text_cols:
        col_tokens = tokens(col)
        if is_id_like(col_tokens):
            continue

        stats = content_stats(col)
        if stats["avg_words"] < MIN_AVG_WORDS or stats["non_null_ratio"] < MIN_NON_NULL_RATIO:
            continue

        name_score = 0
        if has_kw(col_tokens, STRONG_REVIEW_KEYWORDS):
            name_score = 100
        elif has_kw(col_tokens, WEAK_REVIEW_KEYWORDS):
            name_score = 20

        # Content score rewards longer, more varied free text -- repeated
        # short categorical strings ("Positive"/"Negative"/"N/A") score low
        # on both word count and uniqueness even if verbose columns exist.
        content_score = stats["avg_words"] + (stats["uniqueness"] * 10)

        scored_candidates.append((col, name_score + content_score))

    review_col = None
    if scored_candidates:
        review_col = max(scored_candidates, key=lambda x: x[1])[0]

    # --- Product column ---
    # Split into strong vs weak keywords like the review column, and explicitly
    # exclude columns that clearly identify a *person* rather than a product
    # (e.g. "user_name", "reviewer_name", "customer_id") -- otherwise a bare
    # "name" match on a person column can outrank the real product field.
    STRONG_PRODUCT_KEYWORDS = ['product', 'item', 'sku', 'asin']
    WEAK_PRODUCT_KEYWORDS = ['title', 'name']
    PERSON_KEYWORDS = ['user', 'customer', 'reviewer', 'author', 'client', 'buyer', 'reviewedby']

    product_candidates = []
    for col in df.columns:
        if col == review_col:
            continue
        col_tokens = tokens(col)
        if has_kw(col_tokens, PERSON_KEYWORDS):
            continue

        if has_kw(col_tokens, STRONG_PRODUCT_KEYWORDS):
            name_score = 100
        elif has_kw(col_tokens, WEAK_PRODUCT_KEYWORDS):
            name_score = 20
        else:
            continue

        avg_len = content_stats(col)["avg_len"] if col in text_cols else 0
        # Among equally-named candidates, prefer the shorter field -- product
        # identifiers/names are short labels, not paragraphs (which rules out
        # e.g. a verbose "about_product" description column beating "product_id").
        product_candidates.append((col, name_score, -avg_len))

    product_col = None
    if product_candidates:
        product_col = max(product_candidates, key=lambda x: (x[1], x[2]))[0]

    # --- Rating column (numeric, small bounded range like 1-5 or 1-10) ---
    rating_col = None
    rating_candidates = []
    for col in df.columns:
        if col == review_col or col == product_col:
            continue
        col_tokens = tokens(col)
        if has_kw(col_tokens, RATING_KEYWORDS):
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            valid_ratio = numeric_col.notna().mean()
            if valid_ratio > 0.5:
                rating_candidates.append(col)

    if rating_candidates:
        rating_col = rating_candidates[0]

    return {
        "review_col": review_col or "",
        "product_col": product_col or "",
        "rating_col": rating_col or ""
    }


def analyze_review_sentiments(df: pd.DataFrame, review_col: str, product_col: str = None) -> dict:
    """
    Computes global metrics, product-level breakdowns, and structured comment listings.
    """
    if review_col not in df.columns:
        raise ValueError(f"Review column '{review_col}' not found in dataset.")

    sia = SentimentIntensityAnalyzer()

    def classify_text(text):
        if pd.isna(text) or not str(text).strip():
            return "Neutral", 0.0

        scores = sia.polarity_scores(str(text))
        compound = scores['compound']

        if compound >= 0.05:
            return "Positive", compound
        elif compound <= -0.05:
            return "Negative", compound
        else:
            return "Neutral", compound

    df_reviews = df.copy()

    # Clean missing values in product column to prevent dropping in groupby
    if product_col and product_col in df_reviews.columns:
        df_reviews[product_col] = df_reviews[product_col].fillna("Unknown Product").astype(str)

    # Apply sentiment analysis
    results = df_reviews[review_col].apply(classify_text)
    df_reviews['sentiment'] = [r[0] for r in results]
    df_reviews['compound_score'] = [r[1] for r in results]

    # Clean text representation for comment modal displaying
    df_reviews['clean_review'] = df_reviews[review_col].fillna('').astype(str).str.strip()

    # 1. Global Sentiment Metrics
    total_reviews = len(df_reviews)
    sent_counts = df_reviews['sentiment'].value_counts()

    pos_count = int(sent_counts.get("Positive", 0))
    neu_count = int(sent_counts.get("Neutral", 0))
    neg_count = int(sent_counts.get("Negative", 0))

    global_distribution = {
        "total": total_reviews,
        "positive": {
            "count": pos_count,
            "pct": round((pos_count / total_reviews * 100), 1) if total_reviews > 0 else 0
        },
        "neutral": {
            "count": neu_count,
            "pct": round((neu_count / total_reviews * 100), 1) if total_reviews > 0 else 0
        },
        "negative": {
            "count": neg_count,
            "pct": round((neg_count / total_reviews * 100), 1) if total_reviews > 0 else 0
        }
    }

    # 2. Product-Level Sentiment Breakdown & Modal Comments Map
    product_sentiment_table = []
    product_comments_map = {}

    if product_col and product_col in df_reviews.columns:
        grouped = df_reviews.groupby([product_col, 'sentiment']).size().unstack(fill_value=0)

        # Ensure all columns exist
        for col_name in ['Positive', 'Neutral', 'Negative']:
            if col_name not in grouped.columns:
                grouped[col_name] = 0

        grouped['Total'] = grouped['Positive'] + grouped['Neutral'] + grouped['Negative']

        # Calculate percentages safely
        grouped['Positive_Pct'] = np.where(grouped['Total'] > 0, ((grouped['Positive'] / grouped['Total']) * 100).round(1), 0)
        grouped['Neutral_Pct'] = np.where(grouped['Total'] > 0, ((grouped['Neutral'] / grouped['Total']) * 100).round(1), 0)
        grouped['Negative_Pct'] = np.where(grouped['Total'] > 0, ((grouped['Negative'] / grouped['Total']) * 100).round(1), 0)

        # Sort top 15 products by review volume
        top_products = grouped.sort_values(by='Total', ascending=False).head(15).reset_index()

        for idx, row in top_products.iterrows():
            prod_name = str(row[product_col])
            prod_id = f"prod_{idx}"

            product_sentiment_table.append({
                "prod_id": prod_id,
                "product_name": prod_name,
                "total": int(row['Total']),
                "positive_str": f"{int(row['Positive'])} / {int(row['Total'])} ({row['Positive_Pct']}%)",
                "neutral_str": f"{int(row['Neutral'])} / {int(row['Total'])} ({row['Neutral_Pct']}%)",
                "negative_str": f"{int(row['Negative'])} / {int(row['Total'])} ({row['Negative_Pct']}%)",
                "pos_count": int(row['Positive']),
                "neu_count": int(row['Neutral']),
                "neg_count": int(row['Negative']),
                "pos_pct": float(row['Positive_Pct']),
                "neu_pct": float(row['Neutral_Pct']),
                "neg_pct": float(row['Negative_Pct'])
            })

            prod_df = df_reviews[df_reviews[product_col] == row[product_col]]

            # Filter empty strings for clean modal rendering
            pos_comments = [c for c in prod_df[prod_df['sentiment'] == 'Positive']['clean_review'].head(30).tolist() if c]
            neu_comments = [c for c in prod_df[prod_df['sentiment'] == 'Neutral']['clean_review'].head(30).tolist() if c]
            neg_comments = [c for c in prod_df[prod_df['sentiment'] == 'Negative']['clean_review'].head(30).tolist() if c]

            product_comments_map[prod_id] = {
                "product_name": prod_name,
                "positive": pos_comments,
                "neutral": neu_comments,
                "negative": neg_comments
            }

    return {
        "global": global_distribution,
        "product_breakdown": product_sentiment_table,
        "comments_map": product_comments_map
    }