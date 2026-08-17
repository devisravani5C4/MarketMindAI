import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from mlxtend.frequent_patterns import apriori, association_rules

def detect_recommendation_columns(df: pd.DataFrame) -> dict:
    """
    Smart fuzzy detection for recommendation features using regex and data profiling.
    """
    columns = df.columns.tolist()
    mapping = {
        "invoice_col": None,
        "user_col": None,
        "item_col": None,
        "qty_col": None
    }

    invoice_patterns = [r'invoice', r'order', r'transaction', r'trans_id', r'receipt']
    user_patterns = [r'cust', r'user', r'client', r'member', r'account']
    item_patterns = [r'prod', r'item', r'desc', r'title', r'stock', r'sku', r'article']
    qty_patterns = [r'qty', r'quantity', r'rating', r'amount', r'unit', r'score']

    for col in columns:
        col_lower = str(col).lower()
        
        # Detect Invoice ID
        if not mapping["invoice_col"] and any(re.search(pat, col_lower) for pat in invoice_patterns):
            mapping["invoice_col"] = col

        # Detect User ID
        if not mapping["user_col"] and any(re.search(pat, col_lower) for pat in user_patterns):
            mapping["user_col"] = col

        # Detect Item Column
        if not mapping["item_col"] and any(re.search(pat, col_lower) for pat in item_patterns):
            mapping["item_col"] = col

        # Detect Quantity/Rating Column
        if not mapping["qty_col"] and any(re.search(pat, col_lower) for pat in qty_patterns):
            mapping["qty_col"] = col

    return mapping


def check_recommendation_suitability(df: pd.DataFrame) -> dict:
    """
    Evaluates dataset suitability for recommendation models using fuzzy mapping.
    """
    mapping = detect_recommendation_columns(df)
    
    # Dataset is suitable if we can identify at least (Item ID + (Invoice ID or User ID))
    has_item = mapping["item_col"] is not None
    has_user_or_invoice = (mapping["user_col"] is not None) or (mapping["invoice_col"] is not None)
    
    is_suitable = bool(has_item and has_user_or_invoice)

    return {
        "is_suitable": is_suitable,
        "detected_mapping": mapping
    }


def generate_recommendations(df: pd.DataFrame, user_col: str = None, item_col: str = None, 
                            invoice_col: str = None, qty_col: str = None, top_n: int = 5) -> dict:
    """
    Auto-selects the best algorithm (Apriori Market Basket Analysis vs. Collaborative Filtering vs Popularity)
    based on available columns and dataset structure.
    """
    if not item_col or item_col not in df.columns:
        raise ValueError("Valid Product/Item column is required.")

    # Clean data copy
    data = df.copy()
    data[item_col] = data[item_col].astype(str).str.strip()
    
    # 1. Fallback: Global Popularity Calculation
    if qty_col and qty_col in data.columns and pd.api.types.is_numeric_dtype(data[qty_col]):
        popular_series = data.groupby(item_col)[qty_col].sum().sort_values(ascending=False).head(top_n)
    else:
        popular_series = data[item_col].value_counts().head(top_n)
        
    popular_items = [{"item": str(k), "score": int(v)} for k, v in popular_series.items()]

    # -------------------------------------------------------------
    # PATH A: Market Basket Analysis (Apriori / Association Rules)
    # Selected if Invoice Column is present and has good transaction density
    # -------------------------------------------------------------
    if invoice_col and invoice_col in data.columns:
        try:
            basket = (
                data.groupby([invoice_col, item_col])[qty_col if (qty_col and qty_col in data.columns) else item_col]
                .count().unstack().fillna(0)
            )
            # Binary encode (purchased / not purchased)
            basket_sets = basket.applymap(lambda x: 1 if x > 0 else 0)

            # Limit to top items if matrix is too large for performance
            if basket_sets.shape[1] > 500:
                top_cols = basket_sets.sum(axis=0).sort_values(ascending=False).head(500).index
                basket_sets = basket_sets[top_cols]

            frequent_itemsets = apriori(basket_sets, min_support=0.01, use_colnames=True)
            
            if not frequent_itemsets.empty:
                rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
                rules = rules.sort_values(by="lift", ascending=False).head(top_n * 2)

                rules_output = []
                for _, row in rules.iterrows():
                    antecedents = list(row["antecedents"])
                    consequents = list(row["consequents"])
                    rules_output.append({
                        "if_bought": ", ".join(antecedents),
                        "recommended": ", ".join(consequents),
                        "confidence": round(float(row["confidence"]) * 100, 1),
                        "lift": round(float(row["lift"]), 2)
                    })

                return {
                    "algorithm_used": "Market Basket Analysis (Apriori)",
                    "algorithm_code": "apriori",
                    "popular_items": popular_items,
                    "association_rules": rules_output,
                    "user_recommendations": {}
                }
        except Exception as e:
            print(f"Apriori execution skipped, falling back to Collaborative Filtering: {e}")

    # -------------------------------------------------------------
    # PATH B: User-Item Collaborative Filtering (Cosine Similarity)
    # Selected if Customer/User Column is present
    # -------------------------------------------------------------
    if user_col and user_col in data.columns:
        recommendations_by_user = {}
        try:
            if qty_col and qty_col in data.columns and pd.api.types.is_numeric_dtype(data[qty_col]):
                user_item_matrix = data.pivot_table(
                    index=user_col, columns=item_col, values=qty_col, aggfunc='sum', fill_value=0
                )
            else:
                user_item_matrix = data.groupby([user_col, item_col]).size().unstack(fill_value=0)

            # Cosine Similarity between users
            user_sim = cosine_similarity(user_item_matrix)
            user_sim_df = pd.DataFrame(user_sim, index=user_item_matrix.index, columns=user_item_matrix.index)

            # Sample top 10 active customers for preview
            sample_users = user_item_matrix.index[:10]

            for user in sample_users:
                sim_users = user_sim_df[user].sort_values(ascending=False).iloc[1:6].index
                
                user_bought = set(user_item_matrix.columns[user_item_matrix.loc[user] > 0])
                sim_users_bought = set(
                    user_item_matrix.columns[user_item_matrix.loc[sim_users].sum(axis=0) > 0]
                )
                
                recs = list(sim_users_bought - user_bought)[:top_n]
                if not recs:
                    recs = [item["item"] for item in popular_items[:top_n]]

                recommendations_by_user[str(user)] = recs

            return {
                "algorithm_used": "User-Based Collaborative Filtering (Cosine Similarity)",
                "algorithm_code": "collaborative",
                "popular_items": popular_items,
                "association_rules": [],
                "user_recommendations": recommendations_by_user
            }
        except Exception as e:
            print(f"Collaborative filtering warning: {e}")

    # -------------------------------------------------------------
    # PATH C: Fallback to Global Popularity Engine
    # -------------------------------------------------------------
    return {
        "algorithm_used": "Popularity-Based Engine (Top Sellers)",
        "algorithm_code": "popularity",
        "popular_items": popular_items,
        "association_rules": [],
        "user_recommendations": {}
    }