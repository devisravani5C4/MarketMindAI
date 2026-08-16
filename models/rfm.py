import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def compute_rfm_segments(df: pd.DataFrame, id_col: str, date_col: str, amount_col: str) -> pd.DataFrame:
    # 1. Clean Dates and Calculate Raw RFM Metrics
    df[date_col] = pd.to_datetime(df[date_col])
    max_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(id_col).agg({
        date_col: lambda x: (max_date - x.max()).days,
        id_col: 'count',
        amount_col: 'sum'
    }).rename(columns={
        date_col: 'Recency',
        id_col: 'Frequency',
        amount_col: 'Monetary'
    }).reset_index()

    # Exclude zero or negative monetary records before scoring
    rfm = rfm[rfm['Monetary'] > 0].copy()
    if len(rfm) < 10:
        # Fallback if dataset is too small for LOF / Quantiles
        rfm['Segment'] = 'Standard Customer'
        return rfm

    # 2. Identify Extreme Multivariate Outliers (VIP / Whales / B2B) using LOF
    # Scale log-transformed features for LOF calculation
    features = np.log1p(rfm[['Recency', 'Frequency', 'Monetary']])
    
    # Identify top ~2% most isolated records as potential Whales/Anomalies
    contamination_rate = 0.02
    lof = LocalOutlierFactor(n_neighbors=min(20, len(rfm) - 1), contamination=contamination_rate)
    outlier_flags = lof.fit_predict(features)
    
    # Flag points as 'Whale/VIP' if LOF marks them as outlier AND they have high Monetary/Frequency
    high_monetary_cutoff = rfm['Monetary'].quantile(0.90)
    rfm['Is_Whale'] = (outlier_flags == -1) & (rfm['Monetary'] >= high_monetary_cutoff)

    # 3. Apply Non-Parametric IQR Capping (Winsorization) on non-whale rows for stable RFM Binning
    rfm['Monetary_Capped'] = rfm['Monetary']
    rfm['Frequency_Capped'] = rfm['Frequency']
    
    # Cap Monetary at 99th percentile
    m_cap = rfm['Monetary'].quantile(0.99)
    f_cap = rfm['Frequency'].quantile(0.99)
    
    rfm['Monetary_Capped'] = rfm['Monetary_Capped'].clip(upper=m_cap)
    rfm['Frequency_Capped'] = rfm['Frequency_Capped'].clip(upper=f_cap)

    # 4. Quartile Scoring using rank-based qcut to safely handle duplicate values
    rfm['R_Score'] = pd.qcut(rfm['Recency'].rank(method='first'), 4, labels=[4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency_Capped'].rank(method='first'), 4, labels=[1, 2, 3, 4])
    rfm['M_Score'] = pd.qcut(rfm['Monetary_Capped'].rank(method='first'), 4, labels=[1, 2, 3, 4])

    rfm['RFM_Score'] = (
        rfm['R_Score'].astype(str) + 
        rfm['F_Score'].astype(str) + 
        rfm['M_Score'].astype(str)
    )

    # 5. Business Segmentation Logic
    def segment_customer(row):
        # High-value multivariate outliers get promoted directly to VIP / Whale
        if row['Is_Whale']:
            return "VIP / Whale"
            
        r = int(row['R_Score'])
        f = int(row['F_Score'])
        m = int(row['M_Score'])
        score = r + f + m

        if score >= 11:
            return "Premium / Champion"
        elif f >= 3:
            return "Loyal Customer"
        elif r >= 3 and m <= 2:
            return "Discount Seeker"
        elif r == 2:
            return "At Risk"
        else:
            return "Lost Customer"

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)

    # Clean up temporary processing columns
    rfm.drop(columns=['Monetary_Capped', 'Frequency_Capped', 'Is_Whale'], inplace=True)

    return rfm