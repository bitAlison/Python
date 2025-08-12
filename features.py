# features.py
"""
Feature engineering for the Extreme Incremental Predictor project.

Main exported function:
- prepare_features_and_matrix(df, seq_len, supplemental_range, target_col)
-> X_flat, X_seq, y, dates_list, feature_cols, scaler_X, scaler_y
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config as cfg
from io_utils import log

# ==============================
# Expand date features
# ==============================
def expand_date_features(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    try:
        date_col = date_col or cfg.DATE_COL
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.weekday
        df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
        return df
    except Exception as e:
        log(f"An error occurred while trying to expand date features. {e}", "ERROR")
        return None

# ==============================
# Expand date features
# ==============================
def add_valor_freq(df: pd.DataFrame, target_col: str):
    freq = df[target_col].value_counts(normalize=True).to_dict()
    df[cfg.FREQUENCY_COL] = df[target_col].map(freq).fillna(0.0)
    return df

# ==============================
# Expand date features
# ==============================
def detect_min_max(df: pd.DataFrame):
    return int(df[cfg.TARGET_COL].min()), int(df[cfg.TARGET_COL].max())

# ==============================
# Expand date features
# ==============================
def get_suplementar_cols(df: pd.DataFrame, suplementares_range: Tuple[int, int]) -> List[str]:
    cols = []
    a,b = suplementares_range
    for i in range(a, b+1):
        name = f"{cfg.SUPPLEMENTARY_COL}{i}"
        if name in df.columns:
            cols.append(name)
    return cols

# ==============================
# Prepare features and matrix
# ==============================
def prepare_features_and_matrix(df: pd.DataFrame,
                                seq_len: int = 14,
                                suplementares_range: Tuple[int,int] = cfg.SUPPLEMENTARY_RANGE,
                                target_col: str = cfg.TARGET_COL
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list, list, StandardScaler, StandardScaler]:
    """
    Receives loaded dataframe config.DATA_SOURCE and returns:
    X_flat: (n_samples, n_feats) - features of t (used by LGB) 
    X_seq: (n_samples, seq_len, n_feats) - sequency for RNN 
    y: (n_samples,) 
    date_list:data list (each sample corresponds to a date of the next step) 
    feature_cols: order of columns used as features (without target)
    scaler_X: StandardScaler ajusted in X_flat 
    scaler_y: StandardScaler ajusted in y (floar colomuns)
    """
    try:
        df = df.copy().sort_values(cfg.DATE_COL).reset_index(drop=True)
        if len(df) <= seq_len:
            raise ValueError(f"Insufficient data({len(df)}) to seq_len={seq_len}")

        # expand temporal and freq resources
        df = expand_date_features(df, date_col=cfg.DATE_COL)
        df = add_valor_freq(df, target_col)
        suplementares = get_suplementar_cols(df, suplementares_range)

        # features base
        base_feats = ['year','month','day','dayofweek','is_weekend',cfg.FREQUENCY_COL]
        feature_cols = base_feats + suplementares

        # guarantee existence
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0

        # build windows: for each i, sample is window [i:i+seq_len] -> predict at i+seq_len
        n = len(df)
        samples = n - seq_len
        n_feats = len(feature_cols)

        X_seq = np.zeros((samples, seq_len, n_feats), dtype=float)
        X_flat = np.zeros((samples, n_feats), dtype=float)
        y = np.zeros((samples,), dtype=float)
        dates_list = []

        for i in range(samples):
            window = df.iloc[i:i+seq_len]
            step = df.iloc[i+seq_len]
            X_seq[i,:,:] = window[feature_cols].values
            X_flat[i,:] = step[feature_cols].values
            y[i] = step[target_col]
            dates_list.append(step[cfg.DATE_COL])

        # scale X_flat and y using StandardScaler (RNN will use scaler_X after reshaping)
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_flat_scaled = scaler_X.fit_transform(X_flat)
        y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).flatten()

        # also scale X_seq: flatten then transform then reshape
        X_seq_2d = X_seq.reshape(samples * seq_len, n_feats)
        X_seq_2d_scaled = scaler_X.transform(X_seq_2d)
        X_seq_scaled = X_seq_2d_scaled.reshape(samples, seq_len, n_feats)

        log(f"Features prepared: {samples} samples, {seq_len} steps, {n_feats} features", "INFO")
        return X_flat_scaled, X_seq_scaled, y_scaled, dates_list, feature_cols, scaler_X, scaler_y
    except Exception as e:
        log(f". {e}", "ERROR")
