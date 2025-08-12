# data_prep.py
"""
Module for preparing data for the Extreme Incremental Predictor.
Includes:
- Reading the .csv data source
- Filtering outliers
- Optional normalization
- Creating time windows (sequences)
- Separation into training and validation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import config as cfg
from io_utils import load_data_source, log

# ==============================
# Outliers filters
# ==============================
def outliers_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers from the target field using IQR.
    """
    try:
        Q1 = df[cfg.TARGET_COL].quantile(0.25)
        Q3 = df[cfg.TARGET_COL].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - cfg.IQR_MULTIPLIER * IQR
        lim_sup = Q3 + cfg.IQR_MULTIPLIER * IQR
        return df[(df[cfg.TARGET_COL] >= lim_inf) & (df[cfg.TARGET_COL] <= lim_sup)]
    except Exception as e:
        log(f"An error occurred while trying to filter outliers.{e}", "ERROR")

# ==============================
# Creates properly windows
# ==============================
def create_windows(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates time windows of size `seq_len`.
    Returns X (sequences) and y (future target values).
    """
    try:
        dados = df[cfg.FEATURES_COLS].values
        target = df[cfg.TARGET_COL].values
        X, y = [], []
        for i in range(len(dados) - seq_len):
            X.append(dados[i:i + seq_len])
            y.append(target[i + seq_len])
        return np.array(X), np.array(y)
    except Exception as e:
        log(f"An error occurred while trying to create the windows. {e}", 'ERROR')

# ==============================
# Prepare data for prediction
# ==============================
def prepare_data(seq_len: int = cfg.SEQ_LEN, normalizar: bool = True) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Reads, filters, normalizes and creates windows for training.
    Retorns:
     - X: array 3D (samples, seq_len, features)
     - y: array 1D (target values)
     - scaler: MinMaxScaler (if used)
    """
    try:
        df = load_data_source()
        df = outliers_filters(df)

        scaler = None
        if normalizar:
            scaler = MinMaxScaler()
            df[cfg.FEATURES_COLS] = scaler.fit_transform(df[cfg.FEATURES_COLS])

        X, y = create_windows(df, seq_len)
        return X, y, scaler
    except Exception as e:
        log(f"An error occurred while trying to prepare data. {e}", 'ERROR')

# ==============================
# Prepare data for training
# ==============================
def prepare_data_for_forecasts(df_hist: pd.DataFrame, seq_len: int, scaler: MinMaxScaler = None) -> np.ndarray:
    """
    Prepares last historical sequence for prediction.
    """
    try:
        if scaler is not None:
            dados = scaler.transform(df_hist[cfg.FEATURES_COLS])
        else:
            dados = df_hist[cfg.FEATURES_COLS].values

        ultima_seq = dados[-seq_len:]
        return np.expand_dims(ultima_seq, axis=0)
    except Exception as e:
        log(f"An error occurred while trying to prepare data for forecasts. {e}", 'ERROR')
