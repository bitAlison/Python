# features.py
"""
Feature engineering for the Extreme Incremental Predictor project.
"""

from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config as cfg
from io_utils import log


def expand_date_features(df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
    try:
        date_col = date_col or cfg.DATE_COL
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.weekday
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        return df
    except Exception as e:
        log(f"Erro ao expandir features de data: {e}", "ERROR")
        return None


def add_valor_freq(df: pd.DataFrame, target_col: str):
    freq = df[target_col].value_counts(normalize=True).to_dict()
    df[cfg.FREQUENCY_COL] = df[target_col].map(freq).fillna(0.0)
    return df


def detect_min_max(df: pd.DataFrame):
    return int(df[cfg.TARGET_COL].min()), int(df[cfg.TARGET_COL].max())


def get_suplementar_cols(df: pd.DataFrame, suplementares_range: Tuple[int, int]) -> List[str]:
    cols = []
    a, b = suplementares_range
    for i in range(a, b + 1):
        name = f"{cfg.SUPPLEMENTARY_COL}{i}"
        if name in df.columns:
            cols.append(name)
    return cols


def ensure_features_exist(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Garante que todas as colunas de feature existam no DF, criando se faltar."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        log(f"⚠ Criando colunas ausentes para previsão: {missing}", "WARN")
        for c in missing:
            df[c] = 0
    return df


def prepare_features_and_matrix(df: pd.DataFrame,
                                seq_len: int = 14,
                                suplementares_range: Tuple[int, int] = cfg.SUPPLEMENTARY_RANGE,
                                target_col: str = cfg.TARGET_COL):
    try:
        df = df.copy().sort_values(cfg.DATE_COL).reset_index(drop=True)
        if len(df) <= seq_len:
            raise ValueError(f"Insufficient data({len(df)}) to seq_len={seq_len}")

        # expandir features
        df = expand_date_features(df, date_col=cfg.DATE_COL)
        df = add_valor_freq(df, target_col)
        suplementares = get_suplementar_cols(df, suplementares_range)

        # colunas de feature
        base_feats = ['year', 'month', 'day', 'dayofweek', 'is_weekend', cfg.FREQUENCY_COL]
        feature_cols = base_feats + suplementares
        df = ensure_features_exist(df, feature_cols)

        n = len(df)
        samples = n - seq_len
        n_feats = len(feature_cols)

        X_seq = np.zeros((samples, seq_len, n_feats), dtype=float)
        X_flat = np.zeros((samples, n_feats), dtype=float)
        y = np.zeros((samples,), dtype=float)
        dates_list = []

        for i in range(samples):
            window = df.iloc[i:i + seq_len]
            step = df.iloc[i + seq_len]
            X_seq[i, :, :] = window[feature_cols].values
            X_flat[i, :] = step[feature_cols].values
            y[i] = step[target_col]
            dates_list.append(step[cfg.DATE_COL])

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_flat_scaled = scaler_X.fit_transform(X_flat)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        X_seq_2d = X_seq.reshape(samples * seq_len, n_feats)
        X_seq_2d_scaled = scaler_X.transform(X_seq_2d)
        X_seq_scaled = X_seq_2d_scaled.reshape(samples, seq_len, n_feats)

        log(f"Features preparadas: {samples} amostras, {seq_len} passos, {n_feats} features", "INFO")
        return X_flat_scaled, X_seq_scaled, y_scaled, dates_list, feature_cols, scaler_X, scaler_y

    except Exception as e:
        log(f"Erro em prepare_features_and_matrix: {e}", "ERROR")
        return None
