# features.py
"""
Feature engineering for the project.

Exporta:
- expand_date_features
- add_valor_freq
- get_suplementar_cols
- ensure_features_exist
- prepare_features_and_matrix
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import config as cfg
from io_utils import log

# ----------------------------------------
# Datas
# ----------------------------------------
def expand_date_features(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    date_col = date_col or cfg.DATE_COL
    if df is None or df.empty:
        raise ValueError("DataFrame vazio em expand_date_features.")
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isna().any():
        # remove linhas com data inválida
        before = len(df)
        df = df.dropna(subset=[date_col]).reset_index(drop=True)
        log(f"[features] Removidas {before-len(df)} linhas com data inválida.", "WARNING")
    df['year'] = df[date_col].dt.year.astype(int)
    df['month'] = df[date_col].dt.month.astype(int)
    df['day'] = df[date_col].dt.day.astype(int)
    df['dayofweek'] = df[date_col].dt.weekday.astype(int)
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    return df

# ----------------------------------------
# Frequência do alvo
# ----------------------------------------
def add_valor_freq(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Coluna alvo '{target_col}' não encontrada.")
    # garantir numérico
    df = df.copy()
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    if df[target_col].isna().any():
        n_nan = int(df[target_col].isna().sum())
        log(f"[features] {n_nan} valores NaN em '{target_col}' -> removendo.", "WARNING")
        df = df.dropna(subset=[target_col]).reset_index(drop=True)
    freq = df[target_col].round().astype(int).value_counts(normalize=True).to_dict()
    df[cfg.FREQUENCY_COL] = df[target_col].round().astype(int).map(freq).fillna(0.0)
    return df

# ----------------------------------------
# Suplementares
# ----------------------------------------
def get_suplementar_cols(df: pd.DataFrame, suplementares_range: Tuple[int, int]) -> List[str]:
    a, b = suplementares_range
    cols = []
    for i in range(a, b + 1):
        name = f"{cfg.SUPPLEMENTARY_COL}{i}"
        if name in df.columns:
            cols.append(name)
    return cols

# ----------------------------------------
# Garante existência das features (ordem estável)
# ----------------------------------------
def ensure_features_exist(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    # ordena colunas do df de acordo com a lista de features quando selecionadas
    return df

# ----------------------------------------
# Monta matrizes (X_flat escalado, X_seq escalado) + y escalado
# ----------------------------------------
def prepare_features_and_matrix(
    df: pd.DataFrame,
    seq_len: int = 14,
    suplementares_range: Tuple[int, int] = cfg.SUPPLEMENTARY_RANGE,
    target_col: str = cfg.TARGET_COL
):
    """
    Retorna:
      X_flat_scaled: (n_amostras, n_feats)
      X_seq_scaled:  (n_amostras, seq_len, n_feats)
      y_scaled:      (n_amostras,)
      dates_list:    lista de datas (alvo em i+seq_len)
      feature_cols:  ordem das features usadas
      scaler_X, scaler_y
    """
    if df is None or df.empty:
        raise ValueError("DataFrame vazio em prepare_features_and_matrix.")

    df = df.copy().sort_values(cfg.DATE_COL).reset_index(drop=True)
    if len(df) <= seq_len:
        raise ValueError(f"Dados insuficientes ({len(df)}) para seq_len={seq_len}.")

    # amplia datas e frequência
    df = expand_date_features(df, date_col=cfg.DATE_COL)
    df = add_valor_freq(df, target_col)

    suplementares = get_suplementar_cols(df, suplementares_range)
    base_feats = ['year', 'month', 'day', 'dayofweek', 'is_weekend', cfg.FREQUENCY_COL]
    feature_cols = base_feats + suplementares
    df = ensure_features_exist(df, feature_cols)

    # garantir tipos numéricos para features
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    # alvo contínuo para regressão (depois arredondamos ao salvar)
    y_all = pd.to_numeric(df[target_col], errors='coerce')
    if y_all.isna().any():
        # remove onde o alvo não existe
        mask = ~y_all.isna()
        df = df.loc[mask].reset_index(drop=True)
        y_all = y_all.loc[mask].reset_index(drop=True)

    n = len(df)
    samples = n - seq_len
    n_feats = len(feature_cols)

    X_seq = np.zeros((samples, seq_len, n_feats), dtype=float)
    X_flat = np.zeros((samples, n_feats), dtype=float)
    y = np.zeros((samples,), dtype=float)
    dates_list = []

    for i in range(samples):
        win = df.iloc[i:i + seq_len]
        step = df.iloc[i + seq_len]
        X_seq[i, :, :] = win[feature_cols].values
        X_flat[i, :] = step[feature_cols].values
        y[i] = float(step[target_col])
        dates_list.append(step[cfg.DATE_COL])

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_flat_scaled = scaler_X.fit_transform(X_flat)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    X_seq_2d = X_seq.reshape(samples * seq_len, n_feats)
    X_seq_2d_scaled = scaler_X.transform(X_seq_2d)
    X_seq_scaled = X_seq_2d_scaled.reshape(samples, seq_len, n_feats)

    log(f"[features] Preparadas {samples} amostras, {seq_len} passos, {n_feats} features.", "INFO")
    return X_flat_scaled, X_seq_scaled, y_scaled, dates_list, feature_cols, scaler_X, scaler_y
