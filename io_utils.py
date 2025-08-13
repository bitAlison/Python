# io_utils.py
"""
Input/output and file manipulation functions for the Extreme Incremental Predictor.
Includes:
- Secure CSV reading
- Incremental writing to predictions.csv
- Automatic backup
- Directory creation
"""

import os
from pathlib import Path
import pandas as pd
from config import *
import shutil
import datetime
import logging


# ==============================
# Ensure dirs func
# ==============================
def ensure_dirs():
    """Creates directories needed for the project."""
    try:
        Path(EXPORT_DIR).mkdir(parents=True, exist_ok=True)
        Path(BACKUP_DIR).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log(f"An error occurred while trying to create the directories. {e}", "ERROR")

# ==============================
# Log Function
# ==============================
def log(msg: str, level: str = "INFO"):
    getattr(logging, level.lower())(msg)

def _create_template_if_missing():
    if not Path(DATA_SOURCE).exists():
        log(f"[I/O] Arquivo {DATA_SOURCE} não encontrado — criando template.", "WARNING")
        Path(DATA_SOURCE).parent.mkdir(parents=True, exist_ok=True)
        cols = [DATE_COL, TARGET_COL] + [f"{SUPPLEMENTARY_COL}{i}" for i in range(SUPPLEMENTARY_RANGE[0], SUPPLEMENTARY_RANGE[1] + 1)]
        pd.DataFrame(columns=cols).to_csv(DATA_SOURCE, sep=CSV_SEP, index=False, encoding=FILE_ENCODING)

# ==============================
# Backup predictions
# ==============================
def backup_predictions(path: str = None):
    path = path or FORECAST_FILE
    if Path(path).exists():
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = Path(BACKUP_DIR) / f"previsoes_{ts}.bak.csv"
        shutil.copy2(path, dest)
        log(f"[I/O] Backup criado: {dest}", "INFO")

# ==============================
# Load data from historical source csv file
# ==============================
def load_data_source() -> pd.DataFrame:
    """Reads the DataSource.csv file and returns a clean DataFrame."""
    try:
        _create_template_if_missing()
        df = pd.read_csv(DATA_SOURCE, sep=CSV_SEPARATOR, encoding=FILE_ENCODING)
        if DATE_COL not in df.columns or TARGET_COL not in df.columns:
            raise ValueError(f"Arquivo {DATA_SOURCE} deve conter as colunas '{DATE_COL}' e '{TARGET_COL}'.")
        return df
    except Exception as e:
        log(f"[I/O] Erro ao ler DataSource: {e}", "ERROR")
        raise

# ==============================
# Load prediction data from csv file
# ==============================
def load_predictions_csv() -> pd.DataFrame:
    if not Path(FORECAST_FILE).exists():
        Path(FORECAST_FILE).parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(columns=[DATE_COL, NEXT_TARGET_COL_VALUE, LATEST_TARGET_COL_VALUE, NEXT_TARGET_COL_PERCENTUAL, NEXT_TARGET_COL_RESULT])
        df.to_csv(FORECAST_FILE, sep=CSV_SEPARATOR, index=False, encoding=FILE_ENCODING)
        return df
    try:
        df = pd.read_csv(FORECAST_FILE, sep=CSV_SEPARATOR, encoding=FILE_ENCODING, dtype=str)
        return df
    except Exception as e:
        log(f"[I/O] Erro ao ler previsões: {e}", "ERROR")
        raise

# ==============================
# Save predictions in the correct format
# ==============================
def save_predictions(df: pd.DataFrame, path: str = None):
    path = path or FORECAST_FILE
    # Garantir ordem e tipos
    out = pd.DataFrame(columns=[DATE_COL, NEXT_TARGET_COL_VALUE, LATEST_TARGET_COL_VALUE, NEXT_TARGET_COL_PERCENTUAL, NEXT_TARGET_COL_RESULT])
    if not df.empty:
        # coage e arredonda
        tmp = df.copy()
        if NEXT_TARGET_COL_VALUE in tmp.columns:
            tmp[NEXT_TARGET_COL_VALUE] = pd.to_numeric(tmp[NEXT_TARGET_COL_VALUE], errors='coerce').round().astype('Int64')
        if LATEST_TARGET_COL_VALUE in tmp.columns:
            tmp[LATEST_TARGET_COL_VALUE] = pd.to_numeric(tmp[LATEST_TARGET_COL_VALUE], errors='coerce')
        if NEXT_TARGET_COL_PERCENTUAL in tmp.columns:
            tmp[NEXT_TARGET_COL_PERCENTUAL] = pd.to_numeric(tmp[NEXT_TARGET_COL_PERCENTUAL], errors='coerce').round(2)
        out = tmp[[DATE_COL, NEXT_TARGET_COL_VALUE, LATEST_TARGET_COL_VALUE, NEXT_TARGET_COL_PERCENTUAL, NEXT_TARGET_COL_RESULT]]
    out.to_csv(path, sep=CSV_SEPARATOR, index=False, encoding=FILE_ENCODING)

# ==============================
# Apeend a new prediction record in the csv file
# ==============================
def append_prediction_line(date_str: str, pred_int: int, percent: float, path: str = None):
    """Opcional: helper para acrescentar uma linha de previsão."""
    df_prev = load_predictions_csv(path)
    new_line = {
        DATE_COL: date_str,
        NEXT_TARGET_COL_VALUE: int(pred_int),
        LATEST_TARGET_COL_VALUE: '',
        NEXT_TARGET_COL_PERCENTUAL: float(round(percent, 2)),
        NEXT_TARGET_COL_RESULT: ''
    }
    df_prev = pd.concat([df_prev, pd.DataFrame([new_line])], ignore_index=True)
    save_predictions(df_prev, path)

# ==============================
# Update prediction result in the csv file
# ==============================
def update_prediction_result(data_prev, real_value):
    """
    Updates the actual_value and result in predictions.csv based on the date.
    
    """
    try:
        df_prev = load_predictions_csv()
        data_prev = pd.to_datetime(data_prev)

        if data_prev not in df_prev[DATE_COL].values:
            return False

        # Update prediction value
        idx = df_prev[df_prev[DATE_COL] == data_prev].index[0]
        df_prev.at[idx, LATEST_TARGET_COL_VALUE] = round(real_value, 2)

        # Compare with expected_value (also rounded)
        vp = round(df_prev.at[idx, NEXT_TARGET_COL_VALUE], 2)
        vr = round(real_value, 2)
        df_prev.at[idx, NEXT_TARGET_COL_RESULT] = NEXT_STATUS_SUCCESS if vp == vr else NEXT_STATUS_ERROR

        # Calculate predicted percentage (inverse relative error)
        if vr != 0:
            erro_rel = abs(vp - vr) / abs(vr)
            df_prev.at[idx, NEXT_TARGET_COL_PERCENTUAL] = round((1 - erro_rel) * 100, 2)
        else:
            df_prev.at[idx, NEXT_TARGET_COL_PERCENTUAL] = None

        backup_predictions()
        df_prev.to_csv(FORECAST_FILE, sep=";", index=False, decimal=",")
        return True
    except Exception as e:
        log(f"An error occurred while trying to append a new record to the predicions file. {e}", "ERROR")
        return False
