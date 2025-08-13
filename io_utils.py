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
import shutil
import pandas as pd
from datetime import datetime
from pathlib import Path
import config as cfg

# ==============================
# Ensure dirs func
# ==============================
def ensure_dirs():
    """Creates directories needed for the project."""
    try:
        for d in [cfg.PROJECT_DIR, cfg.EXPORT_DIR, cfg.LOG_DIR, cfg.BACKUP_DIR]:
            if not os.path.exists(d):
                Path(d).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        log(f"An error occurred while trying to create the directories. {e}", "ERROR")

# ==============================
# Log Function
# ==============================
def log(msg, tipo="INFO"):
    if tipo in ("ERROR", "WARNING"):
        print(f"[{tipo}] {msg}")

# ==============================
# Backup predictions
# ==============================
def backup_predictions():
    """Creates a backup of the forecast file before overwriting."""
    if not cfg.AUTO_BACKUP:
        return None

    try:
        predicionts_path = Path(cfg.FORECAST_FILE)
        if predicionts_path.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{cfg.OUTPUT_NAME}_{ts}{cfg.OUTPUT_EXTENSION}"
            backup_path = Path(cfg.BACKUP_DIR) / backup_name
            shutil.copy2(predicionts_path, backup_path)

            # Remove old backups if limit exceeded
            backups = sorted(Path(cfg.BACKUP_DIR).glob(f"{cfg.OUTPUT_NAME}_*{cfg.OUTPUT_EXTENSION}"), key=os.path.getmtime)
            if len(backups) > cfg.PREV_BACKUP_KEEP:
                for old in backups[:-cfg.PREV_BACKUP_KEEP]:
                    try:
                        old.unlink()
                    except OSError:
                        pass
            return backup_path
        return None
    except Exception as e:
        log(f"An error occurred while trying to perform automatic backup. {e}", "ERROR")
        return None

# ==============================
# Load data from historical source csv file
# ==============================
def load_data_source():
    """Reads the DataSource.csv file and returns a clean DataFrame."""
    try:
        path = Path(cfg.DATA_SOURCE)
        if not path.exists():
            raise FileNotFoundError(f"Template file not found: {path}")

        # Date conversion
        df = pd.read_csv(path, parse_dates=[cfg.DATE_COL], encoding=cfg.FILE_ENCODING, sep=cfg.CSV_SEPARATOR, dayfirst=True)
        #df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], errors="coerce", dayfirst=True)
        df = df.dropna(subset=[cfg.DATE_COL, cfg.TARGET_COL])

        return df.sort_values(cfg.DATE_COL).reset_index(drop=True)
    except Exception as e:
        log(f"An error occurred while trying to read the data source. {e}", "ERROR")
        return None

# ==============================
# Load prediction data from csv file
# ==============================
def load_predictions_csv(path: str = None) -> pd.DataFrame:
    path = path or cfg.FORECAST_FILE
    if not os.path.exists(path):
        # retorna DF vazio com colunas no padrão
        cols = [cfg.DATE_COL, cfg.NEXT_TARGET_COL_VALUE, cfg.LATEST_TARGET_COL_VALUE, cfg.NEXT_TARGET_COL_PERCENTUAL, cfg.NEXT_TARGET_COL_RESULT]
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(path, sep=';', encoding=cfg.FILE_ENCODING, dtype=str)
    # normaliza colunas e tipos
    for c in [cfg.NEXT_TARGET_COL_VALUE, cfg.LATEST_TARGET_COL_VALUE]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if cfg.NEXT_TARGET_COL_PERCENTUAL in df.columns:
        df[cfg.NEXT_TARGET_COL_PERCENTUAL] = pd.to_numeric(df[cfg.NEXT_TARGET_COL_PERCENTUAL], errors='coerce')
    return df

# ==============================
# Save predictions in the correct format
# ==============================
def save_predictions(df: pd.DataFrame, path: str = None):
    path = path or cfg.FORECAST_FILE
    # garante ordem e tipos
    out = pd.DataFrame({
        cfg.DATE_COL: df.get(cfg.DATE_COL, pd.Series(dtype=str)),
        cfg.NEXT_TARGET_COL_VALUE: df.get(cfg.NEXT_TARGET_COL_VALUE, pd.Series(dtype=float)).round().astype('Int64'),
        cfg.LATEST_TARGET_COL_VALUE: df.get(cfg.LATEST_TARGET_COL_VALUE, pd.Series(dtype=float)).round().astype('Int64'),
        cfg.NEXT_TARGET_COL_PERCENTUAL: df.get(cfg.NEXT_TARGET_COL_PERCENTUAL, pd.Series(dtype=float)).round(2),
        cfg.NEXT_TARGET_COL_RESULT: df.get(cfg.NEXT_TARGET_COL_RESULT, pd.Series(dtype=str)).fillna('')
    })
    out.to_csv(path, sep=';', index=False, encoding=cfg.FILE_ENCODING)

# ==============================
# Apeend a new prediction record in the csv file
# ==============================
def append_prediction_line(date_str: str, pred_int: int, percent: float, path: str = None):
    """Opcional: helper para acrescentar uma linha de previsão."""
    df_prev = load_predictions_csv(path)
    new_line = {
        cfg.DATE_COL: date_str,
        cfg.NEXT_TARGET_COL_VALUE: int(pred_int),
        cfg.LATEST_TARGET_COL_VALUE: '',
        cfg.NEXT_TARGET_COL_PERCENTUAL: float(round(percent, 2)),
        cfg.NEXT_TARGET_COL_RESULT: ''
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

        if data_prev not in df_prev[cfg.DATE_COL].values:
            return False

        # Update prediction value
        idx = df_prev[df_prev[cfg.DATE_COL] == data_prev].index[0]
        df_prev.at[idx, cfg.LATEST_TARGET_COL_VALUE] = round(real_value, 2)

        # Compare with expected_value (also rounded)
        vp = round(df_prev.at[idx, cfg.NEXT_TARGET_COL_VALUE], 2)
        vr = round(real_value, 2)
        df_prev.at[idx, cfg.NEXT_TARGET_COL_RESULT] = cfg.NEXT_STATUS_SUCCESS if vp == vr else cfg.NEXT_STATUS_ERROR

        # Calculate predicted percentage (inverse relative error)
        if vr != 0:
            erro_rel = abs(vp - vr) / abs(vr)
            df_prev.at[idx, cfg.NEXT_TARGET_COL_PERCENTUAL] = round((1 - erro_rel) * 100, 2)
        else:
            df_prev.at[idx, cfg.NEXT_TARGET_COL_PERCENTUAL] = None

        backup_predictions()
        df_prev.to_csv(cfg.FORECAST_FILE, sep=";", index=False, decimal=",")
        return True
    except Exception as e:
        log(f"An error occurred while trying to append a new record to the predicions file. {e}", "ERROR")
        return False
