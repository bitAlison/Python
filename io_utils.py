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
def load_predictions_csv():
    """Reads the forecast file, creates it if it does not exist."""
    try:
        path = Path(cfg.FORECAST_FILE)
        if not path.exists():
            cols = [cfg.DATE_COL, cfg.NEXT_TARGET_COL_VALUE, cfg.LATEST_TARGET_COL_VALUE, cfg.NEXT_TARGET_COL_PERCENTUAL, cfg.NEXT_TARGET_COL_RESULT]
            df_empty = pd.DataFrame(columns=cols)
            df_empty.to_csv(path, sep=";", index=False, decimal=",")
            return df_empty
        df = pd.read_csv(path, parse_dates=[cfg.DATE_COL], encoding=cfg.FILE_ENCODING, sep=cfg.CSV_SEPARATOR, dayfirst=True)
        # if cfg.DATE_COL in df.columns:
        #     df[cfg.DATE_COL] = pd.to_datetime(df[cfg.DATE_COL], errors="coerce", dayfirst=True)
        return df
    except Exception as e:
        log(f"An error occurred while trying to read the predicions. {e}", "ERROR")
        return None

# ==============================
# Save predictions in the correct format
# ==============================
def save_predictions(data, predicted_value, real_value=None, percentual_prev=None, result=None):
    """
    Adds a new line to the forecast file.
    Only one record is added per run.
    """
    try:
        if not os.path.exists(cfg.FORECAST_FILE):
            df_prev = pd.DataFrame(columns=cfg.TARGET_COL)
        else:
            df_prev = pd.read_csv(cfg.FORECAST_FILE, encoding=cfg.FILE_ENCODING, sep=cfg.CSV_SEPARATOR)

        # Rounds expected value to integer
        predicted_value_int = int(round(predicted_value))

        new_line = {
            cfg.DATE_COL: data,
            cfg.NEXT_TARGET_COL_VALUE: predicted_value_int,
            cfg.LATEST_TARGET_COL_VALUE: real_value if real_value is not None else "",
            cfg.NEXT_TARGET_COL_PERCENTUAL: round(percentual_prev, 2) if percentual_prev is not None else "",
            cfg.NEXT_TARGET_COL_RESULT: result if result else ""
        }

        df_prev = pd.concat([df_prev, pd.DataFrame([new_line])], ignore_index=True)
        df_prev.to_csv(cfg.FORECAST_FILE, sep=";", decimal=".", index=False)

        log(f"Prediction saved: {new_line}")
    except Exception as e:
        log(f"An error occurred while trying to save the predicions. {e}", "ERROR")

# ==============================
# Apeend a new prediction record in the csv file
# ==============================
def append_single_prediction(data_prev, predicted_value, real_value=None, perc_prev=None, result=None):
    """
    Adds a single forecast row to predictions.csv.
    - Rounds prev_value and actual_value to avoid float vs. int comparison issues.
    - Ensures there are no multiple records on the same prev_date.
    """
    try:
        # Round values
        predicted_value = round(predicted_value, 2) if predicted_value is not None else None
        real_value = round(real_value, 2) if real_value is not None else None
        perc_prev = round(perc_prev, 2) if perc_prev is not None else None

        df_prev = load_predictions_csv()

        # Remove predictions with the same date
        df_prev = df_prev[df_prev[cfg.DATE_COL] != pd.to_datetime(data_prev)]

        # New prediction record
        new_line = {
            cfg.DATE_COL: pd.to_datetime(data_prev),
            cfg.NEXT_TARGET_COL_VALUE: predicted_value,
            cfg.LATEST_TARGET_COL_VALUE: real_value,
            cfg.NEXT_TARGET_COL_PERCENTUAL: perc_prev,
            cfg.NEXT_TARGET_COL_RESULT: result
        }

        df_prev = pd.concat([df_prev, pd.DataFrame([new_line])], ignore_index=True)

        # Sort
        df_prev = df_prev.sort_values(cfg.DATE_COL).reset_index(drop=True)

        backup_predictions()
        df_prev.to_csv(cfg.FORECAST_FILE, sep=";", index=False, decimal=".")
    except Exception as e:
        log(f"An error occurred while trying to append a new record to the predicions file. {e}", "ERROR")

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
