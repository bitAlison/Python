# sample_data_generator.py
"""
Synthetic sample generator for rapid testing.
Generates a .csv model with columns: date; value; supplemental1..supplemental14
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg
from io_utils import log

# ==============================
# Generates synthetic file for trainig models
# ==============================
def generate_synthetic_file(path=None):
    try:
        next_date = pd.date_range(start=cfg.MIN_SAMPLE_DATE, periods=cfg.MAX_RECORDS, freq='D')
        values = np.random.randint(1, 5, size=cfg.MAX_RECORDS)  # values between 1 and 5
        data = pd.DataFrame({cfg.DATE_COL: next_date, cfg.TARGET_COL: values})
        
        for i in cfg.SUPPLEMENTARY_RANGE:
            # TODO: It is necessary to ensure that the same numbers are not repeated for each of the other columns
            data[f"{cfg.SUPPLEMENTARY_COL}{i}"] = np.random.randint(cfg.TARGET_COL_RANGE_MIN, cfg.TARGET_COL_RANGE_MAX, size=cfg.MAX_RECORDS)

        path = Path(path or cfg.DATA_SOURCE)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data.to_csv(path, index=False, sep=cfg.CSV_SEPARATOR, parse_dates=[cfg.DATE_COL], dayfirst=True, encoding=cfg.FILE_ENCODING)
        print(f"Success in generating synthetic file with {path} {cfg.MAX_RECORDS} lines")
        return path
    except Exception as e:
        log(f"An error occurred while trying to generate the synthetic file for training and testing. {e}", "ERROR")

if __name__ == "__main__":
    generate_synthetic_file()
