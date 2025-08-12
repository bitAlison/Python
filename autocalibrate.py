# autocalibrate.py
"""
AutoCalibrator: functions for:
- update forecasts.csv when model.csv is updated (fill in actual_value + result)
- check if a forecast for next_date already exists
- run a recalibration routine: adjust weights / retrain models when there are errors
"""

import pandas as pd
from pathlib import Path
from config import *
from models import AutoModels
from io_utils import log


#log = logging.getLogger(__name__)

class AutoCalibrator:
    # ==============================
    # Compare predicted value with real
    # ==============================
    @staticmethod
    def compare_predicted_value_with_real(df_prev: pd.DataFrame, df_model: pd.DataFrame):
        """
        For each row in df_prev without actual_value, attempts to find a match in df_model
        by date and updates actual_value/predicted_percentage/result.
        Returns (df_prev_updated, updated_any)
        """
        try:
            if df_prev is None or df_prev.empty:
                return df_prev, False
            
            df_prev = df_prev.copy()
            updated_any = False
            for i, row in df_prev.iterrows():
                if pd.isna(row.get(LASTEST_TARGET_COL_VALUE)) or str(row.get(LASTEST_TARGET_COL_VALUE)).strip() == '':
                    try:
                        dt = pd.to_datetime(row[DATE_COL]).normalize()
                    except Exception:
                        continue
                    match = df_model[df_model[DATE_COL].dt.normalize() == dt]
                    if not match.empty:
                        real_val = int(match.iloc[0][TARGET_COL])
                        previsao = row[NEXT_TARGET_COL_VALUE]

                        # ensure comparable integers
                        previsao_int = int(round(float(previsao)))
                        resultado = NEXT_STATUS_SUCCESS if previsao_int == real_val else NEXT_STATUS_ERROR

                        if real_val != 0:
                            perc = round((1 - abs(previsao_int - real_val) / abs(real_val)) * 100, 2)
                        else:
                            perc = None

                        df_prev.at[i, LASTEST_TARGET_COL_VALUE] = real_val
                        df_prev.at[i, NEXT_TARGET_COL_PERCENTUAL] = perc
                        df_prev.at[i, NEXT_TARGET_COL_RESULT] = resultado

                        updated_any = True
                        log(f"Prediction result for the next date {dt.date()}: prev={previsao_int} real={real_val} -> {resultado}", "INFO")
            return df_prev, updated_any
        except Exception as e:
            log(f"An error occurred while trying to update the previous value with the actual value. {e}", "ERROR")
            return None, False

    # ==============================
    # Check if prediction exists
    # ==============================
    @staticmethod
    def prediction_exists(df_prev: pd.DataFrame, next_date):
        if df_prev is None or df_prev.empty:
            return False
        try:
            return any(pd.to_datetime(df_prev[DATE_COL]).dt.normalize() == pd.to_datetime(next_date).normalize())
        except Exception:
            return False

    def __init__(self, export_dir=None):
        try:
            self.export_dir = Path(export_dir or EXPORT_DIR)
            _ = self.export_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log(f"An error occurred while trying to initializing autocalibrate object. {e}", "ERROR")

    # ==============================
    # Run auto calibration
    # ==============================
    def run_autocalibration(self, df_model: pd.DataFrame, df_prev: pd.DataFrame):
        """
        Example of a self-calibration routine:
        - when it detects recent errors, it increases weights and retrains models
        - it performs up to MAX_RETRIES_RECALIBRATION
        """
        try:

            # first identify current percentages
            if df_prev is None or df_prev.empty:
                log("No previous predictions to calibrate.", "INFO")
                return False

            last_errors = df_prev[df_prev[NEXT_TARGET_COL_RESULT] == 'erro']
            if last_errors.empty:
                log("No recent errors detected—self-calibration not required.", "INFO")
                return False

            autom = AutoModels(export_dir=self.export_dir, use_gpu=USE_GPU, random_state=RANDOM_SEED)

            # reconstruct X/y using features.py equivalent (caller should pass full matrices, but we keep generic)
            from features import prepare_features_and_matrix
            X_flat, X_seq, y, dates, feature_cols, scaler_X, scaler_y = prepare_features_and_matrix(df_model,
                                                                                               seq_len=SEQ_LEN,
                                                                                               suplementares_range=SUPPLEMENTARY_RANGE,
                                                                                               target_col=TARGET_COL)
            # increase Optuna trials for autocalibration (optional)
            orig_trials = OPTUNA_TRIALS
            try:
                log("Starting self-calibration cycle: re-training with weights and Optuna.", "INFO")
                autom.fit_or_load(X_flat, X_seq, y, dates_list=dates, df_prev=df_prev)
                autom.evaluate_and_set_weights(X_flat, X_seq, y)
                autom.save_artifacts()
                log("Auto-calibration finished.", "INFO")
                return True
            except Exception as e:
                log(f"Auto-calibration failed: {e}", "ERROR")
                return False
        except Exception as e:
            log(f"An error occurred running autocalibration. {e}", "ERROR")
            return False
