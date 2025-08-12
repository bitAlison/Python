#!/usr/bin/env python3
# run_predictor.py
"""
Main orchestrator script:
- Loads configurations (config.py)
- Populates pending predictions (when DataSource.csv has been updated)
- Trains/updates models (LightGBM + RNN) using supplementary features
- Generates ONLY ONE prediction per run (appends line in predictions.csv)
- Triggers autocalibration if an error is detected in previous predictions
"""

import logging
import traceback

from config import *
from io_utils import load_data_source, load_predictions_csv, save_predictions, ensure_dirs, backup_predictions, log
from features import prepare_features_and_matrix
from models import AutoModels
from autocalibrate import AutoCalibrator
from hardware import setup_hardware

def main():
    log("=== Initializing run_predictor ===", "INFO")
    try:
        ensure_dirs(PROJECT_DIR, EXPORT_DIR)
        setup_hardware(RAM_LIMIT_GB, GPU_MEMORY_LIMIT_MB)  # try to configure GPU/limits

        # Load data
        log(f"[I/O] Loading {DATA_SOURCE} ...", "INFO")
        df_model = load_data_source(DATA_SOURCE)
        log(f"[I/O] Loading {FORECAST_FILE} ...", "INFO")
        df_prev = load_predictions_csv(FORECAST_FILE)

        # Update pending forecasts (fill in actual value and outcome when possible)
        log("[FLOW] Updating pending forecasts (if any)...", "INFO")
        df_prev_updated, updated_any = AutoCalibrator.compare_predicted_value_with_real(df_prev, df_model)
        if updated_any:
            backup_predictions(FORECAST_FILE)
            save_predictions(df_prev_updated, FORECAST_FILE)
            df_prev = df_prev_updated  # refresh in-memory

        # Prepares features and matrices (X_flat, X_seq, y, dates, feature_cols, scalers)
        log("[FEATURES] Preparing matrices/features...", "INFO")
        prep = prepare_features_and_matrix(df_model,
                                          seq_len=SEQ_LEN,
                                          suplementares_range=SUPPLEMENTARY_RANGE,
                                          target_col=TARGET_COL)
        
        X_flat, X_seq, y, dates_list, feature_cols, scaler_X, scaler_y = prep

        # Create/Load models (training and/or reuse)
        log("[MODELS] Building/training models (LGB + RNN)...", "INFO")
        automodels = AutoModels(export_dir=EXPORT_DIR, use_gpu=USE_GPU, random_state=RANDOM_SEED)
        automodels.fit_or_load(X_flat, X_seq, y, dates_list, df_prev)

        # Evaluates internal validation and defines ensemble weights
        automodels.evaluate_and_set_weights(X_flat, X_seq, y)

        # Generate only ONE forecast for next_date
        log("[PRED] Generating a forecast for the next date...", "INFO")
        next_date, pred_int, pred_score = automodels.predict_next(df_model, feature_cols, scaler_X, scaler_y)
        log(f"[PRED] Next date: {next_date.date()} -> {NEXT_TARGET_COL_VALUE}={pred_int} (score proxy={pred_score:.2f})", "INFO")

        # Append forecast (only one record) - empty actual_value, empty result
        if not AutoCalibrator.prediction_exists(df_prev, next_date):
            new_row = {
                DATE_COL: next_date.strftime(DATE_FORMAT_INPUT),
                NEXT_TARGET_COL_VALUE: int(pred_int),
                LASTEST_TARGET_COL_VALUE: '',
                NEXT_TARGET_COL_PERCENTUAL: round(pred_score * 100.0, 2),
                NEXT_TARGET_COL_RESULT: ''
            }
            df_prev2 = df_prev.copy()
            df_prev2 = df_prev2.append(new_row, ignore_index=True)
            backup_predictions(FORECAST_FILE)
            save_predictions(df_prev2, FORECAST_FILE)
            logging.info(f"[I/O] Prediction saved in: {FORECAST_FILE}")
        else:
            logging.info("[I/O] There is already a forecast for the next date - nothing saved.")

        # If there are updates (new errors detected), it triggers autocalibration
        if updated_any:
            logging.info("[AUTO] Errors detected — starting auto-calibration (de-weight training)...")
            calibrator = AutoCalibrator(export_dir=EXPORT_DIR)
            calibrator.run_autocalibration(df_model, df_prev)
            logging.info("[AUTO] auto calibration done.")

        logging.info("=== run_predictor done with success ===")

    except Exception as e:
        logging.error("Critical error in run_predictor:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
