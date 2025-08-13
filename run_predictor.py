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
import pandas as pd 
import logging
import traceback

from config import *
from io_utils import (
    load_data_source, 
    load_predictions_csv,
    save_predictions,
    ensure_dirs,
    backup_predictions,
    log
)
from features import (
    expand_date_features,
    add_valor_freq,
    get_suplementar_cols,
    ensure_features_exist,
    prepare_features_and_matrix
)
from models import AutoModels
from autocalibrate import AutoCalibrator
from hardware import setup_hardware

def main():
    log("=== Initializing run_predictor ===", "INFO")
    try:
        ensure_dirs()
        setup_hardware()  # try to configure GPU/limits

        # Load data
        log(f"[I/O] Loading {DATA_SOURCE} ...", "INFO")
        df_model = load_data_source()
        log(f"[I/O] Loading {FORECAST_FILE} ...", "INFO")
        df_prev = load_predictions_csv()

        # Update pending forecasts (fill in actual value and outcome when possible)
        log("[FLOW] Updating pending forecasts (if any)...", "INFO")
        df_prev_updated, updated_any = AutoCalibrator.compare_predicted_value_with_real(df_prev, df_model)
        if updated_any:
            backup_predictions()
            save_predictions(df_prev_updated, FORECAST_FILE)
            df_prev = df_prev_updated  # refresh in-memory

        # ========================
        # Preparar features
        # ========================
        df_model = expand_date_features(df_model, date_col=DATE_COL)
        df_model = add_valor_freq(df_model, TARGET_COL)
        suplementares = get_suplementar_cols(df_model, SUPPLEMENTARY_RANGE)
        base_feats = ['year', 'month', 'day', 'dayofweek', 'is_weekend', FREQUENCY_COL]
        feature_cols = base_feats + suplementares
        df_model = ensure_features_exist(df_model, feature_cols)

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
        #automodels.fit_or_load(X_flat, X_seq, y, dates_list, df_prev)
        automodels.fit_or_load(X_flat, X_seq, y, feature_cols, dates_list=dates_list, df_prev=df_prev)


        # Evaluates internal validation and defines ensemble weights
        automodels.evaluate_and_set_weights(X_flat, X_seq, y)

        # Generate only ONE forecast for next_date
        log("[PRED] Generating a forecast for the next date...", "INFO")
        #next_date, pred_int, pred_score = automodels.predict_next(df_model, feature_cols, scaler_X, scaler_y)
        next_date, pred_int, pred_score = automodels.predict_next(df_model)

        log(f"[PRED] Next date: {next_date.date()} -> {NEXT_TARGET_COL_VALUE}={pred_int} (score proxy={pred_score:.2f})", "INFO")

        # Append forecast (only one record) - empty actual_value, empty result
        if not AutoCalibrator.prediction_exists(df_prev, next_date):
            new_row = {
                DATE_COL: next_date.strftime(DATE_FORMAT_INPUT),
                NEXT_TARGET_COL_VALUE: int(pred_int),
                LATEST_TARGET_COL_VALUE: '',
                NEXT_TARGET_COL_PERCENTUAL: round(pred_score * 100.0, 2),
                NEXT_TARGET_COL_RESULT: ''
            }
            df_prev2 = df_prev.copy()
            # Garante que todas as colunas já existam no new_row
            #new_row = {col: new_row.get(col, None) for col in df_prev.columns}
            df_prev2 = pd.concat([df_prev2, pd.DataFrame([new_row])], ignore_index=True)
            backup_predictions()
            save_predictions(df_prev2, int(pred_int), "", round(pred_score * 100.0, 2), "")
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
