#!/usr/bin/env python3
# run_predictor.py
"""
Main orchestrator script:
- carrega config (config.py)
- preenche previsões pendentes (quando modelo.csv foi atualizado)
- treina/atualiza modelos (LightGBM + RNN)
- gera APENAS 1 previsão por execução (acrescenta linha em previsoes.csv)
- dispara auto-calibração se erros foram detectados
"""

import logging
import traceback
import pandas as pd

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
        setup_hardware()

        # Load data
        log(f"[I/O] Loading {DATA_SOURCE} ...", "INFO")
        df_model = load_data_source()
        log(f"[I/O] Loading {FORECAST_FILE} ...", "INFO")
        df_prev = load_predictions_csv()

        # Atualiza previsões pendentes
        log("[FLOW] Updating pending forecasts (if any)...", "INFO")
        df_prev_updated, updated_any = AutoCalibrator.compare_predicted_value_with_real(df_prev, df_model)
        if updated_any:
            backup_predictions()
            save_predictions(df_prev_updated, FORECAST_FILE)
            df_prev = df_prev_updated

        # Features
        df_model = expand_date_features(df_model, date_col=DATE_COL)
        df_model = add_valor_freq(df_model, TARGET_COL)
        suplementares = get_suplementar_cols(df_model, SUPPLEMENTARY_RANGE)
        base_feats = ['year', 'month', 'day', 'dayofweek', 'is_weekend', FREQUENCY_COL]
        feature_cols = base_feats + suplementares
        df_model = ensure_features_exist(df_model, feature_cols)

        # Matrizes
        log("[FEATURES] Preparing matrices/features...", "INFO")
        X_flat, X_seq, y, dates_list, feature_cols, scaler_X, scaler_y = prepare_features_and_matrix(
            df_model,
            seq_len=SEQ_LEN,
            suplementares_range=SUPPLEMENTARY_RANGE,
            target_col=TARGET_COL
        )

        # Modelos
        log("[MODELS] Building/training models (LGB + RNN)...", "INFO")
        automodels = AutoModels(export_dir=EXPORT_DIR, use_gpu=USE_GPU, random_state=RANDOM_SEED)
        automodels.fit_or_load(X_flat, X_seq, y, feature_cols, dates_list=dates_list, df_prev=df_prev)
        automodels.evaluate_and_set_weights(X_flat, X_seq, y)

        # ÚNICA previsão
        log("[PRED] Generating a single forecast for next date...", "INFO")
        pred_tuple = automodels.predict_next(df_model)
        if pred_tuple is None:
            raise RuntimeError("predict_next returned None.")
        next_date, pred_int, pred_score = pred_tuple
        log(f"[PRED] {next_date.date()} -> {NEXT_TARGET_COL_VALUE}={pred_int} (score={pred_score:.2f})", "INFO")

        # Grava somente se não existir previsão para essa data
        if not AutoCalibrator.prediction_exists(df_prev, next_date):
            new_row = {
                DATE_COL: next_date.strftime(DATE_FORMAT_INPUT),
                NEXT_TARGET_COL_VALUE: int(pred_int),
                LATEST_TARGET_COL_VALUE: None,            # ainda não sabemos
                NEXT_TARGET_COL_PERCENTUAL: float(round(pred_score * 100.0, 2)),
                NEXT_TARGET_COL_RESULT: None              # vazio até sabermos o real
            }
            df_prev2 = pd.concat([df_prev, pd.DataFrame([new_row])], ignore_index=True)
            backup_predictions()
            save_predictions(df_prev2, FORECAST_FILE)
            log(f"[I/O] Prediction saved into: {FORECAST_FILE}", "INFO")
        else:
            log("[I/O] Forecast for next date already exists. Nothing saved.", "INFO")

        # Auto-calibração se houve atualização anterior
        if updated_any:
            log("[AUTO] Errors detected — starting auto-calibration (de-weight training)...", "INFO")
            calibrator = AutoCalibrator(export_dir=EXPORT_DIR)
            calibrator.run_autocalibration(df_model, df_prev)
            log("[AUTO] Auto-calibration done.", "INFO")

        log("=== run_predictor done with success ===", "INFO")

    except Exception as e:
        logging.error("Critical error in run_predictor:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
