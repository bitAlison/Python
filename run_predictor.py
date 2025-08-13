#!/usr/bin/env python3
# run_predictor.py
"""
Orquestrador:
- Lê Modelo.csv
- Atualiza previsoes.csv com acertos/erros passados
- Prepara features/matrizes
- Treina ou carrega modelos (LightGBM + RNN, Optuna)
- Gera APENAS UMA previsão (próxima data) e anexa ao previsoes.csv
"""

import logging
import traceback
import pandas as pd

from config import *
from io_utils import (
    ensure_dirs,
    load_data_source,
    load_predictions_csv,
    save_predictions,
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

def main():
    log("=== Iniciando run_predictor ===", "INFO")
    try:
        ensure_dirs()
        setup_hardware()

        # 1) Carga
        df_model = load_data_source()
        df_prev = load_predictions_csv()

        # 2) Atualiza previsões antigas com valores reais (se disponíveis)
        log("[FLOW] Atualizando previsões passadas...", "INFO")
        df_prev_updated, updated_any = AutoCalibrator.compare_predicted_value_with_real(df_prev, df_model)
        if updated_any:
            backup_predictions()
            save_predictions(df_prev_updated, FORECAST_FILE)
            df_prev = df_prev_updated

        # 3) Preparar features
        df_model = expand_date_features(df_model, date_col=DATE_COL)
        df_model = add_valor_freq(df_model, TARGET_COL)
        suplementares = get_suplementar_cols(df_model, SUPPLEMENTARY_RANGE)
        base_feats = ['year', 'month', 'day', 'dayofweek', 'is_weekend', FREQUENCY_COL]
        feature_cols = base_feats + suplementares
        df_model = ensure_features_exist(df_model, feature_cols)

        prep = prepare_features_and_matrix(
            df_model,
            seq_len=SEQ_LEN,
            suplementares_range=SUPPLEMENTARY_RANGE,
            target_col=TARGET_COL
        )
        X_flat, X_seq, y, dates_list, feature_cols, scaler_X, scaler_y = prep

        # 4) Modelos
        automodels = AutoModels(export_dir=EXPORT_DIR, use_gpu=USE_GPU, random_state=RANDOM_SEED)
        automodels.fit_or_load(X_flat, X_seq, y, feature_cols, dates_list=dates_list, df_prev=df_prev)
        automodels.evaluate_and_set_weights(X_flat, X_seq, y)

        # 5) Única previsão
        log("[PRED] Gerando previsão única...", "INFO")
        next_date, pred_int, pred_score = automodels.predict_next(df_model)
        log(f"[PRED] {next_date.date()} -> {NEXT_TARGET_COL_VALUE}={pred_int} (score={pred_score:.2f})", "INFO")

        # 6) Persistência (apenas 1 linha nova se ainda não existir)
        if not AutoCalibrator.prediction_exists(df_prev, next_date):
            new_row = {
                DATE_COL: next_date.strftime(DATE_FORMAT_INPUT),
                NEXT_TARGET_COL_VALUE: int(pred_int),
                LATEST_TARGET_COL_VALUE: "",
                NEXT_TARGET_COL_PERCENTUAL: round(float(pred_score) * 100.0, 2),
                NEXT_TARGET_COL_RESULT: ""
            }
            df_prev2 = df_prev.copy()
            df_prev2 = pd.concat([df_prev2, pd.DataFrame([new_row])], ignore_index=True)
            backup_predictions(FORECAST_FILE)
            save_predictions(df_prev2, FORECAST_FILE)
            log(f"[I/O] Previsão salva em {FORECAST_FILE}", "INFO")
        else:
            log("[I/O] Já existe previsão para a próxima data — nada a salvar.", "INFO")

        # 7) Autocalibração (se houve atualização com erros)
        if updated_any:
            log("[AUTO] Iniciando autocalibração...", "INFO")
            calibrator = AutoCalibrator(export_dir=EXPORT_DIR)
            calibrator.run_autocalibration(df_model, df_prev)
            log("[AUTO] Autocalibração concluída.", "INFO")

        log("=== run_predictor finalizado com sucesso ===", "INFO")

    except Exception:
        logging.error("Critical error in run_predictor:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
