# models.py
"""
AutoModels
- LightGBM (Optuna) + RNN (GRU)
- Armazenamento de artefatos
- Ensemble e predição do próximo valor (apenas 1)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from config import *
from io_utils import log
from features import ensure_features_exist

def _ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def _build_rnn(seq_len, n_feats, units=128):
    model = Sequential()
    model.add(GRU(units, input_shape=(seq_len, n_feats)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

class AutoModels:
    def __init__(self, export_dir=None, use_gpu=True, random_state=42):
        self.export_dir = Path(export_dir or EXPORT_DIR)
        _ensure_dir(self.export_dir)
        self.use_gpu = use_gpu and bool(tf.config.list_physical_devices('GPU'))
        self.random_state = int(random_state)

        self.lgb_model = None
        self.rnn_model = None
        self.scaler_X: StandardScaler = None
        self.scaler_y: StandardScaler = None
        self.feature_cols = None
        self.seq_len = SEQ_LEN
        self.weights = (0.5, 0.5)

        # caminhos artefatos
        self._p_lgb = self.export_dir / LGB_FILENAME        # ex: lgb_model.txt
        self._p_rnn = self.export_dir / RNN_FILENAME        # ex: rnn_model.keras
        self._p_scaler_x = self.export_dir / SCALER_FILENAME
        self._p_scaler_y = self.export_dir / SCALER_Y_FILENAME
        self._p_meta = self.export_dir / METADATA_JSON

    # ------------------------------
    # Persistência
    # ------------------------------
    def save_artifacts(self):
        try:
            if self.lgb_model is not None:
                self.lgb_model.save_model(str(self._p_lgb))
            if self.rnn_model is not None:
                # Keras 3: formato nativo .keras (sem include_optimizer)
                self.rnn_model.save(str(self._p_rnn))
            if self.scaler_X is not None:
                joblib.dump(self.scaler_X, str(self._p_scaler_x))
            if self.scaler_y is not None:
                joblib.dump(self.scaler_y, str(self._p_scaler_y))
            meta = {
                "seq_len": int(self.seq_len),
                "feature_cols": list(self.feature_cols) if self.feature_cols is not None else None,
                "weights": [float(self.weights[0]), float(self.weights[1])],
                "use_gpu": bool(self.use_gpu)
            }
            with open(self._p_meta, "w", encoding=FILE_ENCODING) as f:
                json.dump(meta, f, indent=2)
            log("[models] Artefatos salvos.", "INFO")
        except Exception as e:
            log(f"[models] Falha ao salvar artefatos: {e}", "ERROR")

    def load_artifacts(self):
        try:
            if self._p_lgb.exists():
                self.lgb_model = lgb.Booster(model_file=str(self._p_lgb))
            if self._p_rnn.exists():
                self.rnn_model = load_model(str(self._p_rnn))
            if self._p_scaler_x.exists():
                self.scaler_X = joblib.load(str(self._p_scaler_x))
            if self._p_scaler_y.exists():
                self.scaler_y = joblib.load(str(self._p_scaler_y))
            if self._p_meta.exists():
                meta = json.load(open(self._p_meta, "r", encoding=FILE_ENCODING))
                self.seq_len = meta.get("seq_len", self.seq_len)
                self.feature_cols = meta.get("feature_cols", self.feature_cols)
                w = meta.get("weights", self.weights)
                self.weights = (float(w[0]), float(w[1]))
            log("[models] Artefatos carregados (se existiam).", "INFO")
        except Exception as e:
            log(f"[models] Falha ao carregar artefatos: {e}", "ERROR")

    # ------------------------------
    # Optuna (LightGBM)
    # ------------------------------
    def _optuna_obj(self, trial, X_tr, y_tr, X_val, y_val):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "verbosity": -1,
            "num_threads": NUM_THREADS or 4,
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 3, 100),
        }
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
        callbacks = [lgb.log_evaluation(period=0)]
        booster = lgb.train(params, dtr, valid_sets=[dval], callbacks=callbacks)
        preds = booster.predict(X_val, num_iteration=booster.best_iteration)
        rmse = mean_squared_error(y_val, preds, squared=False)
        return rmse

    # ------------------------------
    # Treino / carga
    # ------------------------------
    def fit_or_load(self, X_flat, X_seq, y, feature_cols, dates_list=None, df_prev=None):
        """
        Se artefatos existirem, carrega. Senão, treina (LGB + RNN).
        """
        self.load_artifacts()
        if (self.lgb_model is not None and self.rnn_model is not None
                and self.scaler_X is not None and self.scaler_y is not None
                and self.feature_cols is not None):
            log("[models] Artefatos existentes — pulando treino.", "INFO")
            return

        # mantém as features usadas no treino
        self.feature_cols = list(feature_cols)

        # split
        n = X_flat.shape[0]
        split = max(int(n * 0.85), 1)
        X_tr, X_val = X_flat[:split], X_flat[split:]
        y_tr, y_val = y[:split], y[split:]
        seq_tr, seq_val = X_seq[:split], X_seq[split:]

        # pesos (deu erro anteriormente -> peso maior p/ datas com 'erro' no histórico)
        weights = np.ones(n, dtype=float)
        if df_prev is not None and not df_prev.empty and dates_list is not None:
            # mapeia data->índice do sample alvo
            date_to_idx = {pd.Timestamp(d).normalize(): i for i, d in enumerate(dates_list)}
            for _, r in df_prev.iterrows():
                res = str(r.get(NEXT_TARGET_COL_RESULT, "")).strip().lower()
                if res == NEXT_STATUS_ERROR:
                    try:
                        d = pd.to_datetime(r[DATE_COL]).normalize()
                        if d in date_to_idx:
                            weights[date_to_idx[d]] *= 3.0
                    except Exception:
                        pass
            weights = weights / np.clip(weights.mean(), 1e-9, None)

        # Optuna
        try:
            log("[models] Iniciando Optuna (LightGBM)...", "INFO")
            study = optuna.create_study(direction="minimize")
            study.optimize(lambda t: self._optuna_obj(t, X_tr, y_tr, X_val, y_val),
                           n_trials=OPTUNA_TRIALS, n_jobs=1)
            best = study.best_params
            lgb_params = {
                "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
                "verbosity": -1, "num_threads": NUM_THREADS or 4,
                "num_leaves": best["num_leaves"],
                "learning_rate": best["learning_rate"],
                "feature_fraction": best["feature_fraction"],
                "bagging_fraction": best["bagging_fraction"],
                "bagging_freq": best["bagging_freq"],
                "min_data_in_leaf": best["min_data_in_leaf"]
            }
            log(f"[models] Optuna concluído: {best}", "INFO")
        except Exception as e:
            log(f"[models] Optuna falhou ({e}) — usando parâmetros padrão.", "WARNING")
            lgb_params = PARAMS_LGBM_DEFAULT.copy()

        # Treina LGB
        dtrain = lgb.Dataset(X_flat, label=y, weight=weights)
        callbacks = [lgb.log_evaluation(period=0)]
        self.lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=LGB_NUM_ROUNDS, callbacks=callbacks)
        log("[models] LightGBM treinado.", "INFO")

        # Scalers (já recebemos X_flat escalado/y escalado do features.py,
        # mas persistimos os mesmos para aplicar em produção)
        self.scaler_X = StandardScaler().fit(X_flat)
        self.scaler_y = StandardScaler().fit(y.reshape(-1, 1))

        # Prepara seq escalada para RNN (coerente com scaler_X)
        samples = X_seq.shape[0]
        seq_2d = X_seq.reshape(samples * self.seq_len, X_seq.shape[2])
        seq_2d_scaled = self.scaler_X.transform(seq_2d)
        X_seq_scaled = seq_2d_scaled.reshape(samples, self.seq_len, X_seq.shape[2])

        Xr_tr, Xr_val = X_seq_scaled[:split], X_seq_scaled[split:]
        yr_tr, yr_val = y_tr, y_val

        # Treina RNN
        self.rnn_model = _build_rnn(self.seq_len, X_seq.shape[2], units=RNN_UNITS)
        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        self.rnn_model.fit(Xr_tr, yr_tr, validation_data=(Xr_val, yr_val),
                           epochs=RNN_EPOCHS, batch_size=RNN_BATCH, callbacks=[es], verbose=0)
        log("[models] RNN treinada.", "INFO")

        self.save_artifacts()

    # ------------------------------
    # Avalia e define pesos do ensemble
    # ------------------------------
    def evaluate_and_set_weights(self, X_flat, X_seq, y):
        if self.lgb_model is None and self.rnn_model is None:
            log("[models] Sem modelos para avaliar.", "WARNING")
            return

        n = X_flat.shape[0]
        split = max(int(n * 0.85), 1)
        X_val = X_flat[split:]
        y_val = y[split:]
        seq_val = X_seq[split:]

        lgb_preds = self.lgb_model.predict(X_val, num_iteration=self.lgb_model.best_iteration) if self.lgb_model is not None else np.zeros_like(y_val)
        rnn_preds = self.rnn_model.predict(seq_val, verbose=0).flatten() if self.rnn_model is not None else np.zeros_like(y_val)

        # desnormaliza p/ medir acerto "exato" (inteiros)
        try:
            y_val_orig = self.scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
            lgb_orig = self.scaler_y.inverse_transform(lgb_preds.reshape(-1, 1)).flatten()
            rnn_orig = self.scaler_y.inverse_transform(rnn_preds.reshape(-1, 1)).flatten()
        except Exception:
            y_val_orig, lgb_orig, rnn_orig = y_val, lgb_preds, rnn_preds

        def acc_exact(y_true, preds):
            y_i = np.round(y_true).astype(int)
            p_i = np.round(preds).astype(int)
            return float((p_i == y_i).sum()) / max(len(y_i), 1)

        acc_lgb = acc_exact(y_val_orig, lgb_orig)
        acc_rnn = acc_exact(y_val_orig, rnn_orig)
        total = acc_lgb + acc_rnn + 1e-9
        self.weights = (acc_lgb / total, acc_rnn / total)
        log(f"[models] Pesos do ensemble: LGB={self.weights[0]:.3f}, RNN={self.weights[1]:.3f}", "INFO")
        self.save_artifacts()

    # ------------------------------
    # Predição do próximo registro (1 linha)
    # ------------------------------
    def predict_next(self, df_model: pd.DataFrame):
        if df_model is None or df_model.empty:
            raise ValueError("df_model vazio em predict_next().")
        if self.feature_cols is None:
            raise ValueError("feature_cols não carregado no modelo.")

        # próxima data = próximo dia útil (evita domingo)
        last_date = pd.to_datetime(df_model[DATE_COL]).max()
        next_date = last_date + pd.Timedelta(days=1)
        while next_date.weekday() == 6:  # domingo
            next_date += pd.Timedelta(days=1)

        # garantir que df_model tenha as mesmíssimas columns na MESMA ordem:
        df_model = ensure_features_exist(df_model, self.feature_cols)
        last_rows = df_model.tail(self.seq_len)
        if len(last_rows) < self.seq_len:
            raise ValueError("Dados insuficientes para seq_len na predição.")

        X_next_seq = last_rows[self.feature_cols].values
        X_next_flat = last_rows[self.feature_cols].iloc[-1].values.reshape(1, -1)

        # aplica scaler_X (foi fitado em X_flat escalado)
        X_next_seq_2d = X_next_seq.reshape(self.seq_len, -1)
        X_next_seq_2d_scaled = self.scaler_X.transform(X_next_seq_2d)
        X_next_seq_scaled = X_next_seq_2d_scaled.reshape(1, self.seq_len, X_next_seq.shape[1])
        X_next_flat_scaled = self.scaler_X.transform(X_next_flat)

        # previsões
        lgb_pred = self.lgb_model.predict(X_next_flat_scaled, num_iteration=self.lgb_model.best_iteration) if self.lgb_model is not None else np.array([0.0])
        rnn_pred = self.rnn_model.predict(X_next_seq_scaled, verbose=0).flatten() if self.rnn_model is not None else np.array([0.0])

        # mistura
        w_lgb, w_rnn = self.weights
        pred_scaled = w_lgb * lgb_pred + w_rnn * rnn_pred

        # volta à escala original
        try:
            pred_orig = self.scaler_y.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()[0]
        except Exception:
            pred_orig = float(pred_scaled[0])

        # inteiro final arredondado
        pred_int = int(np.round(pred_orig))

        # proxy de "confiança": distância do valor arredondado + pesos
        # (quanto mais perto do inteiro, maior o score; normalizamos pra 0..1)
        frac_dist = float(abs(pred_orig - pred_int))
        score_proxy = float(np.clip(1.0 - frac_dist, 0.0, 1.0)) * float(0.5 + 0.5 * (w_lgb + w_rnn) / 2.0)

        return pd.Timestamp(next_date), int(pred_int), float(score_proxy)
