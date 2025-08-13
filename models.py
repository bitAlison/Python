"""
Model wrapper: AutoModels
- LightGBM (Optuna) + RNN (GRU)
- save/load artifacts (EXPORT_DIR)
- ensemble e forecast
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from config import *
from io_utils import log

def _ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

# ==============================
# Build RNN
# ==============================
def _build_rnn(seq_len, n_feats, units=128):
    try:
        model = Sequential()
        model.add(GRU(units, input_shape=(seq_len, n_feats)))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model
    except Exception as e:
        log(f"Failed to build and configure RNN. {e}", "ERROR")

class AutoModels:
    def __init__(self, export_dir=None, use_gpu=True, random_state=42):
        self.export_dir = Path(export_dir or EXPORT_DIR)
        _ensure_dir(self.export_dir)
        self.use_gpu = use_gpu and bool(tf.config.list_physical_devices('GPU'))
        self.random_state = int(random_state)
        self.lgb_model = None
        self.rnn_model = None
        self.scaler_X = None
        self.scaler_y = None
        self.feature_cols = None
        self.seq_len = SEQ_LEN
        self.weights = (0.5, 0.5)

        # artifact paths
        self._p_lgb = self.export_dir / LGB_FILENAME
        self._p_rnn = self.export_dir / RNN_FILENAME
        self._p_scaler_x = self.export_dir / SCALER_FILENAME
        self._p_scaler_y = self.export_dir / SCALER_Y_FILENAME
        self._p_meta = self.export_dir / METADATA_JSON

    # ---------- helpers ----------
    def _align_feature_frame(self, df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
        """Garante que df possua TODAS as colunas em feat_cols (cria faltantes=0) e devolve reordenado."""
        df2 = df.copy()
        for c in feat_cols:
            if c not in df2.columns:
                df2[c] = 0
        return df2[feat_cols]

    # ==============================
    # Save artifacts
    # ==============================
    def save_artifacts(self):
        try:
            if self.lgb_model is not None:
                self.lgb_model.save_model(str(self._p_lgb))
            if self.rnn_model is not None:
                # evita salvar otimizador para reduzir incompatibilidades
                try:
                    self.rnn_model.save(str(self._p_rnn), save_format='keras')
                except TypeError:
                    # fallback para versões antigas
                    self.rnn_model.save(str(self._p_rnn))
            if self.scaler_X is not None:
                joblib.dump(self.scaler_X, str(self._p_scaler_x))
            if self.scaler_y is not None:
                joblib.dump(self.scaler_y, str(self._p_scaler_y))
            meta = {
                'seq_len': int(self.seq_len),
                'feature_cols': list(self.feature_cols) if self.feature_cols is not None else None,
                'weights': [float(self.weights[0]), float(self.weights[1])],
                'use_gpu': bool(self.use_gpu)
            }
            with open(self._p_meta, 'w', encoding=FILE_ENCODING) as f:
                json.dump(meta, f, indent=2)
            log("Artifacts saved.", "INFO")
        except Exception as e:
            log(f"Failed to save artifacts: {e}", "ERROR")

    # ==============================
    # Load artifacts
    # ==============================
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
                meta = json.load(open(self._p_meta, 'r', encoding=FILE_ENCODING))
                self.seq_len = meta.get('seq_len', self.seq_len)
                self.feature_cols = meta.get('feature_cols', self.feature_cols)
                self.weights = tuple(meta.get('weights', self.weights))
            log("Uploaded artifacts (if any).", "INFO")
        except Exception as e:
            log(f"Failed to load artifacts: {e}", "ERROR")

    # ==============================
    # Optuna settings
    # ==============================
    def _optuna_obj(self, trial, X_tr, y_tr, X_val, y_val):
        try:
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'verbosity': -1,
                'num_threads': NUM_THREADS or 4,
                'num_leaves': trial.suggest_int('num_leaves', 16, 256),
                'learning_rate': trial.suggest_float('learning_rate', 1e-3, 3e-1, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 3, 100),
            }
            dtr = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
            # sem early_stopping porque sua versão do lgb não aceita por callback? ok:
            callbacks = [lgb.log_evaluation(period=0)]
            booster = lgb.train(params, dtr, valid_sets=[dval], callbacks=callbacks)
            preds = booster.predict(X_val, num_iteration=booster.best_iteration)
            rmse = mean_squared_error(y_val, preds)
            return rmse
        except Exception as e:
            log(f"Failed to update Optuna settings: {e}", "ERROR")
            return None

    # ==============================
    # Fit or Load model
    # ==============================
    def fit_or_load(self, X_flat, X_seq, y, feature_cols, dates_list=None, df_prev=None):
        """
        Se existirem artefatos completos, carrega; senão treina ambos os modelos.
        Agora salvamos e reutilizamos a lista exata de feature_cols.
        """
        self.load_artifacts()
        if (
            self.lgb_model is not None and
            self.rnn_model is not None and
            self.scaler_X   is not None and
            self.scaler_y   is not None and
            self.feature_cols is not None
        ):
            log("Existing artifacts detected — skipping training.", "INFO")
            return

        # fixar o schema de features desta sessão/treino
        self.feature_cols = list(feature_cols)

        n = X_flat.shape[0]
        split = int(n * 0.85)
        X_tr, X_val = X_flat[:split], X_flat[split:]
        y_tr, y_val = y[:split], y[split:]
        seq_tr, seq_val = X_seq[:split], X_seq[split:]

        # sample weights baseados em df_prev
        weights = np.ones(n, dtype=float)
        if df_prev is not None and not df_prev.empty and dates_list is not None:
            date_to_idx = {pd.Timestamp(d).normalize(): i for i,d in enumerate(dates_list)}
            for _, r in df_prev.iterrows():
                if str(r.get(NEXT_TARGET_COL_RESULT)).strip().lower() == NEXT_STATUS_ERROR:
                    try:
                        d = pd.Timestamp(r[DATE_COL]).normalize()
                        if d in date_to_idx:
                            weights[date_to_idx[d]] *= 3.0
                    except Exception:
                        continue
            weights = weights / max(weights.mean(), 1e-9)

        # Optuna tuning
        try:
            log("Initializing Optuna (LightGBM)...", "INFO")
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda t: self._optuna_obj(t, X_tr, y_tr, X_val, y_val), n_trials=OPTUNA_TRIALS, n_jobs=1)
            best = study.best_params
            lgb_params = {
                'objective':'regression','metric':'rmse','boosting_type':'gbdt','verbosity':-1,
                'num_threads': NUM_THREADS or 4,
                'num_leaves': best['num_leaves'],
                'learning_rate': best['learning_rate'],
                'feature_fraction': best['feature_fraction'],
                'bagging_fraction': best['bagging_fraction'],
                'bagging_freq': best['bagging_freq'],
                'min_data_in_leaf': best['min_data_in_leaf']
            }
            log(f"Optuna done: {best}", "INFO")
        except Exception as e:
            log(f"Optuna fail ({e}) — using default parameters.", "WARNING")
            # se não tiver no seu config, troque por um dicionário mínimo:
            lgb_params = {
                'objective':'regression','metric':'rmse','boosting_type':'gbdt','verbosity':-1,
                'num_threads': NUM_THREADS or 4,
                'num_leaves': 64, 'learning_rate': 0.05,
                'feature_fraction': 0.9, 'bagging_fraction': 0.8,
                'bagging_freq': 1, 'min_data_in_leaf': 20
            }

        # train LGB
        try:
            dtrain = lgb.Dataset(X_flat, label=y, weight=weights)
            callbacks = [lgb.log_evaluation(period=0)]
            self.lgb_model = lgb.train(lgb_params, dtrain, num_boost_round=LGB_NUM_ROUNDS, callbacks=callbacks)
            log("LightGBM trained.", "INFO")
        except Exception as e:
            log(f"Training failure LGB: {e}", "ERROR")
            raise

        # scalers
        self.scaler_X = StandardScaler().fit(X_flat)
        self.scaler_y = StandardScaler().fit(y.reshape(-1,1))

        # scale seq para RNN
        samples = X_seq.shape[0]
        seq_2d = X_seq.reshape(samples * self.seq_len, X_seq.shape[2])
        seq_2d_scaled = self.scaler_X.transform(seq_2d)
        X_seq_scaled = seq_2d_scaled.reshape(samples, self.seq_len, X_seq.shape[2])

        Xr_tr, Xr_val = X_seq_scaled[:split], X_seq_scaled[split:]
        yr_tr, yr_val = y_tr, y_val

        # train RNN
        try:
            log("Training RNN (GRU)...", "INFO")
            self.rnn_model = _build_rnn(self.seq_len, X_seq.shape[2], units=RNN_UNITS)
            es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
            self.rnn_model.fit(Xr_tr, yr_tr, validation_data=(Xr_val, yr_val),
                               epochs=RNN_EPOCHS, batch_size=RNN_BATCH, callbacks=[es], verbose=0)
            log("RNN trained.", "INFO")
        except Exception as e:
            log(f"Training failure RNN: {e}", "WARNING")
            try:
                self.rnn_model = _build_rnn(self.seq_len, X_seq.shape[2], units=max(32, RNN_UNITS//4))
                self.rnn_model.fit(X_seq_scaled, y, epochs=5, batch_size=RNN_BATCH, verbose=0)
                log("fallback trained RNN.", "INFO")
            except Exception as e2:
                log(f"RNN fallback failed: {e2}", "ERROR")
                self.rnn_model = None

        self.save_artifacts()

    # ==============================
    # Evaluate and set weights
    # ==============================
    def evaluate_and_set_weights(self, X_flat, X_seq, y):
        try:
            if self.lgb_model is None and self.rnn_model is None:
                log("No models to evaluate.", "WARNING")
                return
            n = X_flat.shape[0]
            split = int(n * 0.85)
            X_val = X_flat[split:]
            y_val = y[split:]
            seq_val = X_seq[split:]

            lgb_preds = self.lgb_model.predict(X_val, num_iteration=self.lgb_model.best_iteration) if self.lgb_model is not None else np.zeros_like(y_val)
            rnn_preds = self.rnn_model.predict(seq_val).flatten() if self.rnn_model is not None else np.zeros_like(y_val)

            # inverse y
            if self.scaler_y is not None:
                try:
                    y_val_orig = self.scaler_y.inverse_transform(y_val.reshape(-1,1)).flatten()
                    lgb_orig = self.scaler_y.inverse_transform(lgb_preds.reshape(-1,1)).flatten()
                    rnn_orig = self.scaler_y.inverse_transform(rnn_preds.reshape(-1,1)).flatten()
                except Exception:
                    y_val_orig, lgb_orig, rnn_orig = y_val, lgb_preds, rnn_preds
            else:
                y_val_orig, lgb_orig, rnn_orig = y, lgb_preds, rnn_preds

            def acc_exact(y_true, preds):
                y_i = np.round(y_true).astype(int)
                p_i = np.round(preds).astype(int)
                return float((p_i == y_i).sum()) / max(len(y_i),1)

            acc_lgb = acc_exact(y_val_orig, lgb_orig)
            acc_rnn = acc_exact(y_val_orig, rnn_orig)
            total = acc_lgb + acc_rnn + 1e-9
            w_lgb = acc_lgb / total
            w_rnn = acc_rnn / total
            self.weights = (w_lgb, w_rnn)
            log(f"Defined weights: LGB={w_lgb:.3f}, RNN={w_rnn:.3f}", "INFO")
            self.save_artifacts()
        except Exception as e:
            log(f"Error to evaluate and set weights.{e}", "ERROR")

    # ==============================
    # Predict the next value
    # ==============================
    def predict_next(self, df_model, feature_cols=None, scaler_X=None, scaler_y=None):
        try:
            # usar SEMPRE o schema salvo no treino
            feat_cols = list(feature_cols) if feature_cols is not None else self.feature_cols
            if not feat_cols:
                raise ValueError("Feature schema (feature_cols) not found. Train the models first.")

            # próxima data (pula domingo)
            last_date = pd.to_datetime(df_model[DATE_COL]).max()
            next_date = last_date + pd.Timedelta(days=1)
            while next_date.weekday() == 6:
                next_date += pd.Timedelta(days=1)

            last_rows = df_model.tail(self.seq_len)
            if len(last_rows) < self.seq_len:
                raise ValueError("Insufficient data for seq_len.")

            # alinhar features exatamente como no treino
            last_rows_feat = self._align_feature_frame(last_rows, feat_cols)

            X_next_seq = last_rows_feat.values
            X_next_flat = last_rows_feat.iloc[-1].values.reshape(1, -1)

            # escolher scalers do modelo, se não vierem
            if scaler_X is None and self.scaler_X is not None:
                scaler_X = self.scaler_X
            if scaler_y is None and self.scaler_y is not None:
                scaler_y = self.scaler_y

            if scaler_X is not None:
                X_next_seq_2d = X_next_seq.reshape(self.seq_len, -1)
                X_next_seq_2d_scaled = scaler_X.transform(X_next_seq_2d)
                X_next_seq_scaled = X_next_seq_2d_scaled.reshape(1, self.seq_len, X_next_seq.shape[1])
                X_next_flat_scaled = scaler_X.transform(X_next_flat)
            else:
                X_next_seq_scaled = X_next_seq.reshape(1, self.seq_len, X_next_seq.shape[1])
                X_next_flat_scaled = X_next_flat

            # prever
            lgb_pred = (self.lgb_model.predict(X_next_flat_scaled, num_iteration=self.lgb_model.best_iteration)
                        if self.lgb_model is not None else np.array([0.0]))
            rnn_pred = (self.rnn_model.predict(X_next_seq_scaled).flatten()
                        if self.rnn_model is not None else np.array([0.0]))

            w_lgb, w_rnn = self.weights
            pred_scaled = w_lgb * lgb_pred + w_rnn * rnn_pred

            if scaler_y is not None:
                pred_orig = scaler_y.inverse_transform(np.array(pred_scaled).reshape(-1,1)).flatten()[0]
            else:
                pred_orig = float(pred_scaled[0])

            pred_int = int(np.round(pred_orig))
            score_proxy = float((w_lgb + w_rnn) / 2.0)
            return pd.Timestamp(next_date), pred_int, score_proxy
        except Exception as e:
            log(f"Fail to predict the next value.{e}", "ERROR")
            return None, None, None
