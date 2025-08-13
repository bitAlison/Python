# config.py
"""
Main configurations

"""

from pathlib import Path

# ----------------- PATHS / DIRECTORIES -----------------
# Project dirs
PROJECT_DIR: Path = Path(r"D:\Git")                           # Project directory
DATA_SOURCE: Path  = PROJECT_DIR / "Modelo.csv"                # Data source file (data;valor;suplementar1..14)
OUTPUT_NAME: str = r"Forecast"                                       #
OUTPUT_EXTENSION: str = r".csv"                                      #
FORECAST_FILE: Path  = PROJECT_DIR / f"{OUTPUT_NAME}{OUTPUT_EXTENSION}" # Forecast file (data;valor_previsto;valor_real;percentual_previsao;resultado)
EXPORT_DIR: Path  = PROJECT_DIR / "exported_models"            # models
LOG_DIR: Path  = PROJECT_DIR / "logs"                          # logs, backups, etc.
BACKUP_DIR: Path  = PROJECT_DIR / "backups"                    # automatic backups

# Automatic creation of directories (run_predictor does ensure_dirs)
# ----------------- DATA / FEATURES -----------------
TARGET_COL: str = r"valor"                                    # target column in the file DataSource
TARGET_COL_RANGE_MIN: int = 1                                 # minimal interval referring to target column
TARGET_COL_RANGE_MAX: int = 25                                # maximum interval referring to target column

LATEST_TARGET_COL_VALUE: str = r"valor_real"                  # column in the prediction file corresponding of the last value of TARGET_COL
NEXT_TARGET_COL_VALUE: str = r"valor_previsto"                # column in the prediction file corresponding to forecasting value of LATEST_TARGET_COL_VALUE
NEXT_TARGET_COL_PERCENTUAL: str = r"percentual_acerto"        # column in the prediction file corresponding of the percentual of forecasting value of NEXT_TARGET_COL_VALUE
NEXT_TARGET_COL_RESULT: str = r"resultado"                    # column in the prediction file corresponding to the percentual of next value of NEXT_TARGET_COL_VALUE
NEXT_STATUS_SUCCESS: str = r"acertou"                         # predicted value settlement status
NEXT_STATUS_ERROR: str = r"errou"                             # predicted value error status
FREQUENCY_COL: str = r"valor_freq"                            # column of frequency value

# Supplemental: The code automatically considers Supplemental 1..Supplemental 14 when they exist for the model training and prediction values
SUPPLEMENTARY_RANGE = (1, 15)                      # interval referring to supplementary1..supplementary14
SUPPLEMENTARY_COL: str = r"suplementar"                       # interval referring to supplementary1..supplementary14

FEATURES_COLS = [
    "year", "month", "day", "dayofweek", "is_weekend", "valor_freq"
]

DATE_COL: str = r"data"                                       # name of the date column in DataSource (must be parseable)
DATE_FORMAT_INPUT: str = r"%Y-%m-%d"                          # internal format; the loader accepts variations when parsing
DATE_FORMAT_OUTPUT: str = r"%d/%m/%y"                         # internal format; the loader accepts variations when parsing


# Settings used to help increase training prediction accuracy
# ----------------- SYNTHETIC DATA -----------------
MIN_SAMPLE_DATE: str = r"2010-01-01"                          # the minimum date for the auto generation sample 
MIN_RECORDS: int = 1000                                       # the minimum records for the sample
MAX_RECORDS: int = 3500                                       # the maximum records for the sample


# ----------------- TIME WINDOW / MATRICES -----------------
SEQ_LEN: int =  14                                            # number of steps (lags) used in the RNN / sequence
MIN_TRAIN_SAMPLES: int = 100                                  # If there are fewer, the pipeline can use alternative strategies

# ----------------- MODEL / HPO -----------------
RANDOM_SEED: int = 42
OPTUNA_TRIALS: int = 40                                       # number of Optuna attempts (adjustment for environment: 40 -> heavy)
RNN_UNITS: int = 128                                          # GRU units in the RNN
RNN_EPOCHS: int = 80                                          # standard epochs for RNN
RNN_BATCH: int = 64                                           # batch to RNN
LGB_NUM_ROUNDS: int = 1000                                    # maximum rounds for LightGBM (with early stopping by callbacks)

# ----------------- HARDWARE / RESOURCES -----------------
USE_GPU = True                                                # if True, will try to use GPU (TensorFlow + LightGBM when possible)
# Machine specific: RTX 3060 with 12GB. Set TF limit in MB (leave margin).
GPU_MEMORY_LIMIT_MB: int = 10000                              # 10 GB allocated for TF (leaves 2 GB free)
GPU_MEMORY_GROWTH = True                                      # Indicates whether the GPU can expand memory usage
# RAM: your machine has 48GB, maximum limit to be used is 40GB as requested
RAM_LIMIT_GB: int = 40

# ----------------- PERFORMANCE / PARALLELISM -----------------
# threads used by LightGBM / other libs (psutil.cpu_count() is used at runtime)
NUM_THREADS = None                                            # None -> will be resolved dynamically (psutil.cpu_count())
PARAMS_LGBM_DEFAULT = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "num_leaves": 64,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "seed": 42
}


# ----------------- I/O / BACKUPS -----------------
PREV_BACKUP_KEEP: int = 3                                    # how many backups to keep of predictions.csv file
AUTO_BACKUP = True                                           # create backup before overwriting predictions.csv file

# ----------------- MODEL SAVING-----------------
MODEL_NAME_PREFIX: str = r"model_extreme"                    # prefix for saved files (model, scaler, metadata)
SCALER_FILENAME: str = r"scaler_X.save"
SCALER_Y_FILENAME: str = r"scaler_y.save"
LGB_FILENAME: str = r"lgb_model.txt"
RNN_FILENAME: str = r"rnn_model.keras"
METADATA_JSON: str = r"model_metadata.json"

# ----------------- LOG / DEBUG -----------------
LOG_LEVEL: str = r"INFO"                                     # DEBUG / INFO / WARNING / ERROR
VERBOSE = False                                              # general verbose (can be used to reduce output)

# ----------------- OTHERS -----------------
FILE_ENCODING: str = r"utf-8"                                # Default file encoding
CSV_SEPARATOR: str = r";"


# ----------------- TARGET ACCURACY SETTINGS -----------------
# Success Criterion (Target): >= 0.9 OR >= 90% (used in auto-calibration strategies as a target)
SUCCESS_THRESHOLD = 0.9

# Timeout/limits for pipeline parts (seconds) - use with caution
TRAIN_TIMEOUT_SECONDS: int = 300                            # None -> no timeout. Can be configured externally.

# ----------------- USEFUL COMMANDS -----------------
# pip command suggested (see requirements.txt)
PIP_INSTALL_CMD: str = r"python -m pip install -r requirements.txt"

# ----------------- FIM -----------------
