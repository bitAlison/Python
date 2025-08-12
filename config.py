# config.py
"""
Main configurations

"""

from pathlib import Path

# ----------------- PATHS / DIRECTORIES -----------------
# Project dirs
PROJECT_DIR = Path(r"E:\Tests\Prediction")              # Project directory
DATA_SOURCE = PROJECT_DIR / "DataSource.csv"            # Data source file (data;valor;suplementar1..14)
OUTPUT_NAME = "Forecast"                                #
OUTPUT_EXTENSION = ".csv"                               #
FORECAST_FILE = PROJECT_DIR / OUTPUT_NAME + OUTPUT_EXTENSION # Forecast file (data;valor_previsto;valor_real;percentual_previsao;resultado)
EXPORT_DIR = PROJECT_DIR / "exported_models"            # models
LOG_DIR = PROJECT_DIR / "logs"                          # logs, backups, etc.
BACKUP_DIR = PROJECT_DIR / "backups"                    # automatic backups

# Automatic creation of directories (run_predictor does ensure_dirs)
# ----------------- DATA / FEATURES -----------------
TARGET_COL = "valor"                                    # target column in the file DataSource
TARGET_COL_RANGE_MIN = 1                                # minimal interval referring to target column
TARGET_COL_RANGE_MAX = 25                               # maximum interval referring to target column

LASTEST_TARGET_COL_VALUE = "valor_real"                 # column in the prediction file corresponding of the last value of TARGET_COL
NEXT_TARGET_COL_VALUE = "valor_previsto"                # column in the prediction file corresponding to forecasting value of LASTEST_TARGET_COL_VALUE
NEXT_TARGET_COL_PERCENTUAL = "percentual_acerto"        # column in the prediction file corresponding of the percuntual of forecasting value of NEXT_TARGET_COL_VALUE
NEXT_TARGET_COL_RESULT = "resultado"                    # column in the prediction file corresponding to the percuntual of next value of NEXT_TARGET_COL_VALUE
NEXT_STATUS_SUCCESS = "acertou"                         # predicted value settlement status
NEXT_STATUS_ERROR = "errou"                             # predicted value error status
FREQUENCY_COL = "valor_freq"                            # column of frequency value

FEATURES_COLS = ['data', 'sumplementar1', 'sumplementar2', 'sumplementar3', 'sumplementar4', 'sumplementar5', 'sumplementar6', 'sumplementar7', 'sumplementar8', 'sumplementar9', 'sumplementar10', 'sumplementar11', 'sumplementar12', 'sumplementar13', 'sumplementar14']

# Supplemental: The code automatically considers Supplemental 1..Supplemental 14 when they exist for the model training and prediction values
SUPPLEMENTARY_RANGE = (1, 14)                           # interval referring to supplementary1..supplementary14
SUPPLEMENTARY_COL = "sumplementar"                      # interval referring to supplementary1..supplementary14

DATE_COL = "data"                                       # name of the date column in DataSource (must be parseable)
DATE_FORMAT_INPUT = "%Y-%m-%d"                          # internal format; the loader accepts variations when parsing
DATE_FORMAT_OUTPU = "%d/%m/%y"                          # internal format; the loader accepts variations when parsing


# Settings used to help increase training prediction accuracy
# ----------------- SYNTHETIC DATA -----------------
MIN_SAMPLE_DATE = "2010-01-01"                          # the minimum date for the auto generation sample 
MIN_RECORDS = 1000                                      # the minimum records for the sample
MAX_RECORDS = 3500                                      # the maximum records for the sample


# ----------------- TIME WINDOW / MATRICES -----------------
SEQ_LEN = 14                                            # number of steps (lags) used in the RNN / sequence
MIN_TRAIN_SAMPLES = 100                                 # If there are fewer, the pipeline can use alternative strategies

# ----------------- MODEL / HPO -----------------
RANDOM_SEED = 42
OPTUNA_TRIALS = 40                                      # number of Optuna attempts (adjustment for environment: 40 -> heavy)
RNN_UNITS = 128                                         # GRU units in the RNN
RNN_EPOCHS = 80                                         # standard epochs for RNN
RNN_BATCH = 64                                          # batch to RNN
LGB_NUM_ROUNDS = 1000                                   # maximum rounds for LightGBM (with early stopping by callbacks)

# ----------------- HARDWARE / RESOURCES -----------------
USE_GPU = True                                          # if True, will try to use GPU (TensorFlow + LightGBM when possible)
# Machine specific: RTX 3060 with 12GB. Set TF limit in MB (leave margin).
GPU_MEMORY_LIMIT_MB = 10000                             # 10 GB allocated for TF (leaves 2 GB free)
GPU_MEMORY_GROWTH = True                                # Indicates whether the GPU can expand memory usage
# RAM: your machine has 48GB, maximum limit to be used is 40GB as requested
RAM_LIMIT_GB = 40

# ----------------- PERFORMANCE / PARALLELISM -----------------
# threads used by LightGBM / other libs (psutil.cpu_count() is used at runtime)
NUM_THREADS = None                                      # None -> will be resolved dynamically (psutil.cpu_count())

# ----------------- I/O / BACKUPS -----------------
PREV_BACKUP_KEEP = 3                                    # how many backups to keep of predictions.csv file
AUTO_BACKUP = True                                      # create backup before overwriting predictions.csv file

# ----------------- MODEL SAVING-----------------
MODEL_NAME_PREFIX = "model_extreme"                    # prefix for saved files (model, scaler, metadata)
SCALER_FILENAME = "scaler_X.save"
SCALER_Y_FILENAME = "scaler_y.save"
LGB_FILENAME = "lgb_model.txt"
RNN_FILENAME = "rnn_model.keras"
METADATA_JSON = "model_metadata.json"

# ----------------- LOG / DEBUG -----------------
LOG_LEVEL = "INFO"                                      # DEBUG / INFO / WARNING / ERROR
VERBOSE = False                                         # general verbose (can be used to reduce output)

# ----------------- OTHERS -----------------
FILE_ENCODING = "utf-8"                                 # Default file encoding
CSV_SEPARATOR = ";"


# ----------------- TARGET ACCURACY SETTINGS -----------------
# Success Criterion (Target): >= 0.9 OR >= 90% (used in auto-calibration strategies as a target)
SUCCESS_THRESHOLD = 0.9

# Timeout/limits for pipeline parts (seconds) - use with caution
TRAIN_TIMEOUT_SECONDS = 300                            # None -> no timeout. Can be configured externally.

# ----------------- USEFUL COMMANDS -----------------
# pip command suggested (see requirements.txt)
PIP_INSTALL_CMD = "python -m pip install -r requirements.txt"

# ----------------- FIM -----------------
