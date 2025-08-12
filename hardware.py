# hardware.py
"""
Functions for hardware configuration (GPU/TF memory growth, limits).
Detects GPUs, attempts to configure TF memory growth for the RTX 3060, and limits memory according to the configuration.
"""

import tensorflow as tf
import psutil
import config as cfg
from io_utils import log

# ==============================
# Check RAM memory
# ==============================
def check_ram():
    try:
        mem = psutil.virtual_memory()
        used_gb = (mem.total - mem.available) / 1024**3
        log(f"RAM used: {used_gb:.2f} GB (limit: {cfg.RAM_LIMIT_GB} GB)", "DEBUG")
        if used_gb > cfg.RAM_LIMIT_GB:
            log(f"RAM used ({used_gb:.2f} GB) exceeds the limit ({cfg.RAM_LIMIT_GB} GB).", "WARNING")
    except Exception as e:
        log(f"An error occurred while trying to check RAM memory usage.: {e}", "ERROR")

# ==============================
# Setup Hardware
# ==============================
def setup_hardware():
    # GPU config
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            log("No GPUs detected by TensorFlow.", "WARNING")
            return False
        gpu = gpus[0]
        log(f"GPU detected: {gpu}", "INFO")

        # GPU memory growth
        try:
            tf.config.experimental.set_memory_growth(gpu, cfg.GPU_MEMORY_GROWTH)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=cfg.GPU_MEMORY_LIMIT_MB)]
            )
            log(f"Configurated TF: memory_growth=True; GPU Memory Limit = {round(cfg.GPU_MEMORY_LIMIT_MB / 1024)} GB", "INFO")
        except Exception as e:
            log(f"Failed to set memory growth / virtual config: {e}", "WARNING")
        return True
    except Exception as e:
        log(f"Error configuring hardware: {e}", "ERROR")
        return False
