"""
Dataset base directory configuration.

Edit BASE_DIR below if your dataset is stored elsewhere.
"""
from .logging_config import logger
from pathlib import Path

BASE_DIR = Path("/home/stella/computer_vision") #Path.home()  # users might need to change this, depending on where they place the dataset
DATASET_NAME = "PMOF"
DATA_BASE_DIR = BASE_DIR / DATASET_NAME

if not DATA_BASE_DIR.exists():
    logger.warning(f"DATA_BASE_DIR does not exist: {DATA_BASE_DIR}\nPlease update BASE_DIR in utils/databasedir_config.py to point to your dataset location.")