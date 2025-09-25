import torch
import os
from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "malebin"
TRAIN_PATH = os.path.join(DATA_DIR , "train")
TEST_PATH  = os.path.join(DATA_DIR , "test")
VAL_PATH   = os.path.join(DATA_DIR , "val")


BATCH_SIZE = 32
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")