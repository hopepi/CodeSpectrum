from enum import Enum
import torch
import os
from pathlib import Path

ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "malebin"
TRAIN_PATH = os.path.join(DATA_DIR , "train")
TEST_PATH  = os.path.join(DATA_DIR , "test")
VAL_PATH   = os.path.join(DATA_DIR , "val")

class Arch(Enum):
    RESNET18 = "resnet18"
    MOBILENET_V3_SMALL = "mobilenet_v3_small"
    EFFICIENTNET_B0 = "efficientnet_b0"
    DENSENET121 = "densenet121"
    VGG16 = "vgg16"
    MOBILENET_V2 = "mobilenet_v2"
    SHUFFLENET_V2_X0_5 = "shufflenet_v2_x0_5"
    MNASNET_1_0 = "mnasnet_1_0"

EPOCHS = 1
LR = 0.001
BATCH_SIZE = 32
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")