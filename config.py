import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset", "different_weight")
SAVE_DIR = os.path.join(BASE_DIR, "checkpoints")

# --- Data ---
DISP_COLS = ["N1_UZ", "N7_UZ"]
ACC_COLS = ["N1_AZ", "N7_AZ"]
TARGET_COLS = ["front_axle_wt", "rear_axle_wt", "front_wheel_pos", "rear_wheel_pos"]

TRAIN_FILES = [
    os.path.join(DATA_DIR, "train", f"w{w}_v40_labeled.csv")
    for w in [40, 42, 44, 46, 48, 50]
]
VAL_FILES = [os.path.join(DATA_DIR, "val", "w38_v40_labeled.csv")]
TEST_FILES = [os.path.join(DATA_DIR, "test", "w45_v40_labeled.csv")]

# --- Model ---
SEQ_LEN = 7          # sliding window size (s)
DISP_FEATURES = 2    # displacement input dim per time step
ACC_FEATURES = 2     # acceleration input dim per time step
BIGRU_HIDDEN = 32    # BiGRU hidden size (each direction)
FC1_DIM = 64
FC2_DIM = 32
FUSION_DIM = 32
OUTPUT_DIM = 4
DROPOUT = 0.2

# --- Training ---
LR = 0.005
BATCH_SIZE = 64
MAX_EPOCHS = 3000
EARLY_STOP_PATIENCE = 300
LR_SCHEDULER_PATIENCE = 60
LR_SCHEDULER_FACTOR = 0.5
LR_MIN = 1e-6
GRAD_CLIP = 1.0
SEED = 42
