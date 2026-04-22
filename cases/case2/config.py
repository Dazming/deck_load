import os
from shared import model_hparams as M

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "dataset", "different_speed")
SAVE_DIR = os.path.join(ROOT_DIR, "checkpoints", "case2")

BEST_MODEL_NAME = "best_model_case2.pth"
EVAL_FIG_NAME = "test_results_case2.png"
PRED_VIDEO_NAME = "prediction_realtime_case2.mp4"

# --- Data ---
DISP_COLS = ["N1_UZ", "N7_UZ"]
ACC_COLS = ["N1_AZ", "N7_AZ"]
TARGET_COLS = ["front_axle_wt", "rear_axle_wt", "front_wheel_pos", "rear_wheel_pos"]

TRAIN_FILES = [
    os.path.join(DATA_DIR, "train", f"w45_v{v}_labeled.csv")
    for v in [10, 15, 20, 25, 35, 40]
]
VAL_FILES = [os.path.join(DATA_DIR, "val", "w45_v45_labeled.csv")]
TEST_FILES = [os.path.join(DATA_DIR, "test", "w45_v30_labeled.csv")]

# --- Model (shared across all cases) ---
SEQ_LEN = M.SEQ_LEN
DISP_FEATURES = M.DISP_FEATURES
ACC_FEATURES = M.ACC_FEATURES
BIGRU_HIDDEN = M.BIGRU_HIDDEN
FC1_DIM = M.FC1_DIM
FC2_DIM = M.FC2_DIM
OUTPUT_DIM = M.OUTPUT_DIM
DROPOUT = M.DROPOUT

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

