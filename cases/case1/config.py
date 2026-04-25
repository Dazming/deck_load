import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared import model_hparams as M
from shared import sensor_config as S
from shared import training_hparams as T

DATA_DIR = os.path.join(ROOT_DIR, "dataset", "different_weight")
SAVE_DIR = os.path.join(ROOT_DIR, "checkpoints", "case1")

BEST_MODEL_NAME = "best_model_case1.pth"
EVAL_FIG_NAME = "test_results_case1.png"
PRED_VIDEO_NAME = "prediction_realtime_case1.mp4"

# --- Data ---
SENSOR_NODES = S.get_sensor_nodes("case1")
DISP_COLS = S.build_disp_cols(SENSOR_NODES)
ACC_COLS = S.build_acc_cols(SENSOR_NODES)
TARGET_COLS = ["front_axle_wt", "rear_axle_wt", "front_wheel_pos", "rear_wheel_pos"]

TRAIN_FILES = [
    os.path.join(DATA_DIR, "train", f"w{w}_v40_labeled.csv")
    for w in [40, 42, 44, 46, 48, 50]
]
VAL_FILES = [os.path.join(DATA_DIR, "val", "w38_v40_labeled.csv")]
TEST_FILES = [os.path.join(DATA_DIR, "test", "w45_v40_labeled.csv")]

# --- Model (shared across all cases) ---
SEQ_LEN = M.SEQ_LEN
DISP_FEATURES = len(DISP_COLS)
ACC_FEATURES = len(ACC_COLS)
BIGRU_HIDDEN = M.BIGRU_HIDDEN
FC1_DIM = M.FC1_DIM
FC2_DIM = M.FC2_DIM
OUTPUT_DIM = M.OUTPUT_DIM
DROPOUT = M.DROPOUT

# --- Training ---
LR = T.LR
BATCH_SIZE = T.BATCH_SIZE
MAX_EPOCHS = T.MAX_EPOCHS
EARLY_STOP_PATIENCE = T.EARLY_STOP_PATIENCE
LR_SCHEDULER_PATIENCE = T.LR_SCHEDULER_PATIENCE
LR_SCHEDULER_FACTOR = T.LR_SCHEDULER_FACTOR
LR_MIN = T.LR_MIN
GRAD_CLIP = T.GRAD_CLIP
SEED = T.SEED

