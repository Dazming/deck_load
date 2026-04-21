import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "dataset", "different_weight")
SAVE_DIR = os.path.join(ROOT_DIR, "checkpoints", "case1")

BEST_MODEL_NAME = "best_model_case1.pth"
EVAL_FIG_NAME = "test_results_case1.png"
PRED_VIDEO_NAME = "prediction_realtime_case1.mp4"

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
SEQ_LEN = 7
DISP_FEATURES = 2
ACC_FEATURES = 2
BIGRU_HIDDEN = 32
FC1_DIM = 64
FC2_DIM = 32
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

