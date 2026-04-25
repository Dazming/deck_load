"""
Shared AMF-BiGRU architecture hyperparameters.

All cases should import these values so the network structure is guaranteed
to stay identical across scenarios.
"""

# --- Model architecture ---
SEQ_LEN = 7
# Input feature dimensions are now derived from selected sensor columns
# in each case config (len(DISP_COLS), len(ACC_COLS)).
DISP_FEATURES = 2
ACC_FEATURES = 2
BIGRU_HIDDEN = 32
FC1_DIM = 64
FC2_DIM = 32
OUTPUT_DIM = 4
DROPOUT = 0.2
