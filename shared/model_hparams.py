"""
Shared AMF-BiGRU architecture hyperparameters.

All cases should import these values so the network structure is guaranteed
to stay identical across scenarios.
"""

# --- Model architecture ---
SEQ_LEN = 7
# Input feature dimensions are derived per-case from sensor selection:
#   DISP_FEATURES = len(DISP_COLS)
#   ACC_FEATURES = len(ACC_COLS)
BIGRU_HIDDEN = 32
FC1_DIM = 64
FC2_DIM = 32
OUTPUT_DIM = 4
DROPOUT = 0.2
