"""
Shared training hyperparameters.

All cases should import these values so optimizer/scheduler/training strategy
stay consistent across scenarios.
"""

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
