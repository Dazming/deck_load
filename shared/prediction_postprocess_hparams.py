"""
Shared prediction postprocess hyperparameters.

Both case1 and case2 should import these values so behavior is consistent.
Flip ENABLE to globally turn anomaly repair on/off.
"""

# Global switch (both cases)
ENABLE = True

# Generic robust-despike parameters
WEIGHT_THRESHOLD = 500.0
MEDIAN_KERNEL = 9
EMA_ALPHA = 1.0
DESPIKE_NSIGMA = 2.5
BOUNDARY_GUARD = 4

# Physical projection parameters
DECK_LENGTH = 40.0
ENFORCE_PHYSICAL_POSITION = True
POS_VEL_OUTLIER_NSIGMA = 2.2
POS_FIX_MAX_PASSES = 3
AXLE_MASK_MIN_RUN = 5
FORCE_ZERO_OFFDECK = True

# Evaluation behavior
EVAL_USE_SMOOTH_FOR_METRICS = True
EVAL_PLOT_SMOOTHED = True

