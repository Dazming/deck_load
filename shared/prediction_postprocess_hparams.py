"""
Shared prediction postprocess hyperparameters.

Both case1 and case2 should import these numeric defaults.
Enable/disable switch is defined independently in each case config.
"""

# Generic robust-despike parameters
WEIGHT_THRESHOLD = 500.0
WEIGHT_OFF_RATIO = 0.5
MEDIAN_KERNEL = 9
EMA_ALPHA = 1.0
DESPIKE_NSIGMA = 2.5
BOUNDARY_GUARD = 4

# Physical projection parameters
DECK_LENGTH = 40.0
POS_VEL_OUTLIER_NSIGMA = 2.2
POS_FIX_MAX_PASSES = 3
AXLE_MASK_MIN_RUN = 5

