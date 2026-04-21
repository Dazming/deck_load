import numpy as np


def compute_rpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative Percentage Error."""
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true) * 100


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - ss_res / ss_tot

