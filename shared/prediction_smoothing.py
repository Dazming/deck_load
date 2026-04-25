import numpy as np


def _suppress_short_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    """
    Remove short True-runs in a boolean mask (debounce on/off flicker).
    """
    if min_run <= 1 or len(mask) == 0:
        return mask.copy()
    out = mask.copy()
    i = 0
    n = len(mask)
    while i < n:
        if not out[i]:
            i += 1
            continue
        j = i + 1
        while j < n and out[j]:
            j += 1
        if (j - i) < min_run:
            out[i:j] = False
        i = j
    return out


def _median_filter_1d(x: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1 or len(x) == 0:
        return x.copy()
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    if len(x) < k:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        out[i] = np.median(xp[i:i + k])
    return out.astype(x.dtype, copy=False)


def _interp_nan_runs_1d(x: np.ndarray) -> np.ndarray:
    """
    Fill NaN runs by linear interpolation between nearest valid neighbors.
    Keeps endpoints by nearest-value extension.
    """
    if len(x) == 0:
        return x.copy()
    y = x.astype(np.float64).copy()
    idx = np.arange(len(y))
    valid = ~np.isnan(y)
    if valid.all():
        return y.astype(x.dtype, copy=False)
    if not valid.any():
        return np.zeros_like(x)
    y[~valid] = np.interp(idx[~valid], idx[valid], y[valid])
    return y.astype(x.dtype, copy=False)


def _repair_position_velocity_outliers_1d(
    p: np.ndarray,
    n_sigma: float = 3.0,
    max_passes: int = 2,
) -> np.ndarray:
    """
    Repair local position kinks by detecting outlier first-differences (velocity).
    """
    if len(p) < 5:
        return p.copy()
    out = p.astype(np.float64).copy()
    for _ in range(max(1, int(max_passes))):
        d = np.diff(out)
        med = np.median(d)
        mad = np.median(np.abs(d - med))
        sigma = 1.4826 * mad
        if sigma <= 1e-12:
            break
        bad_d_idx = np.where(np.abs(d - med) > n_sigma * sigma)[0]
        if len(bad_d_idx) == 0:
            break

        # Difference d[i] is between point i and i+1; repair point i+1.
        bad_p_idx = bad_d_idx + 1
        # Keep segment endpoints fixed to anchor interpolation.
        bad_p_idx = bad_p_idx[(bad_p_idx > 0) & (bad_p_idx < len(out) - 1)]
        if len(bad_p_idx) == 0:
            break

        temp = out.copy()
        temp[bad_p_idx] = np.nan
        out = _interp_nan_runs_1d(temp).astype(np.float64)

    return out.astype(p.dtype, copy=False)


def _mark_outliers_hampel_1d(x: np.ndarray, window_size: int, n_sigma: float) -> np.ndarray:
    """
    Return boolean mask of outliers (without replacing values).
    """
    if len(x) == 0:
        return np.zeros(0, dtype=bool)
    w = int(window_size)
    if w % 2 == 0:
        w += 1
    if w < 3:
        w = 3
    if len(x) < w:
        return np.zeros(len(x), dtype=bool)

    half = w // 2
    y = x.astype(np.float64)
    outlier = np.zeros(len(y), dtype=bool)
    for i in range(half, len(y) - half):
        local = y[i - half:i + half + 1]
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        sigma = 1.4826 * mad
        if sigma <= 1e-12:
            continue
        if np.abs(y[i] - med) > n_sigma * sigma:
            outlier[i] = True
    return outlier


def _zero_phase_ema_1d(x: np.ndarray, alpha: float) -> np.ndarray:
    if len(x) == 0 or not (0 < alpha <= 1):
        return x.copy()

    def _ema_forward(v):
        y = np.empty_like(v, dtype=np.float64)
        y[0] = v[0]
        for i in range(1, len(v)):
            y[i] = alpha * v[i] + (1.0 - alpha) * y[i - 1]
        return y

    fwd = _ema_forward(x.astype(np.float64))
    bwd = _ema_forward(fwd[::-1])[::-1]
    return bwd.astype(x.dtype, copy=False)


def smooth_predictions_preserve_zero_jumps(
    preds: np.ndarray,
    weight_cols=(0, 1),
    position_cols=(2, 3),
    weight_threshold: float = 500.0,
    deck_length: float = 40.0,
    median_kernel: int = 5,
    ema_alpha: float = 1.0,
    despike_n_sigma: float = 3.0,
    boundary_guard: int = 3,
    enforce_physical_position: bool = True,
    position_vel_n_sigma: float = 3.0,
    position_fix_passes: int = 2,
    axle_mask_min_run: int = 5,
    force_zero_offdeck: bool = True,
) -> np.ndarray:
    """
    Smooth prediction spikes while preserving physical on-deck/off-deck jumps.

    Strategy:
    1) Use predicted axle weights to build an on-deck mask.
    2) Split sequence at mask change points (prevents smoothing across true 0-jumps).
    3) Inside each segment, detect outliers via Hampel/MAD, then repair by
       interpolation; this can handle consecutive outlier runs.
    4) Protect boundary points near true on/off-deck jumps (boundary_guard).
    5) Optional physical projection for wheel positions:
       - off-deck -> position = 0
       - on-deck -> clamp to [0, deck_length] and enforce monotonic non-decreasing.
       - repair local velocity outliers (position kinks) inside on-deck segments.
       - optional hard rule: off-deck axle weight = 0 (remove start/end residual spikes).
    """
    if preds.ndim != 2 or len(preds) == 0:
        return preds.copy()

    out = preds.copy()
    on_deck = np.zeros(len(preds), dtype=bool)
    for c in weight_cols:
        on_deck |= preds[:, c] > weight_threshold
    on_deck = _suppress_short_runs(on_deck, axle_mask_min_run)

    change_idx = np.where(np.diff(on_deck.astype(np.int8)) != 0)[0] + 1
    starts = np.concatenate(([0], change_idx))
    ends = np.concatenate((change_idx, [len(preds)]))

    for s, e in zip(starts, ends):
        seg = out[s:e, :]
        if len(seg) < 3:
            continue
        for col in range(seg.shape[1]):
            y = seg[:, col]

            # Detect outliers first (supports contiguous runs).
            outlier_mask = _mark_outliers_hampel_1d(y, median_kernel, despike_n_sigma)

            # Protect true jump boundaries: do not alter first/last boundary_guard points.
            g = max(0, int(boundary_guard))
            if g > 0:
                outlier_mask[:g] = False
                outlier_mask[-g:] = False

            if outlier_mask.any():
                y_fix = y.astype(np.float64).copy()
                y_fix[outlier_mask] = np.nan
                y = _interp_nan_runs_1d(y_fix)

            if 0 < ema_alpha < 1.0:
                y = _zero_phase_ema_1d(y, ema_alpha)
            seg[:, col] = y
        out[s:e, :] = seg

    if enforce_physical_position:
        # Per axle: project position with corresponding axle weight.
        # Convention: front axle uses cols (weight_cols[0], position_cols[0]),
        # rear axle uses (weight_cols[1], position_cols[1]).
        for wc, pc in zip(weight_cols, position_cols):
            w = out[:, wc]
            p = out[:, pc].copy()

            on_mask = w > weight_threshold
            on_mask = _suppress_short_runs(on_mask, axle_mask_min_run)
            p[~on_mask] = 0.0
            if force_zero_offdeck:
                w[~on_mask] = 0.0
                out[:, wc] = w

            # Clamp deck bounds for on-deck range.
            p[on_mask] = np.clip(p[on_mask], 0.0, deck_length)

            # Enforce monotonic progression within each continuous on-deck segment.
            change = np.where(np.diff(on_mask.astype(np.int8)) != 0)[0] + 1
            seg_starts = np.concatenate(([0], change))
            seg_ends = np.concatenate((change, [len(p)]))
            for s, e in zip(seg_starts, seg_ends):
                if not on_mask[s]:
                    continue
                p[s:e] = _repair_position_velocity_outliers_1d(
                    p[s:e],
                    n_sigma=position_vel_n_sigma,
                    max_passes=position_fix_passes,
                )
                p[s:e] = np.clip(p[s:e], 0.0, deck_length)
                p[s:e] = np.maximum.accumulate(p[s:e])

            out[:, pc] = p
    return out

