import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def load_csv_files(file_list):
    """Load and concatenate multiple CSV files, keeping per-file boundaries."""
    frames = []
    for f in file_list:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        frames.append(df)
    return frames


class StandardScaler:
    """Z-score scaler fitted on training data only."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)
        self.std[self.std == 0] = 1.0
        return self

    def transform(self, data: np.ndarray):
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray):
        return data * self.std + self.mean


def test_time_axis_from_csv(cfg, time_col: str = "TIME"):
    """
    Time stamps for each test sample, aligned with sliding-window targets.

    For window ending at row index i (i >= seq_len - 1), the target matches CSV row i,
    so we use TIME[i] from the same file. Order matches ``prepare_data`` test_loader
    concatenation over ``cfg.TEST_FILES``.
    """
    times = []
    for path in cfg.TEST_FILES:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if time_col not in df.columns:
            raise ValueError(
                f"Missing {time_col!r} in {path}; cannot build time axis from CSV."
            )
        tcol = df[time_col].to_numpy(dtype=np.float64)
        n = len(tcol)
        for i in range(cfg.SEQ_LEN - 1, n):
            times.append(tcol[i])
    return np.asarray(times, dtype=np.float64)


def prediction_time_axis_from_dataframe(
    df: pd.DataFrame,
    seq_len: int,
    time_col: str = "TIME",
    fallback_dt: float = 0.001,
) -> np.ndarray:
    """
    Time stamp for each sliding-window prediction from a single ``DataFrame``.

    Matches ``build_sliding_windows``: prediction ``k`` corresponds to target row
    index ``seq_len - 1 + k``. Uses ``TIME`` at that row when present; otherwise a
    uniform grid: ``k * fallback_dt + seq_len * fallback_dt`` (same as the legacy
    synthetic axis).
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    n = len(df)
    n_out = n - seq_len + 1
    if n_out <= 0:
        return np.zeros(0, dtype=np.float64)

    if time_col in df.columns:
        t = df[time_col].to_numpy(dtype=np.float64)
        return np.array([t[i] for i in range(seq_len - 1, n)], dtype=np.float64)

    return (
        np.arange(n_out, dtype=np.float64) * fallback_dt + seq_len * fallback_dt
    ).astype(np.float64)


def build_sliding_windows(disp: np.ndarray, acc: np.ndarray, targets: np.ndarray, seq_len: int):
    """
    Create sliding-window samples from a single condition's time series.
    Window of length `seq_len` with stride 1.
    """
    n = len(disp)
    X_disp, X_acc, Y = [], [], []
    for i in range(seq_len - 1, n):
        X_disp.append(disp[i - seq_len + 1: i + 1])
        X_acc.append(acc[i - seq_len + 1: i + 1])
        Y.append(targets[i])
    return np.array(X_disp), np.array(X_acc), np.array(Y)


class MovingLoadDataset(Dataset):
    def __init__(self, disp: np.ndarray, acc: np.ndarray, targets: np.ndarray):
        self.disp = torch.tensor(disp, dtype=torch.float32)
        self.acc = torch.tensor(acc, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.disp[idx], self.acc[idx], self.targets[idx]


def prepare_data(cfg):
    """
    Full pipeline: load CSVs -> fit scalers on train -> sliding window -> Datasets.
    Returns DataLoaders and fitted target scaler for inverse transform.
    """
    train_frames = load_csv_files(cfg.TRAIN_FILES)
    val_frames = load_csv_files(cfg.VAL_FILES)
    test_frames = load_csv_files(cfg.TEST_FILES)

    train_all = pd.concat(train_frames, ignore_index=True)
    disp_scaler = StandardScaler().fit(train_all[cfg.DISP_COLS].values)
    acc_scaler = StandardScaler().fit(train_all[cfg.ACC_COLS].values)
    target_scaler = StandardScaler().fit(train_all[cfg.TARGET_COLS].values)

    def _process_frames(frames):
        all_disp, all_acc, all_y = [], [], []
        for df in frames:
            disp = disp_scaler.transform(df[cfg.DISP_COLS].values)
            acc = acc_scaler.transform(df[cfg.ACC_COLS].values)
            tgt = target_scaler.transform(df[cfg.TARGET_COLS].values)

            xd, xa, y = build_sliding_windows(disp, acc, tgt, cfg.SEQ_LEN)
            all_disp.append(xd)
            all_acc.append(xa)
            all_y.append(y)
        return (
            np.concatenate(all_disp),
            np.concatenate(all_acc),
            np.concatenate(all_y),
        )

    train_disp, train_acc, train_y = _process_frames(train_frames)
    val_disp, val_acc, val_y = _process_frames(val_frames)
    test_disp, test_acc, test_y = _process_frames(test_frames)

    train_ds = MovingLoadDataset(train_disp, train_acc, train_y)
    val_ds = MovingLoadDataset(val_disp, val_acc, val_y)
    test_ds = MovingLoadDataset(test_disp, test_acc, test_y)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, target_scaler

