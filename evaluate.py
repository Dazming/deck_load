import os

import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from data_loader import prepare_data
from model import AMFBiGRU


def compute_rpe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative Percentage Error (paper eq. 13), per-target-variable."""
    return np.linalg.norm(y_true - y_pred) / np.linalg.norm(y_true) * 100


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R^2 (paper eq. 14)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return 1 - ss_res / ss_tot


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, target_scaler = prepare_data()

    model = AMFBiGRU().to(device)
    ckpt_path = os.path.join(config.SAVE_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_disp, x_acc, y in test_loader:
            x_disp, x_acc = x_disp.to(device), x_acc.to(device)
            pred = model(x_disp, x_acc)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Inverse transform back to original scale
    preds_orig = target_scaler.inverse_transform(preds)
    targets_orig = target_scaler.inverse_transform(targets)

    # Per-variable metrics
    names = config.TARGET_COLS
    print("\n" + "=" * 70)
    print(f"{'Variable':<22s} {'RPE (%)':>10s} {'R2':>12s}")
    print("-" * 70)
    for i, name in enumerate(names):
        rpe = compute_rpe(targets_orig[:, i], preds_orig[:, i])
        r2 = compute_r2(targets_orig[:, i], preds_orig[:, i])
        print(f"{name:<22s} {rpe:10.4f} {r2:12.6f}")
    print("=" * 70)

    # Time axis (sliding window starts at index seq_len-1)
    dt = 0.001
    n_samples = len(targets_orig)
    times = np.arange(n_samples) * dt + config.SEQ_LEN * dt

    # ── Style ─────────────────────────────────────────────────────────────
    C_TRUE = "#00d4ff"
    C_PRED = "#ff6b6b"
    C_FILL_T = "#00d4ff"
    C_FILL_P = "#ff6b6b"

    plt.rcParams.update({
        "font.size": 11,
        "axes.facecolor": "#0d1117",
        "figure.facecolor": "#0d1117",
        "text.color": "#dfe6e9",
        "axes.labelcolor": "#dfe6e9",
        "xtick.color": "#636e72",
        "ytick.color": "#636e72",
        "axes.edgecolor": "#2d3436",
    })

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#0d1117")
    fig.suptitle("AMF-BiGRU  Test Set Evaluation  (w = 45 kN, v = 40 m/s)",
                 fontsize=16, fontweight="bold", color="white", y=0.97)

    titles = [
        "Front Axle Weight", "Rear Axle Weight",
        "Front Wheel Position", "Rear Wheel Position",
    ]
    ylabels = ["Weight (N)", "Weight (N)", "Position (m)", "Position (m)"]

    for idx, (ax, title, ylabel) in enumerate(zip(axes.flat, titles, ylabels)):
        ax.plot(times, targets_orig[:, idx],
                color=C_TRUE, linewidth=1.2, label="True", zorder=3)
        ax.plot(times, preds_orig[:, idx],
                color=C_PRED, linewidth=1.2, linestyle="--", label="Predicted", zorder=4)

        # Shaded error band
        ax.fill_between(
            times, targets_orig[:, idx], preds_orig[:, idx],
            color=C_PRED, alpha=0.08, zorder=2,
        )

        rpe = compute_rpe(targets_orig[:, idx], preds_orig[:, idx])
        r2 = compute_r2(targets_orig[:, idx], preds_orig[:, idx])
        ax.set_title(title, fontsize=12, pad=8, color="white")

        # Metrics badge
        badge = f"RPE = {rpe:.4f}%    R² = {r2:.6f}"
        ax.text(
            0.98, 0.92, badge, transform=ax.transAxes,
            fontsize=9, color="#ffeaa7", fontweight="bold",
            ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1e272e", ec="#57606f", alpha=0.85),
        )

        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.12, color="#636e72")
        ax.legend(fontsize=9, loc="upper left", framealpha=0.3, edgecolor="#555")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    fig_path = os.path.join(config.SAVE_DIR, "test_results.png")
    plt.savefig(fig_path, dpi=180)
    print(f"\nFigure saved to {fig_path}")
    plt.show()


if __name__ == "__main__":
    evaluate()
