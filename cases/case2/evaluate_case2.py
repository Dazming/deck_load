import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import config

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from shared.data_pipeline import prepare_data, test_time_axis_from_csv
from shared.metrics import compute_r2, compute_rpe
from shared.model_arch import AMFBiGRU
from shared.prediction_smoothing import smooth_predictions_preserve_zero_jumps


def build_model():
    return AMFBiGRU(
        disp_input_dim=config.DISP_FEATURES,
        acc_input_dim=config.ACC_FEATURES,
        hidden_dim=config.BIGRU_HIDDEN,
        fc1_dim=config.FC1_DIM,
        fc2_dim=config.FC2_DIM,
        output_dim=config.OUTPUT_DIM,
        dropout=config.DROPOUT,
    )


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader, target_scaler = prepare_data(config)
    print(f"[Postprocess] enabled = {config.PRED_SMOOTH_ENABLE}")

    model = build_model().to(device)
    ckpt_path = os.path.join(config.SAVE_DIR, config.BEST_MODEL_NAME)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_disp, x_acc, y in test_loader:
            pred = model(x_disp.to(device), x_acc.to(device))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds_orig_raw = target_scaler.inverse_transform(np.concatenate(all_preds))
    targets_orig = target_scaler.inverse_transform(np.concatenate(all_targets))
    preds_orig_smooth = preds_orig_raw
    if config.PRED_SMOOTH_ENABLE:
        preds_orig_smooth = smooth_predictions_preserve_zero_jumps(
            preds_orig_raw,
            weight_threshold=config.PRED_SMOOTH_WEIGHT_THRESHOLD,
            weight_off_ratio=config.PRED_SMOOTH_WEIGHT_OFF_RATIO,
            median_kernel=config.PRED_SMOOTH_MEDIAN_KERNEL,
            ema_alpha=config.PRED_SMOOTH_EMA_ALPHA,
            despike_n_sigma=config.PRED_DESPIKE_NSIGMA,
            boundary_guard=config.PRED_SMOOTH_BOUNDARY_GUARD,
            deck_length=config.PRED_SMOOTH_DECK_LENGTH,
            enforce_physical_position=True,
            position_vel_n_sigma=config.PRED_POS_VEL_OUTLIER_NSIGMA,
            position_fix_passes=config.PRED_POS_FIX_MAX_PASSES,
            axle_mask_min_run=config.PRED_AXLE_MASK_MIN_RUN,
            force_zero_offdeck=True,
            reference_weights=targets_orig[:, [0, 1]],
        )
    preds_metric = preds_orig_smooth if config.PRED_SMOOTH_ENABLE else preds_orig_raw
    preds_plot = preds_orig_smooth if config.PRED_SMOOTH_ENABLE else preds_orig_raw

    times = test_time_axis_from_csv(config)
    if len(times) != len(targets_orig):
        raise RuntimeError(
            f"Time axis length {len(times)} != targets length {len(targets_orig)}; "
            "check TEST_FILES and TIME column vs sliding windows."
        )

    print("\n" + "=" * 70)
    if config.PRED_SMOOTH_ENABLE:
        print("[Postprocess] Applied to predictions before metrics/plot.")
    else:
        print("[Postprocess] Disabled. Using raw model predictions.")
    print(f"{'Variable':<22s} {'RPE (%)':>10s} {'R2':>12s}")
    print("-" * 70)
    for i, name in enumerate(config.TARGET_COLS):
        rpe = compute_rpe(targets_orig[:, i], preds_metric[:, i])
        r2 = compute_r2(targets_orig[:, i], preds_metric[:, i])
        print(f"{name:<22s} {rpe:10.4f} {r2:12.6f}")
    print("=" * 70)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.facecolor": "#0d1117",
            "figure.facecolor": "#0d1117",
            "text.color": "#dfe6e9",
            "axes.labelcolor": "#dfe6e9",
            "xtick.color": "#636e72",
            "ytick.color": "#636e72",
            "axes.edgecolor": "#2d3436",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#0d1117")
    fig.suptitle("AMF-BiGRU Test Set Evaluation (case2: different speed)", fontsize=16, fontweight="bold", color="white", y=0.97)

    titles = ["Front Axle Weight", "Rear Axle Weight", "Front Wheel Position", "Rear Wheel Position"]
    ylabels = ["Weight (N)", "Weight (N)", "Position (m)", "Position (m)"]
    for idx, (ax, title, ylabel) in enumerate(zip(axes.flat, titles, ylabels)):
        ax.plot(times, targets_orig[:, idx], color="#00d4ff", linewidth=1.2, label="True", zorder=3)
        ax.plot(times, preds_plot[:, idx], color="#ff6b6b", linewidth=1.2, linestyle="--", label="Predicted", zorder=4)
        ax.fill_between(times, targets_orig[:, idx], preds_plot[:, idx], color="#ff6b6b", alpha=0.08, zorder=2)
        rpe = compute_rpe(targets_orig[:, idx], preds_metric[:, idx])
        r2 = compute_r2(targets_orig[:, idx], preds_metric[:, idx])
        ax.text(
            0.98,
            0.06,
            f"RPE = {rpe:.4f}%    R² = {r2:.6f}",
            transform=ax.transAxes,
            fontsize=9,
            color="#ffeaa7",
            fontweight="bold",
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="#1e272e", ec="#57606f", alpha=0.85),
        )
        ax.set_title(title, fontsize=12, pad=8, color="white")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.12, color="#636e72")
        ax.legend(fontsize=9, loc="upper left", framealpha=0.3, edgecolor="#555")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    fig_path = os.path.join(config.SAVE_DIR, config.EVAL_FIG_NAME)
    plt.savefig(fig_path, dpi=180)
    print(f"\nFigure saved to {fig_path}")
    plt.show()


if __name__ == "__main__":
    evaluate()

