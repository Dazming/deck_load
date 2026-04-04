"""
Generate a video simulating real-time AMF-BiGRU prediction.
Video duration = 10x vehicle running time.

Usage:
    conda run -n test python generate_video.py

Requires ffmpeg:
    conda install -n test ffmpeg
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

import config
from data_loader import prepare_data
from model import AMFBiGRU

# ── Visual style ──────────────────────────────────────────────────────────
C_TRUE = "#00d4ff"
C_PRED = "#ff6b6b"
C_ACCENT = "#00b894"
C_TIME = "#ffeaa7"
C_SENSOR = "#a29bfe"
C_LANE_LABEL = "#b2bec3"

DECK_LENGTH = 40.0  # m
DECK_WIDTH = 8.0
SLOWDOWN = 10
FPS = 30

N1_POS = 5.0
N7_POS = 35.0


def generate_video():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data & model ──────────────────────────────────────────────────────
    _, _, test_loader, target_scaler = prepare_data()

    model = AMFBiGRU().to(device)
    ckpt = os.path.join(config.SAVE_DIR, "best_model.pth")
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for xd, xa, y in test_loader:
            pred = model(xd.to(device), xa.to(device))
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds = target_scaler.inverse_transform(np.concatenate(all_preds))
    targets = target_scaler.inverse_transform(np.concatenate(all_targets))
    n = len(preds)

    dt = 0.001
    times = np.arange(n) * dt + config.SEQ_LEN * dt
    data_dur = times[-1]
    video_dur = data_dur * SLOWDOWN
    total_frames = int(video_dur * FPS)

    print(f"Samples: {n}  |  Data duration: {data_dur:.3f}s  |  "
          f"Video: {video_dur:.1f}s ({total_frames} frames @ {FPS}fps)")

    # target order: front_axle_wt, rear_axle_wt, front_wheel_pos, rear_wheel_pos
    true_fw, true_rw = targets[:, 0], targets[:, 1]
    true_fp, true_rp = targets[:, 2], targets[:, 3]
    pred_fw, pred_rw = preds[:, 0], preds[:, 1]
    pred_fp, pred_rp = preds[:, 2], preds[:, 3]

    # ── Figure ────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.size": 10,
        "axes.facecolor": "#0d1117",
        "figure.facecolor": "#0d1117",
        "text.color": "#dfe6e9",
        "axes.labelcolor": "#dfe6e9",
        "xtick.color": "#636e72",
        "ytick.color": "#636e72",
        "axes.edgecolor": "#2d3436",
    })

    fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
    gs = GridSpec(
        3, 2, figure=fig, height_ratios=[2.8, 1, 1],
        hspace=0.45, wspace=0.28,
        left=0.06, right=0.97, top=0.89, bottom=0.07,
    )

    ax_deck = fig.add_subplot(gs[0, :])
    ax_fw_plot = fig.add_subplot(gs[1, 0])
    ax_rw_plot = fig.add_subplot(gs[1, 1])
    ax_fp_plot = fig.add_subplot(gs[2, 0])
    ax_rp_plot = fig.add_subplot(gs[2, 1])

    fig.suptitle(
        "AMF-BiGRU  Real-Time Moving Load Prediction",
        fontsize=17, fontweight="bold", color="white", y=0.96,
    )

    # ── Deck view ─────────────────────────────────────────────────────────
    ax_deck.set_xlim(-3, DECK_LENGTH + 3)
    ax_deck.set_ylim(-DECK_WIDTH * 0.75, DECK_WIDTH * 0.75)
    ax_deck.set_aspect("equal")
    ax_deck.set_xlabel("Position along deck (m)", fontsize=10)
    ax_deck.tick_params(left=False, labelleft=False)

    deck = FancyBboxPatch(
        (0, -DECK_WIDTH / 2), DECK_LENGTH, DECK_WIDTH,
        boxstyle="round,pad=0.2", fc="#1b2631", ec="#566573", lw=2,
    )
    ax_deck.add_patch(deck)

    for x in range(0, int(DECK_LENGTH) + 1, 5):
        ax_deck.plot([x, x], [-DECK_WIDTH / 2, DECK_WIDTH / 2],
                     color="#2c3e50", lw=0.5, ls="--")

    # Sensor markers
    for sx, label in [(N1_POS, "N1"), (N7_POS, "N7")]:
        ax_deck.plot(sx, -DECK_WIDTH / 2 + 0.4, "^", color=C_SENSOR, ms=8, zorder=6)
        ax_deck.text(sx, -DECK_WIDTH / 2 - 0.6, label,
                     ha="center", fontsize=8, color=C_SENSOR)

    # Direction arrow
    ax_deck.annotate(
        "", xy=(DECK_LENGTH - 1, DECK_WIDTH / 2 + 1.2),
        xytext=(1, DECK_WIDTH / 2 + 1.2),
        arrowprops=dict(arrowstyle="->", color="#95a5a6", lw=1.5),
    )
    ax_deck.text(DECK_LENGTH / 2, DECK_WIDTH / 2 + 1.9,
                 "Direction of travel →", ha="center", fontsize=9, color="#95a5a6")

    # Lane labels
    ax_deck.text(-2.5, 1.2, "True", ha="center", fontsize=9,
                 color=C_TRUE, fontweight="bold")
    ax_deck.text(-2.5, -1.2, "Pred", ha="center", fontsize=9,
                 color=C_PRED, fontweight="bold")

    # Vehicle artists
    true_front, = ax_deck.plot([], [], "o", color=C_TRUE, ms=13, zorder=7)
    true_rear, = ax_deck.plot([], [], "s", color=C_TRUE, ms=11, zorder=7)
    pred_front, = ax_deck.plot([], [], "o", color=C_PRED, ms=13, zorder=7, alpha=0.9)
    pred_rear, = ax_deck.plot([], [], "s", color=C_PRED, ms=11, zorder=7, alpha=0.9)
    true_body, = ax_deck.plot([], [], "-", color=C_TRUE, lw=3, alpha=0.5, zorder=4)
    pred_body, = ax_deck.plot([], [], "--", color=C_PRED, lw=3, alpha=0.5, zorder=4)

    # Weight text on deck
    wt_text_true = ax_deck.text(0, 2.5, "", fontsize=9, color=C_TRUE, ha="center")
    wt_text_pred = ax_deck.text(0, -2.5, "", fontsize=9, color=C_PRED, ha="center")

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color=C_TRUE, label="True (● front  ■ rear)",
                   ms=8, lw=0),
        plt.Line2D([0], [0], marker="o", color=C_PRED, label="Predicted (● front  ■ rear)",
                   ms=8, lw=0),
    ]
    ax_deck.legend(handles=legend_elements, loc="upper right", fontsize=9,
                   framealpha=0.3, edgecolor="#555")

    # ── Time-series plots ─────────────────────────────────────────────────
    ts_axes = [ax_fw_plot, ax_rw_plot, ax_fp_plot, ax_rp_plot]
    ts_titles = ["Front Axle Weight", "Rear Axle Weight",
                 "Front Wheel Position", "Rear Wheel Position"]
    ts_ylabels = ["Weight (N)", "Weight (N)", "Position (m)", "Position (m)"]
    ts_true = [true_fw, true_rw, true_fp, true_rp]
    ts_pred = [pred_fw, pred_rw, pred_fp, pred_rp]

    lines_t, lines_p, v_lines = [], [], []

    for ax, title, ylabel, yt, yp in zip(ts_axes, ts_titles, ts_ylabels,
                                          ts_true, ts_pred):
        ax.set_xlim(times[0], times[-1])
        lo = min(yt.min(), yp.min())
        hi = max(yt.max(), yp.max())
        pad = max((hi - lo) * 0.12, 1.0)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(title, fontsize=10, pad=4)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.12)

        lt, = ax.plot([], [], color=C_TRUE, lw=1.2, label="True")
        lp, = ax.plot([], [], color=C_PRED, lw=1.2, ls="--", label="Predicted")
        vl = ax.axvline(times[0], color=C_TIME, lw=0.7, alpha=0.5)
        lines_t.append(lt)
        lines_p.append(lp)
        v_lines.append(vl)

    ax_fw_plot.legend(fontsize=8, loc="upper right", framealpha=0.3)
    ax_fp_plot.set_xlabel("Time (s)", fontsize=9)
    ax_rp_plot.set_xlabel("Time (s)", fontsize=9)

    # ── Bottom bar: time + progress ───────────────────────────────────────
    time_text = fig.text(0.5, 0.018, "", ha="center", fontsize=13,
                         color=C_TIME, fontweight="bold")

    ax_prog = fig.add_axes([0.06, 0.005, 0.88, 0.008])
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.axis("off")
    ax_prog.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#2d3436", ec="none"))
    prog_bar = ax_prog.add_patch(plt.Rectangle((0, 0), 0, 1, fc=C_ACCENT, ec="none"))

    # ── Animation ─────────────────────────────────────────────────────────
    def _pos_or_hide(pos_val, weight_val):
        """Return position if on deck (weight > 0), else NaN to hide marker."""
        return pos_val if weight_val > 500 else np.nan

    def update(frame):
        progress = frame / max(total_frames - 1, 1)
        idx = min(int(progress * (n - 1)), n - 1)
        t_now = times[idx]

        # Deck markers
        fp_t = _pos_or_hide(true_fp[idx], true_fw[idx])
        rp_t = _pos_or_hide(true_rp[idx], true_rw[idx])
        fp_p = _pos_or_hide(pred_fp[idx], pred_fw[idx])
        rp_p = _pos_or_hide(pred_rp[idx], pred_rw[idx])

        y_true, y_pred = 1.2, -1.2
        true_front.set_data([fp_t], [y_true])
        true_rear.set_data([rp_t], [y_true])
        pred_front.set_data([fp_p], [y_pred])
        pred_rear.set_data([rp_p], [y_pred])

        if not (np.isnan(fp_t) or np.isnan(rp_t)):
            true_body.set_data([rp_t, fp_t], [y_true, y_true])
        elif not np.isnan(fp_t):
            true_body.set_data([0, fp_t], [y_true, y_true])
        elif not np.isnan(rp_t):
            true_body.set_data([rp_t, DECK_LENGTH], [y_true, y_true])
        else:
            true_body.set_data([], [])

        if not (np.isnan(fp_p) or np.isnan(rp_p)):
            pred_body.set_data([rp_p, fp_p], [y_pred, y_pred])
        elif not np.isnan(fp_p):
            pred_body.set_data([0, fp_p], [y_pred, y_pred])
        elif not np.isnan(rp_p):
            pred_body.set_data([rp_p, DECK_LENGTH], [y_pred, y_pred])
        else:
            pred_body.set_data([], [])

        # Weight annotations
        mid_t = np.nanmean([fp_t, rp_t]) if not (np.isnan(fp_t) and np.isnan(rp_t)) else -5
        mid_p = np.nanmean([fp_p, rp_p]) if not (np.isnan(fp_p) and np.isnan(rp_p)) else -5
        wt_text_true.set_position((mid_t, 2.6))
        wt_text_true.set_text(f"F:{true_fw[idx]/1000:.1f} kN  R:{true_rw[idx]/1000:.1f} kN")
        wt_text_pred.set_position((mid_p, -2.6))
        wt_text_pred.set_text(f"F:{pred_fw[idx]/1000:.1f} kN  R:{pred_rw[idx]/1000:.1f} kN")

        # Time-series curves (reveal up to current index)
        s = slice(0, idx + 1)
        t_s = times[s]
        for i in range(4):
            lines_t[i].set_data(t_s, ts_true[i][s])
            lines_p[i].set_data(t_s, ts_pred[i][s])
            v_lines[i].set_xdata([t_now])

        # Progress
        time_text.set_text(f"Time: {t_now:.3f} s / {data_dur:.3f} s")
        prog_bar.set_width(progress)

        if (frame + 1) % max(total_frames // 20, 1) == 0 or frame == 0:
            print(f"  [{frame+1:>{len(str(total_frames))}}/{total_frames}]  "
                  f"{progress*100:5.1f}%  t={t_now:.3f}s")

        return []

    print("\nRendering video ...")
    anim = animation.FuncAnimation(fig, update, frames=total_frames, blit=False)

    out_path = os.path.join(config.SAVE_DIR, "prediction_realtime.mp4")
    writer = animation.FFMpegWriter(fps=FPS, bitrate=3000,
                                     extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out_path, writer=writer)
    plt.close(fig)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nDone!  Video saved to: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    generate_video()
