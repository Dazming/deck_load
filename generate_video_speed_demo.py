"""
Demo video: AMF-BiGRU prediction for different vehicle speed (v = 30 m/s).
Uses synthesized data (no real simulation / model needed).
Video duration = 10x vehicle running time.

Usage:
    conda run -n test python generate_video_speed_demo.py

Requires ffmpeg:
    conda install -n test ffmpeg
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

# ── Parameters ────────────────────────────────────────────────────────────
VEHICLE_SPEED = 30.0       # m/s
VEHICLE_WEIGHT = 45000.0   # N (per axle, 1:1 ratio)
AXLE_SPACING = 8.0         # m
DECK_LENGTH = 40.0         # m
DECK_WIDTH = 8.0           # m
DT = 0.001                 # s

TOTAL_TIME = (DECK_LENGTH + AXLE_SPACING) / VEHICLE_SPEED + 0.2  # extra margin
SLOWDOWN = 10
FPS = 30

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
OUTPUT_NAME = "demo_speed_30.mp4"

N1_POS = 5.0
N7_POS = 35.0

# Noise levels to mimic prediction error (~RPE 0.3%-1%)
WEIGHT_NOISE_STD = 180.0   # N
POS_NOISE_STD = 0.08       # m

# ── Visual style ──────────────────────────────────────────────────────────
C_TRUE = "#00d4ff"
C_PRED = "#ff6b6b"
C_ACCENT = "#00b894"
C_TIME = "#ffeaa7"
C_SENSOR = "#a29bfe"


def synthesize_data():
    """Generate physically plausible true values and noisy predictions."""
    n_steps = int(TOTAL_TIME / DT)
    times = np.arange(n_steps) * DT

    front_pos_raw = VEHICLE_SPEED * times
    rear_pos_raw = front_pos_raw - AXLE_SPACING

    on_deck_front = (front_pos_raw >= 0) & (front_pos_raw <= DECK_LENGTH)
    on_deck_rear = (rear_pos_raw >= 0) & (rear_pos_raw <= DECK_LENGTH)

    true_fp = np.where(on_deck_front, front_pos_raw, 0.0)
    true_rp = np.where(on_deck_rear, rear_pos_raw, 0.0)

    true_fw = np.where(on_deck_front, VEHICLE_WEIGHT, 0.0)
    true_rw = np.where(on_deck_rear, VEHICLE_WEIGHT, 0.0)

    rng = np.random.default_rng(42)

    def _smooth_noise(n, std, kernel_size=15):
        raw = rng.normal(0, std, n)
        kernel = np.ones(kernel_size) / kernel_size
        return np.convolve(raw, kernel, mode="same")

    pred_fw = true_fw + _smooth_noise(n_steps, WEIGHT_NOISE_STD) * on_deck_front
    pred_rw = true_rw + _smooth_noise(n_steps, WEIGHT_NOISE_STD) * on_deck_rear
    pred_fp = true_fp + _smooth_noise(n_steps, POS_NOISE_STD) * on_deck_front
    pred_rp = true_rp + _smooth_noise(n_steps, POS_NOISE_STD) * on_deck_rear

    # Small transient at on/off-deck edges for realism
    for arr_pred, mask in [(pred_fw, on_deck_front), (pred_rw, on_deck_rear)]:
        transitions = np.where(np.diff(mask.astype(int)))[0]
        for t_idx in transitions:
            win = slice(max(t_idx - 8, 0), min(t_idx + 12, n_steps))
            arr_pred[win] += rng.normal(0, WEIGHT_NOISE_STD * 2, len(arr_pred[win]))

    pred_fw = np.clip(pred_fw, 0, None)
    pred_rw = np.clip(pred_rw, 0, None)
    pred_fp = np.clip(pred_fp, 0, None)
    pred_rp = np.clip(pred_rp, 0, None)

    return times, true_fw, true_rw, true_fp, true_rp, pred_fw, pred_rw, pred_fp, pred_rp


def generate_video():
    times, true_fw, true_rw, true_fp, true_rp, \
        pred_fw, pred_rw, pred_fp, pred_rp = synthesize_data()

    n = len(times)
    data_dur = times[-1]
    video_dur = data_dur * SLOWDOWN
    total_frames = int(video_dur * FPS)

    print(f"Speed: {VEHICLE_SPEED} m/s  |  Samples: {n}  |  "
          f"Data: {data_dur:.3f}s  |  Video: {video_dur:.1f}s ({total_frames} frames)")

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
        f"AMF-BiGRU  Different Speed  (v = {VEHICLE_SPEED:.0f} m/s,  "
        f"w = {VEHICLE_WEIGHT/1000:.0f} kN)",
        fontsize=16, fontweight="bold", color="white", y=0.96,
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

    for sx, label in [(N1_POS, "N1"), (N7_POS, "N7")]:
        ax_deck.plot(sx, -DECK_WIDTH / 2 + 0.4, "^", color=C_SENSOR, ms=8, zorder=6)
        ax_deck.text(sx, -DECK_WIDTH / 2 - 0.6, label,
                     ha="center", fontsize=8, color=C_SENSOR)

    ax_deck.annotate(
        "", xy=(DECK_LENGTH - 1, DECK_WIDTH / 2 + 1.2),
        xytext=(1, DECK_WIDTH / 2 + 1.2),
        arrowprops=dict(arrowstyle="->", color="#95a5a6", lw=1.5),
    )
    ax_deck.text(DECK_LENGTH / 2, DECK_WIDTH / 2 + 1.9,
                 "Direction of travel →", ha="center", fontsize=9, color="#95a5a6")

    ax_deck.text(-2.5, 1.2, "True", ha="center", fontsize=9,
                 color=C_TRUE, fontweight="bold")
    ax_deck.text(-2.5, -1.2, "Pred", ha="center", fontsize=9,
                 color=C_PRED, fontweight="bold")

    true_front_dot, = ax_deck.plot([], [], "o", color=C_TRUE, ms=13, zorder=7)
    true_rear_dot, = ax_deck.plot([], [], "s", color=C_TRUE, ms=11, zorder=7)
    pred_front_dot, = ax_deck.plot([], [], "o", color=C_PRED, ms=13, zorder=7, alpha=0.9)
    pred_rear_dot, = ax_deck.plot([], [], "s", color=C_PRED, ms=11, zorder=7, alpha=0.9)
    true_body_line, = ax_deck.plot([], [], "-", color=C_TRUE, lw=3, alpha=0.5, zorder=4)
    pred_body_line, = ax_deck.plot([], [], "--", color=C_PRED, lw=3, alpha=0.5, zorder=4)

    wt_text_true = ax_deck.text(0, 2.6, "", fontsize=9, color=C_TRUE, ha="center")
    wt_text_pred = ax_deck.text(0, -2.6, "", fontsize=9, color=C_PRED, ha="center")

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

    # ── Progress bar ──────────────────────────────────────────────────────
    time_text = fig.text(0.5, 0.018, "", ha="center", fontsize=13,
                         color=C_TIME, fontweight="bold")

    ax_prog = fig.add_axes([0.06, 0.005, 0.88, 0.008])
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.axis("off")
    ax_prog.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#2d3436", ec="none"))
    prog_bar = ax_prog.add_patch(plt.Rectangle((0, 0), 0, 1, fc=C_ACCENT, ec="none"))

    # ── Animation ─────────────────────────────────────────────────────────
    def _val_or_nan(pos, weight):
        return pos if weight > 500 else np.nan

    def update(frame):
        progress = frame / max(total_frames - 1, 1)
        idx = min(int(progress * (n - 1)), n - 1)
        t_now = times[idx]

        fp_t = _val_or_nan(true_fp[idx], true_fw[idx])
        rp_t = _val_or_nan(true_rp[idx], true_rw[idx])
        fp_p = _val_or_nan(pred_fp[idx], pred_fw[idx])
        rp_p = _val_or_nan(pred_rp[idx], pred_rw[idx])

        yt, yp = 1.2, -1.2
        true_front_dot.set_data([fp_t], [yt])
        true_rear_dot.set_data([rp_t], [yt])
        pred_front_dot.set_data([fp_p], [yp])
        pred_rear_dot.set_data([rp_p], [yp])

        # Body lines
        for body, f_pos, r_pos, y_lane in [
            (true_body_line, fp_t, rp_t, yt),
            (pred_body_line, fp_p, rp_p, yp),
        ]:
            f_nan, r_nan = np.isnan(f_pos), np.isnan(r_pos)
            if not f_nan and not r_nan:
                body.set_data([r_pos, f_pos], [y_lane, y_lane])
            elif not f_nan:
                body.set_data([0, f_pos], [y_lane, y_lane])
            elif not r_nan:
                body.set_data([r_pos, DECK_LENGTH], [y_lane, y_lane])
            else:
                body.set_data([], [])

        mid_t = np.nanmean([fp_t, rp_t]) if not (np.isnan(fp_t) and np.isnan(rp_t)) else -5
        mid_p = np.nanmean([fp_p, rp_p]) if not (np.isnan(fp_p) and np.isnan(rp_p)) else -5
        wt_text_true.set_position((mid_t, 2.6))
        wt_text_true.set_text(f"F:{true_fw[idx]/1000:.1f} kN  R:{true_rw[idx]/1000:.1f} kN")
        wt_text_pred.set_position((mid_p, -2.6))
        wt_text_pred.set_text(f"F:{pred_fw[idx]/1000:.1f} kN  R:{pred_rw[idx]/1000:.1f} kN")

        s = slice(0, idx + 1)
        t_s = times[s]
        for i in range(4):
            lines_t[i].set_data(t_s, ts_true[i][s])
            lines_p[i].set_data(t_s, ts_pred[i][s])
            v_lines[i].set_xdata([t_now])

        time_text.set_text(f"Time: {t_now:.3f} s / {data_dur:.3f} s")
        prog_bar.set_width(progress)

        if (frame + 1) % max(total_frames // 20, 1) == 0 or frame == 0:
            print(f"  [{frame+1:>{len(str(total_frames))}}/{total_frames}]  "
                  f"{progress*100:5.1f}%  t={t_now:.3f}s")

        return []

    print("\nRendering video ...")
    anim = animation.FuncAnimation(fig, update, frames=total_frames, blit=False)

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, OUTPUT_NAME)
    writer = animation.FFMpegWriter(fps=FPS, bitrate=3000,
                                     extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out_path, writer=writer)
    plt.close(fig)

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\nDone!  Video saved to: {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    generate_video()
