"""
Flask API server for AMF-BiGRU moving load identification.

Usage:
    conda run -n test python api_server.py
"""

import io
import os

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from cases.case1 import config as case_config
from shared.data_pipeline import (
    StandardScaler,
    build_sliding_windows,
    load_csv_files,
    prediction_time_axis_from_dataframe,
)
from shared.model_arch import AMFBiGRU

app = Flask(__name__)
CORS(app)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_disp_scaler = None
_acc_scaler = None
_target_scaler = None


@app.errorhandler(Exception)
def handle_exception(e):
    """Return JSON instead of HTML for unhandled server errors."""
    code = getattr(e, "code", 500)
    msg = str(e) if str(e) else "Internal server error"
    return jsonify({"error": msg}), code


def _fit_scalers():
    global _disp_scaler, _acc_scaler, _target_scaler
    if _disp_scaler is not None:
        return
    train_frames = load_csv_files(case_config.TRAIN_FILES)
    train_all = pd.concat(train_frames, ignore_index=True)
    _disp_scaler = StandardScaler().fit(train_all[case_config.DISP_COLS].values)
    _acc_scaler = StandardScaler().fit(train_all[case_config.ACC_COLS].values)
    _target_scaler = StandardScaler().fit(train_all[case_config.TARGET_COLS].values)


def _load_model():
    global _model
    if _model is not None:
        return _model
    ckpt = os.path.join(case_config.SAVE_DIR, case_config.BEST_MODEL_NAME)
    if not os.path.exists(ckpt):
        return None
    _model = AMFBiGRU(
        disp_input_dim=case_config.DISP_FEATURES,
        acc_input_dim=case_config.ACC_FEATURES,
        hidden_dim=case_config.BIGRU_HIDDEN,
        fc1_dim=case_config.FC1_DIM,
        fc2_dim=case_config.FC2_DIM,
        output_dim=case_config.OUTPUT_DIM,
        dropout=case_config.DROPOUT,
    ).to(device)
    _model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    _model.eval()
    return _model


def _run_inference(csv_path: str):
    _fit_scalers()
    model = _load_model()
    if model is None:
        return None

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    disp = _disp_scaler.transform(df[case_config.DISP_COLS].values)
    acc = _acc_scaler.transform(df[case_config.ACC_COLS].values)
    tgt = _target_scaler.transform(df[case_config.TARGET_COLS].values)

    xd, xa, y = build_sliding_windows(disp, acc, tgt, case_config.SEQ_LEN)
    xd_t = torch.tensor(xd, dtype=torch.float32).to(device)
    xa_t = torch.tensor(xa, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(xd_t, xa_t).cpu().numpy()

    preds_orig = _target_scaler.inverse_transform(preds)
    targets_orig = _target_scaler.inverse_transform(y)

    n = len(targets_orig)
    times_arr = prediction_time_axis_from_dataframe(df, case_config.SEQ_LEN)
    if len(times_arr) != n:
        raise ValueError("Time axis length mismatch; check TIME column and CSV row count.")
    times = times_arr.tolist()

    def _rpe(yt, yp):
        return float(np.linalg.norm(yt - yp) / np.linalg.norm(yt) * 100)

    def _r2(yt, yp):
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        return 1.0 if ss_tot == 0 else float(1 - ss_res / ss_tot)

    # Downsample for frontend (every 10th point to keep payload small)
    step = max(1, n // 200)
    idx = list(range(0, n, step))

    metrics = {}
    series = {}
    for i, col in enumerate(case_config.TARGET_COLS):
        metrics[col] = {
            "rpe": round(_rpe(targets_orig[:, i], preds_orig[:, i]), 4),
            "r2": round(_r2(targets_orig[:, i], preds_orig[:, i]), 6),
        }
        series[col] = {
            "true": [float(targets_orig[j, i]) for j in idx],
            "pred": [float(preds_orig[j, i]) for j in idx],
        }

    return {
        "times": [times[j] for j in idx],
        "metrics": metrics,
        "series": series,
    }


def _discover_conditions():
    conditions = []
    for split in ["train", "val", "test"]:
        folder = os.path.join(case_config.DATA_DIR, split)
        if not os.path.isdir(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            if not fname.endswith(".csv"):
                continue
            parts = fname.replace("_labeled.csv", "").split("_")
            w = parts[0].replace("w", "")
            v = parts[1].replace("v", "") if len(parts) > 1 else "40"
            conditions.append({
                "weight": int(w),
                "speed": int(v),
                "split": split,
                "file": fname,
                "path": os.path.join(folder, fname),
            })
    return conditions


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    ckpt = os.path.join(case_config.SAVE_DIR, case_config.BEST_MODEL_NAME)
    return jsonify({"status": "ok", "model_loaded": os.path.exists(ckpt)})


@app.route("/api/conditions")
def get_conditions():
    conds = _discover_conditions()
    return jsonify([{k: v for k, v in c.items() if k != "path"} for c in conds])


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True)
    w = body.get("weight", 45)
    v = body.get("speed", 40)

    conds = _discover_conditions()
    match = next(
        (c for c in conds if c["weight"] == w and c["speed"] == v),
        None,
    )
    if match is None:
        return jsonify({"error": f"No data for w={w}, v={v}"}), 404

    result = _run_inference(match["path"])
    if result is None:
        return jsonify({"error": f"Model not trained yet ({case_config.BEST_MODEL_NAME} missing)"}), 503

    result["condition"] = {"weight": w, "speed": v, "split": match["split"]}
    return jsonify(result)


@app.route("/api/results")
def results():
    conds = [c for c in _discover_conditions() if c["split"] == "test"]
    all_results = []
    for c in conds:
        res = _run_inference(c["path"])
        if res is None:
            continue
        all_results.append({
            "condition": {"weight": c["weight"], "speed": c["speed"]},
            "metrics": res["metrics"],
        })
    return jsonify(all_results)


@app.route("/api/upload_predict", methods=["POST"])
def upload_predict():
    """Accept a CSV with sensor columns only, return predictions (no ground truth)."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    _fit_scalers()
    model = _load_model()
    if model is None:
        return jsonify({"error": f"Model not trained yet ({case_config.BEST_MODEL_NAME} missing)"}), 503

    file = request.files["file"]
    try:
        df = pd.read_csv(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {e}"}), 400

    df.columns = df.columns.str.strip()

    required = case_config.DISP_COLS + case_config.ACC_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing columns: {missing}"}), 400

    disp = _disp_scaler.transform(df[case_config.DISP_COLS].values)
    acc = _acc_scaler.transform(df[case_config.ACC_COLS].values)

    n = len(disp)
    X_disp, X_acc = [], []
    for i in range(case_config.SEQ_LEN - 1, n):
        X_disp.append(disp[i - case_config.SEQ_LEN + 1: i + 1])
        X_acc.append(acc[i - case_config.SEQ_LEN + 1: i + 1])

    xd_t = torch.tensor(np.array(X_disp), dtype=torch.float32).to(device)
    xa_t = torch.tensor(np.array(X_acc), dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(xd_t, xa_t).cpu().numpy()

    preds_orig = _target_scaler.inverse_transform(preds)

    n_out = len(preds_orig)
    times_arr = prediction_time_axis_from_dataframe(df, case_config.SEQ_LEN)
    if len(times_arr) != n_out:
        return jsonify({"error": "Time axis length mismatch; check TIME column and row count."}), 400
    times = times_arr.tolist()

    step = max(1, n_out // 300)
    idx = list(range(0, n_out, step))

    series = {}
    output_names = case_config.TARGET_COLS
    for i, col in enumerate(output_names):
        series[col] = {"pred": [float(preds_orig[j, i]) for j in idx]}

    return jsonify({
        "times": [times[j] for j in idx],
        "series": series,
    })


@app.route("/api/videos")
def list_videos():
    video_dir = case_config.SAVE_DIR
    videos = []
    for fname in sorted(os.listdir(video_dir)):
        if fname.endswith(".mp4"):
            videos.append({"name": fname, "url": f"/api/video/{fname}"})
    return jsonify(videos)


@app.route("/api/video/<path:filename>")
def serve_video(filename):
    video_dir = case_config.SAVE_DIR
    return send_from_directory(video_dir, filename)


if __name__ == "__main__":
    print(f"[API] device = {device}")
    print(f"[API] data dir = {case_config.DATA_DIR}")
    _fit_scalers()
    _load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
