# ---------------------------------------------------------------------------
# NOTICE (Apache License 2.0, section 4b)
# ---------------------------------------------------------------------------
# This file is a MODIFIED version of code derived from
#     vehicles-detection-and-counting.ipynb  (Apache License 2.0)
# Original: [original author, if known] — [original source URL]
#
# Modifications made in this repository (see NOTICE.md for the full list):
#   - detector upgraded YOLOv8 -> YOLO26
#   - notebook logic extracted into this library
#   - BoT-SORT tracking, shadow filtering, EMA smoothing and lane-specific
#     geometry removed
#   - functions renamed to be format-neutral (load_mot_counts,
#     evaluate_sequence, ...)
#   - ground-truth evaluation added (frames_to_video, convert_video,
#     plot_count_errors, evaluate_sequence)
#   - ffmpeg odd-dimension encoding and video-inference memory bugs fixed
#   - analyze_danger_video() and BoxMomentumTracker added - per-vehicle danger
#     values from bounding-box growth (looming), with CSV + annotated video
#   - frame rate unified into the single CLIP_FPS constant shared by clip
#     assembly, the counting video and the danger analysis (was 20 fps vs 14)
#   - plot_count_errors / plot_rolling_stats can write their figures to disk
#     (save_path) and evaluate_sequence forwards save_plots_dir, so the paper's
#     count-error and rolling-count figures exist as files, not just inline
#   - show_val_curves now finds the Box*-prefixed curves Ultralytics saves
#     (BoxPR_curve.png, BoxF1_curve.png), which it previously reported missing
#   - summarise_danger() and plot_danger_figures() added, so the danger table and
#     its three figures are reproducible from the notebook instead of from a
#     throwaway script (episodes count consecutive observations; percentages are
#     shares of scored rows, hits >= DANGER_MIN_FRAMES)
#   - plot_count_summary() added and used by evaluate_sequence, so one clip is
#     one count figure instead of two (error, scatter and rolling counts)
#   - plot_detection_quality() added, stacking the validation curves into one
#     figure (the Box*/Mask* lookup is now shared with show_val_curves)
#   - the per-track growth math factored into _track_step() and reused by
#     danger_from_gt(), so the danger value can be computed from the ground-truth
#     boxes and checked against the detector's (validate_danger_against_gt)
#   - matching_stats() added - one-to-one TP / FP / FN matching of the detected
#     boxes against the clip annotations at an overlap of 0.5, which is the
#     paper's Table 1; the per-frame over/under-count equals FP minus FN
#   - load_mot_boxes now shifts MOT frame ids down by one: VisDrone numbers its
#     frames from 1 while the pipeline's detection CSVs number them from 0, so
#     matching_stats / validate_danger_against_gt had compared each annotated
#     frame with the NEXT image's detections (off-by-one, now fixed)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# ---------------------------------------------------------------------------

"""
vd_lib26.py — Library built from vehicles-detection-and-counting.ipynb

YOLO26 edition of vd_lib: everything below is model-agnostic Ultralytics code,
so it works identically with YOLO26 weights (yolo26s.pt pre-trained, best26.pt
fine-tuned).

Packages the vehicle detection / counting / traffic-density workflow from the
notebook into reusable, importable functions:

  * load_model / pick_device           — load a YOLO model (pretrained or fine-tuned)
  * infer_image                        — run detection on a single image
  * infer_video_save                   — run detection on a video, save annotated output
  * count_vehicles_in_video           — per-frame vehicle counting + traffic intensity
                                          (the notebook's "Real Time Traffic Intensity Estimator")
  * count_vehicles_in_images          — per-image vehicle counts over a list of images
                                          (streaming prediction)
  * prepare_visdrone_val              — download only the VisDrone-DET val split (~70 MB),
                                          convert annotations to YOLO, cache locally
  * prepare_visdrone_vehicle_dataset  — full VisDrone-DET (~2 GB) filtered to a single
                                          vehicle class for fine-tuning
  * load_mot_counts                   — parse MOT-style GT CSV into per-frame counts
  * frames_to_video                    — assemble a numbered-frame folder into an MP4
  * convert_video                      — transcode a video to H.264 MP4 (ffmpeg)
  * plot_count_errors                 — RMSE/MAE evaluation of predicted vs ground-truth
                                          counts (replaces the old count-accuracy metric)
  * prepare_visdrone_val_vehicle      — vehicle-only (single class) VisDrone val split for mAP
  * evaluate_map                      — COCO-style val: prints mAP50 / mAP50-95 / P / R
  * evaluate_sequence                 — one-call sequence evaluation (counter + results + plot)
  * analyze_danger_video               — per-vehicle danger values from
                                          bounding-box growth (looming) + CSV

Typical usage (from a notebook in the same folder as the model weights):

    import vd_lib26

    model = vd_lib26.load_model("best26.pt")
    vd_lib26.count_vehicles_in_video(
        model,
        source="sample_video_copy.mp4",
        output_avi="vehicle_count.avi",
    )
"""

# ---------------------------------------------------------------------------
# Apache 2.0 notice — MODIFIED FILE
# ---------------------------------------------------------------------------
# This file is a derivative of "vehicles-detection-and-counting.ipynb"
# (Apache License 2.0). It has been substantially modified; see
# "Changes from the original notebook" in README.md for the full list.
# Licensed under the Apache License, Version 2.0 — see LICENSE.
# ---------------------------------------------------------------------------

import os
import shutil
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import ASSETS_URL
from ultralytics.utils.downloads import download

warnings.filterwarnings("ignore")

__all__ = [
    "pick_device",
    "load_model",
    "infer_image",
    "infer_video_save",
    "draw_info_panels",
    "count_vehicles_in_video",
    "count_vehicles_in_images",
    "plot_sample_grid",
    "plot_rolling_stats",
    "BoxMomentumTracker",
    "associate_boxes",
    "danger_from_ttc",
    "analyze_danger_video",
    "danger_from_gt",
    "load_mot_boxes",
    "validate_danger_against_gt",
    "matching_stats",
    "summarise_danger",
    "plot_danger_figures",
    "prepare_visdrone_val",
    "prepare_visdrone_vehicle_dataset",
    "load_mot_counts",
    "frames_to_video",
    "convert_video",
    "plot_count_errors",
    "plot_count_summary",
    "clip_to_val_dataset",
    "evaluate_clip_map",
    "prepare_visdrone_val_vehicle",
    "evaluate_map",
    "plot_map_summary",
    "plot_detection_quality",
    "show_val_curves",
    "evaluate_sequence",
    "BEST_WEIGHTS",
]

# ---------------------------------------------------------------------------
# Default geometry / visualization parameters (copied from the notebook)
# ---------------------------------------------------------------------------

HEAVY_TRAFFIC_THRESHOLD = 10          # vehicle count above this => "Heavy"
X1, X2 = 325, 635                     # vertical slice range (blacked-out regions)
TEXT_POSITION = (10, 50)              # vehicle-count panel position
INTENSITY_POSITION = (10, 100)        # traffic-intensity panel position
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1
FONT_COLOR = (255, 255, 255)          # white text
BACKGROUND_COLOR = (0, 0, 255)        # red panel background

# Frame rate of the drone clips, in frames per second.
#
# VisDrone distributes its sequences as image frames with no frame rate, so the
# rate must be assumed rather than read from the data. It is part of the method
# and not a display setting: the danger pipeline multiplies each frame's box
# growth by this number to express it per second, so every growth rate and time
# to collision scales with it. Defining it once here keeps the stages that need
# a rate (clip assembly, the counter's output video, the danger analysis) from
# disagreeing about it.
CLIP_FPS = 20

# Fine-tuned YOLO26 model copy shipped next to this library (best26.pt).
# load_model() defaults to it, so the library works standalone with just the
# weights file sitting beside it.
BEST_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best26.pt")


# ---------------------------------------------------------------------------
# Device / model loading
# ---------------------------------------------------------------------------

def pick_device():
    """Return 'mps' if Apple Metal is available, otherwise 'cpu'."""
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights=BEST_WEIGHTS, device=None):
    """Load a YOLO model and move it onto the best available device.

    weights : path to a pretrained (yolo26s.pt) or fine-tuned (best26.pt) weights file.
              Defaults to BEST_WEIGHTS, the YOLO26 trained-model copy in this repo folder.
    """
    model = YOLO(weights)
    if device is None:
        device = pick_device()
    model.to(device)
    return model


# ---------------------------------------------------------------------------
# Single-image / video inference helpers
# ---------------------------------------------------------------------------

def infer_image(model, image_path, imgsz=640, conf=0.5, line_width=2):
    """Run detection on one image and return the annotated frame in RGB.

    Returns an RGB numpy array ready for matplotlib display.
    """
    results = model.predict(source=image_path, imgsz=imgsz, conf=conf, verbose=False)
    annotated = results[0].plot(line_width=line_width)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def infer_video_save(model, source, project=".", name="predict1", exist_ok=True):
    """Run detection over a video and save the annotated output via Ultralytics.

    Equivalent to the notebook's best_model.predict(source=..., name=..., save=True).
    Uses stream=True and consumes the results so frames are written to disk
    incrementally - without it, Ultralytics keeps every frame's results in RAM
    for the whole video (memory grows without bound on long clips).

    Returns the number of frames processed.
    """
    results = model.predict(source=source, project=project, name=name,
                            exist_ok=exist_ok, save=True, stream=True, verbose=False)
    frames = 0
    for _ in results:      # consume the stream; each frame is saved as it goes
        frames += 1
    return frames


# ---------------------------------------------------------------------------
# On-frame drawing helpers
# ---------------------------------------------------------------------------

def draw_info_panels(frame, vehicle_count, intensity, frame_idx=None):
    """Overlay the vehicle-count / traffic-intensity panels (notebook styling)."""
    cv2.rectangle(frame, (TEXT_POSITION[0] - 10, TEXT_POSITION[1] - 25),
                  (TEXT_POSITION[0] + 460, TEXT_POSITION[1] + 10),
                  BACKGROUND_COLOR, -1)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", TEXT_POSITION,
                FONT, FONT_SCALE, FONT_COLOR, 2, cv2.LINE_AA)

    cv2.rectangle(frame, (INTENSITY_POSITION[0] - 10, INTENSITY_POSITION[1] - 25),
                  (INTENSITY_POSITION[0] + 460, INTENSITY_POSITION[1] + 10),
                  BACKGROUND_COLOR, -1)
    cv2.putText(frame, f"Traffic Intensity: {intensity}", INTENSITY_POSITION,
                FONT, FONT_SCALE, FONT_COLOR, 2, cv2.LINE_AA)

    if frame_idx is not None:
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 150),
                    FONT, FONT_SCALE, FONT_COLOR, 2, cv2.LINE_AA)
    return frame


def traffic_intensity(vehicle_count, heavy_threshold=HEAVY_TRAFFIC_THRESHOLD):
    """'Heavy' if vehicle_count exceeds the threshold, else 'Smooth'."""
    return "Heavy" if vehicle_count > heavy_threshold else "Smooth"


# ---------------------------------------------------------------------------
# Pipeline 1: real-time vehicle counting (main_workflow26.ipynb, section 5.1)
# ---------------------------------------------------------------------------

def count_vehicles_in_video(model, source, output_avi,
                            conf=0.4, imgsz=640, fps=CLIP_FPS,
                            x1=X1, x2=X2,
                            heavy_threshold=HEAVY_TRAFFIC_THRESHOLD,
                            return_counts=False):
    """Per-frame detection with vehicle counting + traffic-intensity panels.

    Mirrors the notebook's "Real Time Traffic Intensity Estimator": the frame
    outside the vertical slice [x1, x2) is blacked out before detection. The
    annotated video is written to output_avi.

    Returns the number of frames processed, or (frames, per_frame_counts)
    when return_counts=True (video frame i <-> counts[i]).
    """
    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_avi, fourcc, fps,
                          (int(cap.get(3)), int(cap.get(4))))

    frames_processed = 0
    frame_counts = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detection_frame = frame.copy()
        detection_frame[:x1, :] = 0          # black out top region
        detection_frame[x2:, :] = 0          # black out bottom region

        results = model.predict(detection_frame, imgsz=imgsz, conf=conf, verbose=False)
        processed_frame = results[0].plot(line_width=1)

        # restore the original top/bottom regions
        processed_frame[:x1, :] = frame[:x1, :].copy()
        processed_frame[x2:, :] = frame[x2:, :].copy()

        vehicles_in_frame = 0
        bounding_boxes = results[0].boxes

        for box in bounding_boxes.xyxy:
            vehicles_in_frame += 1

            # box size + coordinates labels (as in the notebook)
            x01, y01, x02, y02 = box
            width = x02 - x01
            height = y02 - y01
            cv2.putText(processed_frame, f"{round(float(width * height))}",
                        (int(x02), int(y02) - 10), FONT, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(processed_frame, f"x: {round(float(x01))}, y: {round(float(y01))}",
                        (int(x01), int(y01) - 10), FONT, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        draw_info_panels(
            processed_frame,
            vehicle_count=vehicles_in_frame,
            intensity=traffic_intensity(vehicles_in_frame, heavy_threshold),
        )
        out.write(processed_frame)
        frame_counts.append(vehicles_in_frame)
        frames_processed += 1

    cap.release()
    out.release()
    if return_counts:
        return frames_processed, frame_counts
    return frames_processed


# ---------------------------------------------------------------------------
# Evaluation helpers: MOT-style ground truth, frame folders, accuracy plots
# ---------------------------------------------------------------------------

def load_mot_counts(gt_path, classes=(4, 5, 6, 9)):
    """Parse a MOT-style annotation CSV into {frame_id: vehicle_count}.

    MOT-style format (CSV, one box per line):
        frame, id, x, y, w, h, score, class, truncation, occlusion
    Only boxes whose class is in `classes` are counted (class ids follow the
    VisDrone convention: 4=car, 5=van, 6=truck, 9=bus; pedestrians/bikes are
    excluded by default).
    """
    counts = {}
    with open(gt_path) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) >= 8 and int(p[7]) in classes:
                fid = int(p[0])
                counts[fid] = counts.get(fid, 0) + 1
    return counts


def frames_to_video(frames_dir, output_video, fps=CLIP_FPS):
    """Assemble a folder of sequentially-numbered frames into an MP4.

    Uses ffmpeg (must be on PATH). The scale filter forces even dimensions
    because H.264 (yuv420p) cannot encode odd heights (e.g. the 1360x765
    aerial frames). Returns (width, height) of the assembled video.
    """
    import re
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found - install it and make sure it is on PATH")

    frames_dir = os.fspath(frames_dir)
    output_video = os.fspath(output_video)
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)
    if os.path.exists(output_video):
        os.remove(output_video)          # clear any stale/empty output

    names = sorted(f for f in os.listdir(frames_dir)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not names:
        raise RuntimeError(f"no image frames found in {frames_dir}")

    stem = os.path.splitext(names[0])[0]
    digits = len(re.sub(r"\D", "", stem)) or 1
    ext = os.path.splitext(names[0])[1]
    pattern = os.path.join(frames_dir, f"%0{digits}d{ext}")

    subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                    "-framerate", str(fps),
                    "-i", pattern,
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    output_video], check=True)

    img = cv2.imread(os.path.join(frames_dir, names[0]))
    h, w = img.shape[:2] if img is not None else (0, 0)
    return w, h


def convert_video(source, destination):
    """Transcode a video to an H.264 MP4 (e.g. an annotated AVI -> viewable MP4)."""
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found - install it and make sure it is on PATH")

    destination = os.fspath(destination)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error",
                    "-i", os.fspath(source),
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    destination], check=True)
    return destination


def plot_count_errors(gt_counts, pred_counts, show=True, label="frame",
                      save_path=None):
    """RMSE/MAE evaluation of predicted vs ground-truth counts.

    Replaces the old "count accuracy" metric (1 - |pred-gt| / max(gt, 1)) with
    proper error statistics: RMSE and MAE of the per-frame/image vehicle
    counts. gt_counts is a dict {id: count} where GT id i matches
    pred_counts[i - 1] (as returned by count_vehicles_in_video /
    count_vehicles_in_images); works equally for image sets by keying images
    from 1. Plots the absolute error per item plus a predicted-vs-ground-truth
    scatter. ``label`` ("frame" or "image") is used in the plot and print
    text. Returns (rmse, mae, n). When ``save_path`` is given, the two-panel
    figure is written to that path as a PNG (this is how the paper's
    count-error figure is produced).
    """
    import matplotlib.pyplot as plt

    ids, pairs = count_errors(gt_counts, pred_counts)
    gts = np.array([g for g, _ in pairs])
    preds = np.array([p for _, p in pairs])
    errs = preds - gts
    rmse = float(np.sqrt(np.mean(errs ** 2)))
    mae = float(np.mean(np.abs(errs)))
    n = len(pairs)
    print(f"count RMSE over {n} annotated {label}s: {rmse:.3f}  "
          f"(MAE {mae:.3f}, mean ground-truth count {gts.mean():.1f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(ids, np.abs(errs), "-", lw=1, alpha=0.6, label="|error|")
    ax1.axhline(rmse, color="tab:red", ls="--", label=f"RMSE = {rmse:.3f}")
    ax1.axhline(mae, color="tab:orange", ls=":", label=f"MAE = {mae:.3f}")
    ax1.set(xlabel=label, ylabel="|predicted - ground truth|",
            title=f"Count error per {label}")
    ax1.legend(); ax1.grid(alpha=0.3)

    lim = max(int(gts.max()), int(preds.max()), 1) + 1
    ax2.scatter(gts, preds, alpha=0.7, s=18)
    ax2.plot([0, lim], [0, lim], "r--", lw=1)
    ax2.set(xlabel="ground-truth vehicles", ylabel="predicted vehicles",
            title="Predicted vs ground truth", xlim=(0, lim), ylim=(0, lim))
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("saved figure:", save_path)
    if show:
        plt.show()
    return rmse, mae, n


def count_errors(gt_counts, pred_counts):
    """Paired (ground_truth, predicted) counts plus the matching ids.

    gt_counts is a dict {id: count} where id i (1-based) matches
    pred_counts[i - 1]. Returns (ids, pairs) with pairs = [(gt, pred), ...].
    """
    pairs, ids = [], []
    for g, gt in sorted(gt_counts.items()):
        if 1 <= g <= len(pred_counts):
            ids.append(g)
            pairs.append((int(gt), int(pred_counts[g - 1])))
    if not pairs:
        raise RuntimeError("no ground-truth entries matched the prediction range")
    return ids, pairs


def clip_to_val_dataset(frames_dir, gt_path, dst_root, classes=(4, 5, 6, 9)):
    """Build a single-class YOLO val dataset from a VisDrone MOT clip.

    Hard-links the clip frames and converts gt.txt (MOT format:
    ``frame,id,x,y,w,h,score,class,trunc,occ``) into YOLO labels filtered to
    the vehicle classes (renumbered to a single class 0), so Ultralytics
    val() can report mAP50/mAP50-95 for the clip. ``dst_root`` is where the
    mirror (images/val + labels/val) and the dataset YAML are created.
    Returns the YAML path.
    """
    frames_dir = Path(frames_dir)
    files = sorted(f for f in frames_dir.glob("*.jpg"))
    if not files:
        raise RuntimeError(f"no jpg frames found in {frames_dir}")
    img_size = Image.open(files[0]).size           # clip frames share one size
    dw, dh = 1.0 / img_size[0], 1.0 / img_size[1]

    dst_root = Path(dst_root)
    img_dir = dst_root / "images" / "val"
    lbl_dir = dst_root / "labels" / "val"
    img_dir.parent.mkdir(parents=True, exist_ok=True)
    lbl_dir.parent.mkdir(parents=True, exist_ok=True)

    if img_dir.exists():                            # rebuild image hard-link mirror
        for old in img_dir.glob("*"):
            old.unlink()
        img_dir.rmdir()
    img_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        os.link(f, img_dir / f.name)

    for old in list(lbl_dir.glob("*.txt")) + list(lbl_dir.glob("*.cache")):
        old.unlink()
    lbl_dir.mkdir(parents=True, exist_ok=True)

    by_frame = {}
    for line in Path(gt_path).read_text().splitlines():
        p = line.strip().split(",")
        if len(p) < 8 or int(p[7]) not in classes:
            continue
        fid, x, y, w, h = int(p[0]), int(float(p[2])), int(float(p[3])), int(float(p[4])), int(float(p[5]))
        if w <= 0 or h <= 0:
            continue
        by_frame.setdefault(fid, []).append((x, y, w, h))

    n_lbl = 0
    for fid, boxes in sorted(by_frame.items()):
        if not (1 <= fid <= len(files)):            # frame id <-> file index
            continue
        lines = []
        for x, y, w, h in boxes:
            xc, yc = (x + w / 2) * dw, (y + h / 2) * dh
            lines.append(f"0 {xc:.6f} {yc:.6f} {w * dw:.6f} {h * dh:.6f}\n")
        (lbl_dir / f"{files[fid - 1].stem}.txt").write_text("".join(lines))
        n_lbl += 1

    yaml_path = dst_root / "clip_val.yaml"
    yaml_path.write_text(
        f"path: {dst_root}\n"
        "train: images/val\n"       # placeholder: only the val split is used
        "val: images/val\n"
        "names:\n"
        "  0: vehicle\n"
    )
    print(f"  clip val dataset: {len(files)} images, {n_lbl} label files -> {yaml_path}")
    return yaml_path


def evaluate_clip_map(model, frames_dir, gt_path, cache_dir, classes=(4, 5, 6, 9),
                      imgsz=640):
    """Detection quality (mAP50 / mAP50-95) of `model` on one VisDrone clip.

    Converts the clip (frames + MOT gt.txt) into a single-class YOLO val
    dataset under ``cache_dir`` (hard links - no image copies), runs
    Ultralytics val() on it and prints mAP50 / mAP50-95 / precision / recall.
    Returns the metrics object.
    """
    yaml_path = clip_to_val_dataset(frames_dir, gt_path,
                                    Path(cache_dir) / Path(frames_dir).name,
                                    classes=classes)
    metrics = model.val(data=str(yaml_path), imgsz=imgsz,
                        project=str(Path(cache_dir) / "results"),
                        name="val", exist_ok=True, verbose=False)
    print(f"  clip detection quality ({Path(frames_dir).name}): "
          f"mAP50 {metrics.box.map50:.4f} | mAP50-95 {metrics.box.map:.4f} | "
          f"P {metrics.box.p[0]:.4f} | R {metrics.box.r[0]:.4f}")
    return metrics


def evaluate_sequence(model, frames_dir, gt_path, out_dir,
                      classes=(4, 5, 6, 9), conf=0.4, fps=CLIP_FPS,
                      video_name="sequence_test.mp4",
                      count_avi="sequence_count.avi",
                      results_mp4="sequence_test_results.mp4",
                      compute_map=True, map_imgsz=640, rolling_window=10,
                      save_plots_dir=None):
    """End-to-end sequence evaluation: ground truth -> counter -> results.

    Assembles `frames_dir` into a video, runs count_vehicles_in_video with the
    detection slice disabled (aerial frames are not 1280x720), converts the
    annotated output into a viewable MP4, and reports RMSE/MAE of the
    per-frame vehicle counts against the ground-truth annotations, followed by
    a rolling mean +/- rolling std chart of the counts. When ``compute_map`` is
    True (default), the clip is additionally evaluated with Ultralytics val()
    on its own frames for per-clip mAP50 / mAP50-95.

    When ``save_plots_dir`` is set, one combined count figure per clip (error,
    scatter and rolling counts) is written there as a PNG named after the clip
    folder (this is what the paper's count figures are).

    Returns (rmse, mae, n) from plot_count_errors.
    """
    gt_counts = load_mot_counts(gt_path, classes)
    print(f"frames with vehicle annotations: {len(gt_counts)}")

    test_video = os.path.join(os.fspath(out_dir), video_name)
    _, h = frames_to_video(frames_dir, test_video, fps=fps)
    n_frames, pred_counts = count_vehicles_in_video(
        model, test_video,
        output_avi=os.path.join(os.fspath(out_dir), count_avi),
        conf=conf, x1=0, x2=h, return_counts=True)
    if n_frames == 0:
        raise RuntimeError("the assembled video is empty - check frames_to_video")

    results_video = os.path.join(os.fspath(out_dir), results_mp4)
    convert_video(os.path.join(os.fspath(out_dir), count_avi), results_video)
    print("annotated results video:", results_video)

    tag = Path(frames_dir).name
    plots = None
    if save_plots_dir is not None:
        plots = Path(save_plots_dir)
        plots.mkdir(parents=True, exist_ok=True)

    # 1) count error + rolling counts, drawn as one combined figure per clip
    rmse, mae, n = plot_count_summary(
        gt_counts, pred_counts,
        window=rolling_window if (rolling_window and rolling_window > 0) else 1,
        save_path=(plots / f"count_summary_{tag}.png") if plots else None,
        title=tag)

    # 3) per-clip detection quality (mAP50 / mAP50-95) via Ultralytics val()
    if compute_map:
        evaluate_clip_map(model, frames_dir, gt_path,
                          os.path.join(os.fspath(out_dir), "clip_val"),
                          classes=classes, imgsz=map_imgsz)
    return rmse, mae, n


# ---------------------------------------------------------------------------
# VisDrone image-set helpers (notebook section 6.1)
# ---------------------------------------------------------------------------

def prepare_visdrone_val(data_root="datasets/VisDrone"):
    """Download only the VisDrone-DET validation split (~70 MB) and prepare it.

    Fetches VisDrone2019-DET-val.zip from the Ultralytics assets, extracts it,
    converts the annotations to YOLO format (classes 0-9), and caches the
    result under ``data_root``. If the split was already prepared (or the full
    dataset was downloaded previously), the cache is reused.

    Returns (val_images_dir, val_labels_dir) as pathlib.Path.
    """
    data_root = Path(data_root)
    val_dir = data_root / "images" / "val"
    labels_dir = data_root / "labels" / "val"

    if not any(labels_dir.glob("*.txt")):
        print("Downloading VisDrone2019-DET-val.zip (~70 MB) ...")
        download(f"{ASSETS_URL}/VisDrone2019-DET-val.zip", dir=data_root)  # dl + unzip
        src = data_root / "VisDrone2019-DET-val"
        val_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for img in (src / "images").glob("*.jpg"):          # move images
            img.rename(val_dir / img.name)
        for f in (src / "annotations").glob("*.txt"):       # VisDrone -> YOLO
            size = Image.open(val_dir / f.with_suffix(".jpg").name).size
            dw, dh = 1.0 / size[0], 1.0 / size[1]
            lines = []
            for row in [x.split(",") for x in f.read_text(encoding="utf-8").strip().splitlines()]:
                if row[4] != "0":                           # skip ignored regions
                    x, y, w, h = map(int, row[:4])
                    cls = int(row[5]) - 1                   # 1-based -> 0-based
                    xc, yc = (x + w / 2) * dw, (y + h / 2) * dh
                    lines.append(f"{cls} {xc:.6f} {yc:.6f} {w * dw:.6f} {h * dh:.6f}\n")
            (labels_dir / f.name).write_text("".join(lines), encoding="utf-8")
        shutil.rmtree(src)
        (data_root / "VisDrone2019-DET-val.zip").unlink()
        print(f"val split ready: {len(list(val_dir.glob('*.jpg')))} images")
    return val_dir, labels_dir


def count_vehicles_in_images(model, image_paths, conf=0.4, imgsz=640,
                             return_frames=False, thumb_width=480):
    """Per-image vehicle counts over a list of images (streaming prediction).

    Uses ``stream=True`` so images are processed one at a time (low memory).
    A 1-class model (fine-tuned) counts every box; a multi-class model
    (e.g. COCO) counts only car / bus / truck (classes 2, 5, 7).

    With ``return_frames=True``, also returns small annotated RGB thumbnails
    (one per image, ordered like ``image_paths``, max ``thumb_width`` px wide)
    for grid display. Returns counts alone, or (counts, thumbnails).
    """
    counts, thumbs = [], []
    for r in model.predict(source=[str(p) for p in image_paths],
                           stream=True, conf=conf, imgsz=imgsz, verbose=False):
        if r.boxes is None or len(r.boxes) == 0:
            counts.append(0)
        else:
            cls = r.boxes.cls.cpu().numpy()
            if len(model.names) == 1:
                counts.append(int(len(cls)))
            else:
                counts.append(int(sum(1 for c in cls if int(c) in {2, 5, 7})))
        if return_frames:
            frame = r.plot(line_width=1)            # annotated BGR frame
            h, w = frame.shape[:2]
            if w > thumb_width:
                frame = cv2.resize(frame, (thumb_width, max(1, int(h * thumb_width / w))))
            thumbs.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if return_frames:
        return counts, thumbs
    return counts


def plot_sample_grid(thumbnails, pred_counts, gt_counts=None, cols=5, show=True):
    """Display evaluated images in a grid, annotated with their counts.

    Each grid tile shows one annotated image; the title reports the predicted
    (and ground-truth) vehicle count - green when they match, red otherwise.
    ``thumbnails`` is the list returned by count_vehicles_in_images with
    return_frames=True. Returns the figure.
    """
    import matplotlib.pyplot as plt

    n = len(thumbnails)
    if n == 0:
        raise RuntimeError("no thumbnails to display")
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(thumbnails[i])
        if gt_counts is not None:
            ok = pred_counts[i] == gt_counts[i]
            ax.set_title(f"#{i + 1}: pred {pred_counts[i]} / GT {gt_counts[i]}",
                         color=("green" if ok else "red"), fontsize=10)
        else:
            ax.set_title(f"#{i + 1}: {pred_counts[i]} vehicles", fontsize=10)
    fig.suptitle("Sampled images - predicted vs ground-truth vehicle counts")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_rolling_stats(pred_counts, gt_counts=None, window=10, show=True,
                       label="image", save_path=None):
    """Rolling mean and rolling standard deviation of the predicted counts.

    Plots the per-{image,frame} predicted counts with the rolling mean and a
    shaded rolling-std band (window sized by ``window``, min_periods=1), plus
    the ground-truth counts when given. Returns (fig, rolling_df), where
    rolling_df is a DataFrame with columns pred / rolling_mean / rolling_std.
    When ``save_path`` is set, the figure is also written there as a PNG.
    """
    import matplotlib.pyplot as plt

    pred = pd.Series(pred_counts, dtype=float, name="pred")
    rolling_mean = pred.rolling(window=window, min_periods=1).mean()
    rolling_std = pred.rolling(window=window, min_periods=1).std().fillna(0.0)
    xs = np.arange(1, len(pred) + 1)

    # gt_counts may be a dict {id: count} (id i <-> pred_counts[i-1], as in
    # plot_count_errors) or a list aligned with pred_counts - normalise both.
    if gt_counts is not None and isinstance(gt_counts, dict):
        gt_series = [gt_counts.get(i, 0) for i in range(1, len(pred) + 1)]
    else:
        gt_series = gt_counts

    fig, ax = plt.subplots(figsize=(14, 5))
    if gt_series is not None:
        ax.plot(xs, gt_series, "o-", lw=1, ms=3, color="gray", alpha=0.55,
                label="ground truth")
    ax.plot(xs, pred, "o-", lw=1, ms=3, color="tab:blue", alpha=0.75, label="predicted")
    ax.plot(xs, rolling_mean, "-", color="tab:red", lw=2,
            label=f"rolling mean (window {window})")
    ax.fill_between(xs, rolling_mean - rolling_std, rolling_mean + rolling_std,
                    color="tab:red", alpha=0.2,
                    label=f"rolling std (window {window})")
    ax.set(xlabel=label, ylabel="vehicles",
           title=f"Rolling mean and std of predicted vehicle counts")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("saved figure:", save_path)
    if show:
        plt.show()

    rolling_df = pd.DataFrame({"pred": pred, "rolling_mean": rolling_mean,
                               "rolling_std": rolling_std})
    last = rolling_df.iloc[-1]
    print(f"rolling stats (window {window}, n={len(pred)}): "
          f"final mean {last['rolling_mean']:.2f} +/- {last['rolling_std']:.2f} "
          f"(std), min {pred.min():.0f}, max {pred.max():.0f}")
    return fig, rolling_df


def plot_count_summary(gt_counts, pred_counts, window=10, save_path=None,
                       title=None, show=False, label="frame"):
    """One combined count figure per clip: error, scatter and rolling counts.

    Stacks the charts that plot_count_errors() and plot_rolling_stats() draw
    separately, so a clip occupies a single figure instead of two: the top row
    holds the absolute error per frame and the predicted-versus-annotated
    scatter, and the bottom row spans the width with the rolling mean of the
    predicted counts, its rolling-std band and the annotated counts.

    Returns (rmse, mae, n), the same values plot_count_errors() returns. When
    ``save_path`` is set the figure is written there as a PNG.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    ids, pairs = count_errors(gt_counts, pred_counts)
    gts = np.array([g for g, _ in pairs])
    preds = np.array([p for _, p in pairs])
    errs = preds - gts
    rmse = float(np.sqrt(np.mean(errs ** 2)))
    mae = float(np.mean(np.abs(errs)))
    n = len(pairs)

    pred = pd.Series(preds, dtype=float)
    window = max(int(window), 1)
    rolling_mean = pred.rolling(window=window, min_periods=1).mean()
    rolling_std = pred.rolling(window=window, min_periods=1).std().fillna(0.0)
    xs = np.arange(1, n + 1)

    fig = plt.figure(figsize=(16, 9))
    grid = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.18)
    ax_err = fig.add_subplot(grid[0, 0])
    ax_sc = fig.add_subplot(grid[0, 1])
    ax_roll = fig.add_subplot(grid[1, :])

    # top left: absolute error per frame, with the two summary statistics
    ax_err.plot(ids, np.abs(errs), "-", lw=1, alpha=0.6, label="|error|")
    ax_err.axhline(rmse, color="tab:red", ls="--", label=f"RMSE = {rmse:.3f}")
    ax_err.axhline(mae, color="tab:orange", ls=":", label=f"MAE = {mae:.3f}")
    ax_err.set(xlabel=label, ylabel="|predicted - ground truth|",
               title=f"Count error per {label}")
    ax_err.legend(); ax_err.grid(alpha=0.3)

    # top right: predicted against annotated counts
    lim = max(int(gts.max()), int(preds.max()), 1) + 1
    ax_sc.scatter(gts, preds, alpha=0.7, s=18)
    ax_sc.plot([0, lim], [0, lim], "r--", lw=1)
    ax_sc.set(xlabel="ground-truth vehicles", ylabel="predicted vehicles",
              title="Predicted vs ground truth", xlim=(0, lim), ylim=(0, lim))
    ax_sc.grid(alpha=0.3)

    # bottom: rolling mean of the prediction with the annotations overlaid
    ax_roll.plot(xs, gts, "o-", lw=1, ms=3, color="gray", alpha=0.55,
                 label="ground truth")
    ax_roll.plot(xs, preds, "o-", lw=1, ms=3, color="tab:blue", alpha=0.75,
                 label="predicted")
    ax_roll.plot(xs, rolling_mean, "-", color="tab:red", lw=2,
                 label=f"rolling mean (window {window})")
    ax_roll.fill_between(xs, rolling_mean - rolling_std, rolling_mean + rolling_std,
                         color="tab:red", alpha=0.2,
                         label=f"rolling std (window {window})")
    ax_roll.set(xlabel=label, ylabel="vehicles",
                title="Rolling mean and std of predicted vehicle counts")
    ax_roll.legend(); ax_roll.grid(alpha=0.3)

    if title:
        fig.suptitle(f"{title}  -  RMSE {rmse:.3f}, MAE {mae:.3f} over {n} {label}s",
                     fontsize=13)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print("saved figure:", save_path)
    if show:
        plt.show()
    return rmse, mae, n


def prepare_visdrone_val_vehicle(data_root="datasets/VisDrone"):
    """Build a vehicle-only (single class 0) VisDrone VAL split for mAP.

    Uses prepare_visdrone_val() (~70 MB download, cached after the first run)
    and mirrors the val images as hard links with labels filtered to
    car / van / truck / bus (classes 3, 4, 5, 8 -> 0), so Ultralytics val()
    evaluates the fine-tuned single-class model against matching labels.
    Returns the path to the generated dataset YAML.
    """
    val_img, val_lbl = prepare_visdrone_val(data_root)
    data_root = Path(data_root).resolve()
    veh_root = data_root.parent / "VisDroneValVehicle"
    vehicle_cls = {3, 4, 5, 8}                  # car, van, truck, bus

    dst_img = veh_root / "images" / "val"
    dst_lbl = veh_root / "labels" / "val"
    dst_lbl.mkdir(parents=True, exist_ok=True)
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    if dst_img.exists():                        # rebuild the image mirror
        for old in dst_img.glob("*"):
            old.unlink()
        dst_img.rmdir()
    dst_img.mkdir(parents=True, exist_ok=True)
    for jpg in Path(val_img).glob("*.jpg"):
        os.link(jpg, dst_img / jpg.name)
    for old in list(dst_lbl.glob("*.txt")) + list(dst_lbl.glob("*.cache")):
        old.unlink()
    n_lbl = 0
    for f in Path(val_lbl).glob("*.txt"):
        kept = []
        for line in f.read_text().splitlines():
            t = line.split()
            if t and int(t[0]) in vehicle_cls:
                kept.append("0 " + " ".join(t[1:]))
        if kept:
            (dst_lbl / f.name).write_text("\n".join(kept) + "\n")
            n_lbl += 1

    yaml_path = veh_root / "visdrone_val_vehicle.yaml"
    yaml_path.write_text(
        f"path: {veh_root}\n"
        "train: images/val\n"       # placeholder: only the val split is used
        "val: images/val\n"
        "names:\n"
        "  0: vehicle\n"
    )
    n_img = len(list(dst_img.glob("*.jpg")))
    print(f"vehicle val ready for mAP: {n_img} images, {n_lbl} label files -> {yaml_path}")
    return yaml_path


def evaluate_map(model, yaml_path=None, imgsz=640, data_root="datasets/VisDrone",
                 project="runs/map_val", name="val"):
    """COCO-style validation of `model`: mAP50 and mAP50-95.

    Evaluates against the vehicle-only VisDrone val split (auto-downloaded,
    ~70 MB once, cached afterwards) so the labels match the fine-tuned
    single-class model. Prints mAP50 / mAP50-95 / precision / recall and
    returns the Ultralytics metrics object (metrics.box.map50, metrics.box.map).
    The standard plots (PR curve, confusion matrix, F1 curve, ...) are saved
    into ``<project>/<name>`` - display them with show_val_curves().
    """
    if yaml_path is None:
        yaml_path = prepare_visdrone_val_vehicle(data_root)
    metrics = model.val(data=str(yaml_path), imgsz=imgsz,
                        project=project, name=name, exist_ok=True, verbose=False)
    print("=" * 60)
    print("Detection quality on the VisDrone val split (vehicles, single class):")
    print(f"  mAP50:     {metrics.box.map50:.4f}")
    print(f"  mAP50-95:  {metrics.box.map:.4f}")
    print(f"  precision: {metrics.box.p[0]:.4f}")
    print(f"  recall:    {metrics.box.r[0]:.4f}")
    print("=" * 60)
    return metrics


def plot_map_summary(metrics, show=True):
    """Grid display of the detection-quality metrics from a model.val() run.

    Draws one bar panel per metric (mAP50, mAP50-95, precision, recall) with
    its value annotated on top. ``metrics`` is what evaluate_map / model.val()
    returns. Returns the figure.
    """
    import matplotlib.pyplot as plt

    names = ["mAP50", "mAP50-95", "Precision", "Recall"]
    values = [float(metrics.box.map50), float(metrics.box.map),
              float(metrics.box.p[0]), float(metrics.box.r[0])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, name, v in zip(np.atleast_1d(axes).ravel(), names, values):
        ax.bar([name], [v], color="#4c72b0", width=0.45)
        ax.set_ylim(0, 1.0)
        ax.axhline(v, color="#c44e52", ls=":", lw=1)
        ax.set_title(f"{name} = {v:.4f}", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Detection quality on the VisDrone val split (vehicles, single class)")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def _find_saved_curve(save_dir, name):
    """Path to a curve Ultralytics saved, accepting the Box*/Mask* prefixed names."""
    save_dir = Path(save_dir)
    for candidate in (save_dir / name, save_dir / f"Box{name}", save_dir / f"Mask{name}"):
        if candidate.exists():
            return candidate
    return None


def _crop_white(path, pad=8):
    """Load a PNG and trim its near-white margins, so stacked panels sit close."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    a = np.asarray(im)
    ink = (a < 245).any(axis=2)
    rows = np.where(ink.any(axis=1))[0]
    cols = np.where(ink.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return im
    y0, y1 = max(int(rows[0]) - pad, 0), min(int(rows[-1]) + pad, a.shape[0])
    x0, x1 = max(int(cols[0]) - pad, 0), min(int(cols[-1]) + pad, a.shape[1])
    return im.crop((x0, y0, x1, y1))


def plot_detection_quality(val_dir, save_path=None, show=False, dpi=150,
                           width_in=11.0, width_fracs=None, label_panels=True,
                           panels=(("PR_curve.png", "Precision-recall curve"),
                                   ("confusion_matrix.png", "Confusion matrix"))):
    """Stack the curves Ultralytics saved during val() into one figure.

    ``val_dir`` is the folder the validation wrote to (the project/name passed
    to evaluate_map / model.val), and the file names are matched with or
    without the Box*/Mask* prefix Ultralytics adds for the detection task.

    The panels are stacked vertically, each spanning the full figure width,
    because two panels side by side would shrink the confusion matrix's inner
    text to the point of being unreadable once the paper scales the figure to a
    page column. Near-white margins are trimmed first. Panels are labelled
    (a), (b), ... and ``width_fracs`` can narrow individual panels (a fraction
    of ``width_in`` each) to bring the overall aspect ratio down to something
    that fits one page. Returns the list of panel labels actually drawn.
    """
    import matplotlib.pyplot as plt

    found = []
    for name, label in panels:
        path = _find_saved_curve(val_dir, name)
        if path is None:
            print("(missing panel:", name, "- nothing drawn for it)")
            continue
        found.append((_crop_white(path), label))
    if not found:
        print("(no panels found in", val_dir, ")")
        return []

    width_px = max(im.size[0] for im, _ in found)
    fracs = list(width_fracs) if width_fracs else [1.0] * len(found)
    if len(fracs) < len(found):
        fracs += [fracs[-1]] * (len(found) - len(fracs))

    title_band = 0.34                          # inches reserved for each panel label
    heights = [width_in * f * im.size[1] / im.size[0]
               for (im, _), f in zip(found, fracs)]
    total_h = sum(heights) + title_band * len(found)

    fig = plt.figure(figsize=(width_in, total_h))
    y = total_h                                # lay the panels out from the top down
    for i, ((im, label), frac, h) in enumerate(zip(found, fracs, heights)):
        ax = fig.add_axes([(1 - frac) / 2, (y - h) / total_h, frac, h / total_h])
        ax.imshow(im)
        ax.set_axis_off()
        if label_panels:
            ax.set_title(f"({chr(97 + i)}) {label}", fontsize=12)
        y -= h + title_band

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"saved figure     : {save_path}  ({len(found)} panels, {width_px}px wide source)")
    if show:
        plt.show()
    return [label for _, label in found]


def show_val_curves(save_dir, curves=("PR_curve.png", "confusion_matrix.png",
                                      "F1_curve.png"), show=True):
    """Display the standard plots Ultralytics saved during model.val().

    ``save_dir`` is the val output folder (the project/name passed to
    evaluate_map / model.val). Only curves that exist are shown; missing files
    are noted. Returns the list of displayed file paths.
    """
    import matplotlib.pyplot as plt

    shown = []
    save_dir = Path(save_dir)
    for name in curves:
        f = _find_saved_curve(save_dir, name)
        if f is None:
            print("(no saved curve:", name, "or Box" + name,
                  "- run val with plots enabled)")
            continue
        img = plt.imread(f)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f.stem.replace("_", " "))
        plt.tight_layout()
        if show:
            plt.show()
        shown.append(f)
    return shown


def prepare_visdrone_vehicle_dataset(data_root="datasets/VisDrone"):
    """Download the full Ultralytics VisDrone-DET dataset (~2 GB) and build a
    vehicle-only, single-class variant for fine-tuning.

    The Ultralytics downloader fetches train + val + test and converts the
    original annotations to YOLO format (classes 0-9). This function keeps only
    the vehicle classes (3=car, 4=van, 5=truck, 8=bus) as a single class 0,
    hard-linking the images into a real directory (no extra disk, no symlinks)
    and writing filtered labels under ``<datasets>/VisDroneVehicle``.
    Idempotent: cached after the first run.

    Returns the path to a ready-to-use dataset YAML.
    """
    dataset = check_det_dataset("VisDrone.yaml")    # full download + YOLO conversion
    data_root = Path(data_root).resolve()
    veh_root = data_root.parent / "VisDroneVehicle"
    vehicle_cls = {3, 4, 5, 8}                  # car, van, truck, bus

    for split in ("train", "val"):
        _src = dataset.get(split)
        src_img = Path(_src[0] if isinstance(_src, list) else _src)   # what Ultralytics resolved
        src_lbl = src_img.parent.parent / "labels" / src_img.name
        dst_img = veh_root / "images" / split
        dst_lbl = veh_root / "labels" / split

        if not src_img.exists():
            raise RuntimeError(
                f"VisDrone images not found at {src_img} after download - "
                "the Ultralytics dataset layout may have changed")
        if not src_lbl.exists():
            raise RuntimeError(
                f"VisDrone labels not found at {src_lbl} after conversion - "
                "the Ultralytics dataset layout may have changed")

        # --- labels: always rebuilt fresh, stale caches removed ---
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for old in list(dst_lbl.glob("*.txt")) + list(dst_lbl.glob("*.cache")):
            old.unlink()
        for f in src_lbl.glob("*.txt"):
            kept = []
            for line in f.read_text().splitlines():
                t = line.split()
                if t and int(t[0]) in vehicle_cls:
                    kept.append("0 " + " ".join(t[1:]))
            if kept:
                (dst_lbl / f.name).write_text("\n".join(kept) + "\n")

        # --- images: real dir of hard links (NOT symlinks) ---
        # Ultralytics resolves symlinked image dirs to their real path and
        # reads the labels next to that path - the original unfiltered
        # 10-class labels - which silently drops every image with a
        # non-vehicle class ("corrupt"). Hard links fix this with no extra disk.
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if dst_img.is_symlink() or dst_img.is_file():
            dst_img.unlink()
        if dst_img.exists():
            for old in dst_img.glob("*"):
                old.unlink()
            dst_img.rmdir()
        dst_img.mkdir(parents=True, exist_ok=True)
        n_links = 0
        for jpg in src_img.glob("*.jpg"):
            os.link(jpg, dst_img / jpg.name)
            n_links += 1

        n_img = n_links
        n_lbl = len(list(dst_lbl.glob("*.txt")))
        print(f"  {split}: {n_img} images, {n_lbl} vehicle label files")
        min_ok = 100 if split == "train" else 50
        if n_img < min_ok or n_lbl < 10:
            raise RuntimeError(
                f"{split} split looks broken ({n_img} images, {n_lbl} label files). "
                "Delete stale caches and re-run:  rm -rf datasets/VisDroneVehicle"
            )

    yaml_path = veh_root / "visdrone_vehicle.yaml"
    yaml_path.write_text(
        f"path: {veh_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: vehicle\n"
    )
    print("VisDrone vehicle-only dataset ready:", yaml_path)
    return yaml_path


# ---------------------------------------------------------------------------
# Bounding-box momentum -> per-vehicle danger values
# ---------------------------------------------------------------------------
#
# The counting pipeline above treats every box as an anonymous detection. This
# section adds what a driving system needs from the same boxes: an identity per
# vehicle, and a danger value derived from how fast that vehicle's box grows.
#
# Geometric justification (looming): for a pinhole camera an object's image size
# s is inversely proportional to its distance Z, so
#     (1/s) (ds/dt) = v_los / Z = 1 / TTC
# i.e. the relative growth rate of the detected box is a ranging-sensor-free
# estimate of 1/TTC. Here the boxes are linked into short tracks with a small
# greedy IoU association (deliberately not BoT-SORT / ByteTrack), the log-size
# slope is smoothed with an EMA so one noisy detection cannot spike the value,
# and the result is mapped to a danger value in [0, 1] written per frame.

DEFAULT_FPS = CLIP_FPS      # per-second growth rates (single source: CLIP_FPS)

DANGER_IOU = 0.30            # minimum IoU to link a box to the previous frame
DANGER_EMA = 0.5             # growth smoothing; higher = slower/steadier
DANGER_MIN_FRAMES = 3        # frames before a danger value is reported
DANGER_MAX_AGE = 8           # drop a track after this many unmatched frames
DANGER_TTC_SAFE = 4.0        # s - danger value is 0 at/above this TTC
DANGER_TTC_CRITICAL = 1.0    # s - danger value saturates at 1 at/below this TTC


def box_iou(a, b):
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def associate_boxes(prev_boxes, cur_boxes, iou_threshold=DANGER_IOU):
    """Greedy IoU association between two frames' boxes.

    Returns ``(matches, unmatched_cur)``: matches is a list of
    ``(prev_index, cur_index)`` pairs ordered by decreasing IoU, and
    unmatched_cur holds the current boxes that found no partner (candidate new
    tracks).
    """
    pairs = []
    for i, pb in enumerate(prev_boxes):
        for j, cb in enumerate(cur_boxes):
            iou = box_iou(pb, cb)
            if iou >= iou_threshold:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)

    matches, used_prev, used_cur = [], set(), set()
    for _, i, j in pairs:
        if i in used_prev or j in used_cur:
            continue
        matches.append((i, j))
        used_prev.add(i)
        used_cur.add(j)
    unmatched_cur = [j for j in range(len(cur_boxes)) if j not in used_cur]
    return matches, unmatched_cur


def danger_from_ttc(ttc, ttc_safe=DANGER_TTC_SAFE, ttc_critical=DANGER_TTC_CRITICAL):
    """Map a time-to-collision (seconds) to a danger value in [0, 1].

    At or above ``ttc_safe`` the value is 0, at or below ``ttc_critical`` it is
    1, and it is linear in between. A non-finite TTC (a receding or stationary
    vehicle) maps to 0.
    """
    if not np.isfinite(ttc) or ttc >= ttc_safe:
        return 0.0
    if ttc <= ttc_critical:
        return 1.0
    return float((ttc_safe - ttc) / (ttc_safe - ttc_critical))


def danger_color(value):
    """BGR colour for a danger value: green (0) -> yellow (0.5) -> red (1)."""
    v = float(np.clip(value, 0.0, 1.0))
    if v <= 0.5:
        return (0, 255, int(round(255 * (v / 0.5))))
    return (0, int(round(255 * (1 - (v - 0.5) / 0.5))), 255)


def _track_step(size, prev_size, prev_growth, gap, fps, ema):
    """One step of the per-track growth estimate, shared by both danger paths.

    Kept as a module-level function so the detector path (BoxMomentumTracker)
    and the annotation path (danger_from_gt) cannot drift apart: both compute
    size = sqrt(w*h), the frame-to-frame change in its logarithm scaled by the
    frame rate, the EMA smoothing of that change, and the implied time to
    collision from 1/growth. Returns (growth, ttc).
    """
    growth = prev_growth
    if prev_size and size > 0:
        inst = float(np.log(size / prev_size)) * fps / max(1, gap)
        growth = inst if prev_growth is None else ema * prev_growth + (1.0 - ema) * inst
    ttc = (1.0 / growth) if (growth is not None and growth > 0) else float("inf")
    return growth, ttc


class BoxMomentumTracker:
    """Link boxes across frames and score each vehicle by its box growth.

    Deliberately lightweight: greedy IoU association plus an EMA-smoothed
    log-size slope. Each :meth:`update` call returns one dict per current box
    with the track id, the box, the smoothed growth rate (1/s, ~ 1/TTC), the
    implied time-to-collision, and the danger value in [0, 1] (``None`` until
    the track has been seen ``min_frames`` times, so a single stray detection
    cannot produce a danger score).

    Note: growth measures approach to the *camera*. On drone footage that is
    the drone; on a vehicle-mounted camera it is the ego vehicle.
    """

    def __init__(self, fps=DEFAULT_FPS, iou_threshold=DANGER_IOU, ema=DANGER_EMA,
                 min_frames=DANGER_MIN_FRAMES, max_age=DANGER_MAX_AGE,
                 ttc_safe=DANGER_TTC_SAFE, ttc_critical=DANGER_TTC_CRITICAL):
        self.fps = float(fps)
        self.iou_threshold = iou_threshold
        self.ema = ema
        self.min_frames = min_frames
        self.max_age = max_age
        self.ttc_safe = ttc_safe
        self.ttc_critical = ttc_critical
        self.tracks = {}            # track id -> state dict
        self.next_id = 1
        self.frame = -1

    def update(self, boxes, frame=None):
        """Process one frame's boxes ``[(x1, y1, x2, y2), ...]``.

        Returns one dict per input box with keys ``track_id``, ``box``,
        ``size``, ``growth_rate``, ``ttc_seconds``, ``danger``, ``hits``.
        """
        boxes = [tuple(float(v) for v in b[:4]) for b in boxes]
        self.frame = self.frame + 1 if frame is None else int(frame)

        ids = list(self.tracks)
        matches, unmatched_cur = associate_boxes(
            [self.tracks[i]["box"] for i in ids], boxes, self.iou_threshold)

        results = []
        for i, j in matches:
            results.append(self._update_track(ids[i], boxes[j]))
        for j in unmatched_cur:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"box": boxes[j], "size": None, "growth": None,
                                "hits": 0, "last": self.frame}
            results.append(self._update_track(tid, boxes[j]))

        stale = [t for t, tr in self.tracks.items()
                 if self.frame - tr["last"] > self.max_age]
        for tid in stale:
            del self.tracks[tid]
        return results

    def _update_track(self, tid, box):
        tr = self.tracks[tid]
        w = max(0.0, box[2] - box[0])
        h = max(0.0, box[3] - box[1])
        size = float(np.sqrt(w * h))

        gap = max(1, self.frame - tr["last"])
        growth, ttc = _track_step(size, tr["size"], tr["growth"], gap,
                                  self.fps, self.ema)
        tr["growth"] = growth
        tr["box"] = box
        if size > 0:
            tr["size"] = size
        tr["last"] = self.frame
        tr["hits"] += 1

        danger = (danger_from_ttc(ttc, self.ttc_safe, self.ttc_critical)
                  if tr["hits"] >= self.min_frames else None)
        return {"track_id": tid, "box": box, "size": size, "growth_rate": growth,
                "ttc_seconds": ttc, "danger": danger, "hits": tr["hits"]}


def _iter_frames(source):
    """Yield ``(index, frame)`` from a video file or a directory of images.

    A directory is searched recursively and its frames are ordered by path, so
    both a flat folder and a VisDrone-style ``clip/uavNNNN_v/*.jpg`` layout work.
    """
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    if os.path.isdir(source):
        paths = []
        for root, _, files in os.walk(source):
            paths += [os.path.join(root, f) for f in files
                      if f.lower().endswith(exts)]
        for i, path in enumerate(sorted(paths)):
            frame = cv2.imread(path)
            if frame is not None:
                yield i, frame
        return

    cap = cv2.VideoCapture(source)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield i, frame
        i += 1
    cap.release()


def analyze_danger_video(model, source, output_avi="data_out/danger/danger_results.avi",
                         csv_path="data_out/danger/danger_tracks.csv",
                         conf=0.4, imgsz=640, fps=DEFAULT_FPS, classes=None,
                         max_frames=None, x1=None, x2=None,
                         iou_threshold=DANGER_IOU, ema=DANGER_EMA,
                         min_frames=DANGER_MIN_FRAMES,
                         ttc_safe=DANGER_TTC_SAFE, ttc_critical=DANGER_TTC_CRITICAL):
    """Per-vehicle danger values from bounding-box growth, with CSV + video.

    Stages: YOLO detection -> greedy IoU association -> per-track smoothed
    log-size growth rate -> danger value in [0, 1] (see ``BoxMomentumTracker``
    and the looming identity at the top of this section).

    ``source`` may be a video file or a directory of frames. Writes:

      * ``output_avi`` - boxes drawn green -> red by danger value, labelled with
        the value and the implied TTC;
      * ``csv_path``   - one row per (frame, track): box, size, growth rate,
        TTC and danger value.

    Returns the results as a pandas DataFrame. The danger value is *relative*
    (no metric distance is recovered), and growth measures approach to the
    camera, so keep the camera still for clean readings.
    """
    if os.path.dirname(output_avi):
        os.makedirs(os.path.dirname(output_avi), exist_ok=True)
    if csv_path and os.path.dirname(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    tracker = BoxMomentumTracker(fps=fps, iou_threshold=iou_threshold, ema=ema,
                                 min_frames=min_frames, ttc_safe=ttc_safe,
                                 ttc_critical=ttc_critical)
    writer, rows, n_frames = None, [], 0

    for _, frame in _iter_frames(source):
        if max_frames is not None and n_frames >= max_frames:
            break

        detection_frame = frame
        if x1 is not None or x2 is not None:
            detection_frame = frame.copy()
            if x1 is not None:
                detection_frame[:x1, :] = 0
            if x2 is not None:
                detection_frame[x2:, :] = 0

        results = model.predict(detection_frame, imgsz=imgsz, conf=conf,
                                classes=classes, verbose=False)
        boxes = [tuple(float(v) for v in b) for b in results[0].boxes.xyxy]
        tracks = tracker.update(boxes, frame=n_frames)

        annotated = frame.copy()
        dangers = []
        for t in tracks:
            d = t["danger"]
            colour = danger_color(d if d is not None else 0.0)
            bx1, by1, bx2, by2 = (int(v) for v in t["box"])
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), colour,
                          3 if (d is not None and d >= 0.6) else 1)
            if d is not None:
                dangers.append(d)
                label = f"#{t['track_id']} d={d:.2f}"
                if np.isfinite(t["ttc_seconds"]):
                    label += f" ttc={t['ttc_seconds']:.1f}s"
                cv2.putText(annotated, label, (bx1, max(by1 - 6, 12)), FONT, 0.5,
                            colour, 1, cv2.LINE_AA)
            rows.append({
                "frame": n_frames, "track_id": t["track_id"],
                "x1": t["box"][0], "y1": t["box"][1],
                "x2": t["box"][2], "y2": t["box"][3],
                "width": t["box"][2] - t["box"][0],
                "height": t["box"][3] - t["box"][1],
                "area": (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]),
                "size": t["size"], "growth_rate": t["growth_rate"],
                "ttc_seconds": t["ttc_seconds"], "danger": d, "hits": t["hits"],
            })

        max_danger = max(dangers) if dangers else 0.0
        label = (f"frame {n_frames} | vehicles {len(tracks)} | "
                 f"max danger {max_danger:.2f}")
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 2)
        cv2.rectangle(annotated, (0, 0), (tw + 20, th + 24), BACKGROUND_COLOR, -1)
        cv2.putText(annotated, label, (10, th + 8), FONT, FONT_SCALE, FONT_COLOR,
                    2, cv2.LINE_AA)

        if writer is None:
            ext = os.path.splitext(output_avi)[1].lower()
            fourcc = cv2.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "XVID"))
            writer = cv2.VideoWriter(output_avi, fourcc, fps,
                                     (annotated.shape[1], annotated.shape[0]))
        writer.write(annotated)
        n_frames += 1

    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows, columns=["frame", "track_id", "x1", "y1", "x2", "y2",
                                     "width", "height", "area", "size",
                                     "growth_rate", "ttc_seconds", "danger", "hits"])
    if csv_path:
        df.to_csv(csv_path, index=False)

    scored = pd.to_numeric(df["danger"], errors="coerce").dropna()
    print("=" * 70)
    print("DANGER ANALYSIS")
    print("=" * 70)
    print(f"frames processed : {n_frames}")
    print(f"boxes scored     : {len(df)} rows over "
          f"{df['track_id'].nunique() if not df.empty else 0} tracks")
    if len(scored):
        print(f"danger values    : mean {scored.mean():.3f}, max {scored.max():.3f}")
        print(f"high danger (>=0.6) rows: {int((scored >= 0.6).sum())}")
    else:
        print(f"danger values    : none reported (a track needs >= {min_frames} frames)")
    if csv_path:
        print("track csv        :", csv_path)
    print("annotated video  :", output_avi)
    return df


# ---------------------------------------------------------------------------
# Danger summaries and summary figures (notebook section 7)
# ---------------------------------------------------------------------------


def load_mot_boxes(gt_path, classes=(4, 5, 6, 9)):
    """Read a VisDrone MOT ground-truth file into {frame: [(track id, box), ...]}.

    MOT rows are (frame, target id, x, y, w, h, score, category, ...), giving
    the box as a top-left corner plus a size, unlike the detector's corner form;
    the boxes returned here are (x1, y1, x2, y2). ``classes=None`` keeps every
    row.

    VisDrone numbers its frames from 1 (frame 1 is the first image) while the
    pipeline's detection CSVs number them from 0, so each MOT frame id is
    shifted down by one here to align the annotation with the detections of
    the same image.
    """
    frames = {}
    for line in Path(gt_path).read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 8:
            continue
        try:
            frame, tid = int(parts[0]) - 1, int(parts[1])
            x, y, w, h = (float(v) for v in parts[2:6])
            cls = int(parts[7])
        except ValueError:
            continue
        if classes is not None and cls not in classes:
            continue
        frames.setdefault(frame, []).append((tid, (x, y, x + w, y + h)))
    return frames


def danger_from_gt(gt_path, classes=(4, 5, 6, 9), fps=DEFAULT_FPS, ema=DANGER_EMA,
                   min_frames=DANGER_MIN_FRAMES, ttc_safe=DANGER_TTC_SAFE,
                   ttc_critical=DANGER_TTC_CRITICAL):
    """Per-vehicle danger values computed from the annotations, not the detector.

    Runs the same size -> growth -> EMA -> TTC -> danger steps as
    BoxMomentumTracker, through the shared ``_track_step``, over the
    ground-truth boxes instead of the detected ones: the two results therefore
    differ only in the boxes they were computed from. Track identities come from
    the annotation, so no association step is needed.

    Returns a DataFrame with the same columns as the CSV that
    analyze_danger_video writes: frame, track_id, x1, y1, x2, y2, width, height,
    area, size, growth_rate, ttc_seconds, danger, hits.
    """
    columns = ["frame", "track_id", "x1", "y1", "x2", "y2", "width", "height",
               "area", "size", "growth_rate", "ttc_seconds", "danger", "hits"]
    frames = load_mot_boxes(gt_path, classes)
    state, rows = {}, []
    for frame in sorted(frames):
        for tid, box in frames[frame]:
            tr = state.setdefault(tid, {"size": None, "growth": None,
                                        "last": frame, "hits": 0})
            w = max(0.0, box[2] - box[0])
            h = max(0.0, box[3] - box[1])
            size = float(np.sqrt(w * h))
            growth, ttc = _track_step(size, tr["size"], tr["growth"],
                                      max(1, frame - tr["last"]), fps, ema)
            tr["growth"] = growth
            if size > 0:
                tr["size"] = size
            tr["last"] = frame
            tr["hits"] += 1
            rows.append({
                "frame": frame, "track_id": tid,
                "x1": box[0], "y1": box[1], "x2": box[2], "y2": box[3],
                "width": w, "height": h, "area": w * h, "size": size,
                "growth_rate": growth, "ttc_seconds": ttc,
                "danger": (danger_from_ttc(ttc, ttc_safe, ttc_critical)
                           if tr["hits"] >= min_frames else float("nan")),
                "hits": tr["hits"]})
    return pd.DataFrame(rows, columns=columns)


def validate_danger_against_gt(det_csv, gt_path, iou_threshold=0.5,
                               classes=(4, 5, 6, 9), fps=DEFAULT_FPS,
                               threshold=0.6, out_csv=None):
    """Check the detector's danger values against the same values from annotations.

    In every frame, each annotated vehicle is matched to the detected track with
    the highest IoU above ``iou_threshold``, and the two danger values are
    paired. The report says how far the detector's value can be trusted and
    where it comes from:

      coverage_gt    share of annotated vehicle-frames that had a matching
                     detection at all (the danger path's version of recall);
      coverage_det   share of the detector's scored rows that sat on a real
                     vehicle (its version of precision);
      pearson_r      correlation between the paired values;
      mae            mean absolute difference between them;
      gt_high, det_high  number of matched pairs the annotation, respectively
                     the detector, calls at or above ``threshold``;
      flag_precision, flag_recall  how well the detector's high-danger call
                     reproduces the annotation's, so the verdict does not rest
                     on the majority-zero class;
      pct_agreement  share of matched pairs where both agree on that call.

    Returns (summary dict, matched pairs DataFrame). The pairs are written to
    ``out_csv`` when given.
    """
    det = pd.read_csv(det_csv)
    det["danger"] = pd.to_numeric(det["danger"], errors="coerce").fillna(0.0)
    gt = danger_from_gt(gt_path, classes=classes, fps=fps)
    gt["danger"] = pd.to_numeric(gt["danger"], errors="coerce")

    det_by_frame = {int(f): g for f, g in det.groupby("frame")}
    gt_by_frame = {int(f): g for f, g in gt.groupby("frame")}

    pairs = []
    for frame, g_rows in gt_by_frame.items():
        d_rows = det_by_frame.get(frame)
        if d_rows is None:
            continue
        det_boxes = [(int(t), b, float(d))
                     for t, b, d in zip(d_rows["track_id"],
                                        d_rows[["x1", "y1", "x2", "y2"]].to_numpy(float),
                                        d_rows["danger"])]
        for _, g_row in g_rows.iterrows():
            g_box = g_row[["x1", "y1", "x2", "y2"]].to_numpy(float)
            g_danger = g_row["danger"]
            if pd.isna(g_danger):
                continue
            best, best_iou = None, 0.0
            for tid, d_box, d_danger in det_boxes:
                value = box_iou(g_box, d_box)
                if value > best_iou:
                    best, best_iou = (tid, d_danger), value
            if best is None or best_iou < iou_threshold:
                continue
            pairs.append({"frame": int(frame), "gt_track": int(g_row["track_id"]),
                          "det_track": int(best[0]), "iou": float(best_iou),
                          "danger_gt": float(g_danger), "danger_det": float(best[1])})
    pdf = pd.DataFrame(pairs)

    gt_scored = int(gt["danger"].notna().sum())
    det_scored = int((det["hits"] >= DANGER_MIN_FRAMES).sum())
    matched_det_rows = len({(int(r["frame"]), int(r["det_track"]))
                            for _, r in pdf.iterrows()}) if len(pdf) else 0
    summary = {
        "clip": Path(det_csv).parent.name,
        "gt_scored_rows": gt_scored,
        "det_scored_rows": det_scored,
        "matched_gt_rows": len(pdf),
        "matched_det_rows": matched_det_rows,
        "coverage_gt": (len(pdf) / gt_scored) if gt_scored else float("nan"),
        "coverage_det": (matched_det_rows / det_scored) if det_scored else float("nan"),
        "pearson_r": float("nan"),
        "mae": float("nan"),
        "gt_high": 0,
        "det_high": 0,
        "both_high": 0,
        "flag_precision": float("nan"),
        "flag_recall": float("nan"),
        "pct_agreement": float("nan"),
    }
    if len(pdf):
        g, d = pdf["danger_gt"].to_numpy(float), pdf["danger_det"].to_numpy(float)
        g_high, d_high = g >= threshold, d >= threshold
        summary["mae"] = float(np.mean(np.abs(g - d)))
        if len(pdf) > 1 and g.std() > 0 and d.std() > 0:
            summary["pearson_r"] = float(np.corrcoef(g, d)[0, 1])
        summary["gt_high"] = int(g_high.sum())
        summary["det_high"] = int(d_high.sum())
        summary["both_high"] = int((g_high & d_high).sum())
        if d_high.sum():
            summary["flag_precision"] = float((g_high & d_high).sum() / d_high.sum())
        if g_high.sum():
            summary["flag_recall"] = float((g_high & d_high).sum() / g_high.sum())
        summary["pct_agreement"] = 100 * float((g_high == d_high).mean())
    if out_csv and len(pdf):
        pdf.to_csv(out_csv, index=False)
        print("matched pairs    :", out_csv)
    return summary, pdf


# ---------------------------------------------------------------------------
# TP / FP / FN correspondence (the paper's Table 1, notebook section 7.3)
# ---------------------------------------------------------------------------


def matching_stats(gt_path, det_csv, iou_threshold=0.5, classes=(4, 5, 6, 9)):
    """One-to-one matching of the detected boxes against the annotated vehicles.

    In every frame, each detection is paired with the annotated vehicle it
    overlaps most, taking candidate pairs in order of descending overlap and
    using each box at most once; a pair counts only if its overlap reaches
    ``iou_threshold``. A matched pair is a true positive, a detection left
    over is a false positive, and an annotated vehicle left over is a false
    negative. Because every detection and every annotation ends up in exactly
    one of the three, the over- or under-count of a frame is exactly the
    frame's false positives minus its false negatives.

    Returns (summary dict, per-frame DataFrame). The summary holds frames,
    tp, fp, fn, precision, recall and net_per_frame (false positives minus
    false negatives over the whole clip, i.e. the mean over- or under-count
    per frame). The DataFrame has one row per frame with gt, det, tp, fp, fn.
    """
    gt = load_mot_boxes(gt_path, classes)
    det = pd.read_csv(det_csv)
    det_by_frame = {int(f): g for f, g in det.groupby("frame")}

    rows = []
    for frame in sorted(set(gt) | set(det_by_frame)):
        g_boxes = [b for _, b in gt.get(frame, [])]
        d_group = det_by_frame.get(frame)
        d_boxes = ([] if d_group is None else
                   [tuple(b) for b in d_group[["x1", "y1", "x2", "y2"]].to_numpy(float)])
        pairs = sorted(((box_iou(gb, db), i, j)
                        for i, gb in enumerate(g_boxes)
                        for j, db in enumerate(d_boxes)), reverse=True)
        used_g, used_d, tp = set(), set(), 0
        for value, i, j in pairs:
            if value < iou_threshold or i in used_g or j in used_d:
                continue
            used_g.add(i)
            used_d.add(j)
            tp += 1
        rows.append({"frame": frame, "gt": len(g_boxes), "det": len(d_boxes),
                     "tp": tp, "fp": len(d_boxes) - tp, "fn": len(g_boxes) - tp})

    per_frame = pd.DataFrame(rows)
    n = len(per_frame)
    total = per_frame[["tp", "fp", "fn"]].sum()
    summary = {
        "frames": n,
        "tp": int(total["tp"]),
        "fp": int(total["fp"]),
        "fn": int(total["fn"]),
        "precision": float(total["tp"] / (total["tp"] + total["fp"])),
        "recall": float(total["tp"] / (total["tp"] + total["fn"])),
        "net_per_frame": float((total["fp"] - total["fn"]) / n),
    }
    return summary, per_frame


# Figure sizes chosen to match the paper's existing PNGs at dpi=150.
DANGER_FIGSIZE = {"distribution": (11.0, 3.2),
                  "per_frame_max": (10.0, 5.4),
                  "vs_track_length": (11.0, 3.4)}


def _danger_csvs(csv_dir, clips):
    """Resolve <csv_dir>/<clip>/danger_<clip>.csv, noting any clip that is absent."""
    found = []
    for clip in clips:
        p = Path(csv_dir) / clip / f"danger_{clip}.csv"
        if p.exists():
            found.append((clip, p))
        else:
            print(f"(no danger csv for '{clip}': {p} - run analyze_danger_video first)")
    return found


def _track_peaks(df):
    """Per-track length (observations), peak danger and the track ids."""
    length = df.groupby("track_id").size()
    peak = df.groupby("track_id")["danger"].max()
    return length, peak


def _episodes(values, threshold, min_frames):
    """Runs of >= min_frames consecutive observations at or above threshold.

    A frame the tracker missed does not break a run: an episode counts
    consecutive *observations* of one vehicle, not consecutive frame numbers.
    Returns (number of episodes, longest run).
    """
    episodes, longest, run = 0, 0, 0
    for value in values:
        if value >= threshold:
            run += 1
            if run > longest:
                longest = run
        else:
            if run >= min_frames:
                episodes += 1
            run = 0
    if run >= min_frames:
        episodes += 1
    return episodes, longest


def summarise_danger(csv_dir, clips=("day", "night", "heavy"), threshold=0.6,
                     min_frames=5, long_track=10, short_track=5, out_csv=None):
    """One summary row per clip for the danger values written by analyze_danger_video.

    Reads <csv_dir>/<clip>/danger_<clip>.csv. Two definitions, both used by the
    paper, are fixed here so the numbers can be checked:

      * a *scored* row is a row whose track had at least DANGER_MIN_FRAMES hits,
        the point at which a danger value is first defined. Every percentage in
        the danger table is a share of scored rows, because an unscored row
        carries no value to threshold;
      * an *episode* is a run of at least ``min_frames`` consecutive
        observations of one vehicle at or above ``threshold``; a frame the
        tracker missed does not break a run.

    Writes the table to ``out_csv`` when given. Returns it as a DataFrame.
    """
    rows_out = []
    for clip, path in _danger_csvs(csv_dir, clips):
        df = pd.read_csv(path)
        df["danger"] = pd.to_numeric(df["danger"], errors="coerce").fillna(0.0)
        scored = df[df["hits"] >= DANGER_MIN_FRAMES]
        length, peak = _track_peaks(df)
        n_long = length.index[length >= long_track]
        n_short = length.index[length < short_track]

        episodes, longest = 0, 0
        for _, sub in df.groupby("track_id"):
            e, l = _episodes(sub["danger"].tolist(), threshold, min_frames)
            episodes += e
            longest = max(longest, l)

        rows_out.append({
            "clip": clip,
            "frames": int(df["frame"].max()) + 1,
            "rows": len(df),
            "rows_scored": len(scored),
            "tracks": len(length),
            "tracks_scored": int((length >= DANGER_MIN_FRAMES).sum()),
            "mean_track_len": float(length.mean()),
            "median_track_len": float(length.median()),
            "veh_per_frame": len(df) / (int(df["frame"].max()) + 1),
            "danger_mean": float(scored["danger"].mean()),
            "danger_median": float(scored["danger"].median()),
            "danger_p90": float(scored["danger"].quantile(0.90)),
            "danger_max": float(scored["danger"].max()),
            "pct_rows_ge06": 100 * float((scored["danger"] >= threshold).mean()),
            "pct_rows_gt0": 100 * float((scored["danger"] > 0).mean()),
            "pct_rows_approaching":
                100 * float((pd.to_numeric(scored["growth_rate"], errors="coerce") > 0).mean()),
            "tracks_ge06": int((peak >= threshold).sum()),
            # share of the *scored* tracks, matching the paper's denominator
            "pct_tracks_ge06": (100 * float((peak >= threshold).sum())
                                / int((length >= DANGER_MIN_FRAMES).sum())),
            "tracks_eq1": int((peak >= 0.9999).sum()),
            "pct_tracks_ge10_ge06": 100 * float((peak[n_long] >= threshold).mean()) if len(n_long) else float("nan"),
            "pct_tracks_lt5_ge06": 100 * float((peak[n_short] >= threshold).mean()) if len(n_short) else float("nan"),
            "median_peak_ge10": float(peak[n_long].median()) if len(n_long) else float("nan"),
            "episodes_ge5": episodes,
            "longest_run": longest,
        })
    summary = pd.DataFrame(rows_out)
    if out_csv and len(summary):
        summary.to_csv(out_csv, index=False)
        print("danger summary   :", out_csv)
    return summary


def plot_danger_figures(csv_dir, out_dir=None, clips=("day", "night", "heavy"),
                        threshold=0.6, dpi=150, show=False):
    """The three summary figures for the danger values, saved as PNGs.

    Reads the same <csv_dir>/<clip>/danger_<clip>.csv files as
    summarise_danger() and writes, into ``out_dir``:

      danger_distribution.png    danger-value histogram, one panel per clip
      danger_per_frame_max.png   highest danger value in each frame over time
      danger_vs_track_length.png peak danger against how long the track lasted

    Every panel draws the ``threshold`` (0.6 by default) as a reference line.
    Returns the list of written paths.
    """
    import matplotlib.pyplot as plt

    clips_csv = _danger_csvs(csv_dir, clips)
    if not clips_csv:
        return []
    data = {clip: pd.read_csv(path) for clip, path in clips_csv}
    for df in data.values():
        df["danger"] = pd.to_numeric(df["danger"], errors="coerce").fillna(0.0)
    out_dir = Path(out_dir) if out_dir is not None else Path(csv_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    def _save(fig, name):
        p = out_dir / name
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        written.append(p)
        print("saved figure     :", p)

    # 1) distribution of the danger values
    fig, axes = plt.subplots(1, len(data), figsize=DANGER_FIGSIZE["distribution"],
                             squeeze=False)
    for ax, (clip, df) in zip(axes[0], data.items()):
        scored = df[df["hits"] >= DANGER_MIN_FRAMES]
        ax.hist(scored["danger"], bins=20, range=(0, 1), color="tab:blue", alpha=0.75)
        ax.axvline(threshold, color="tab:red", ls="--", lw=1)
        ax.set(title=clip, xlabel="danger value", ylabel="vehicle-frame rows")
        ax.grid(alpha=0.3)
    fig.suptitle("Distribution of danger values")
    fig.tight_layout()
    _save(fig, "danger_distribution.png")

    # 2) highest danger value in each frame
    fig, axes = plt.subplots(len(data), 1, figsize=DANGER_FIGSIZE["per_frame_max"],
                             squeeze=False, sharex=False)
    for ax, (clip, df) in zip(axes[:, 0], data.items()):
        per_frame = df.groupby("frame")["danger"].max()
        ax.plot(per_frame.index, per_frame.values, lw=1, color="tab:blue")
        ax.axhline(threshold, color="tab:red", ls="--", lw=1)
        ax.set(title=clip, ylabel="max danger", ylim=(0, 1.02))
        ax.grid(alpha=0.3)
    axes[-1, 0].set(xlabel="frame")
    fig.suptitle("Highest danger value per frame")
    fig.tight_layout()
    _save(fig, "danger_per_frame_max.png")

    # 3) peak danger against track length
    fig, axes = plt.subplots(1, len(data), figsize=DANGER_FIGSIZE["vs_track_length"],
                             squeeze=False)
    for ax, (clip, df) in zip(axes[0], data.items()):
        length, peak = _track_peaks(df)
        ax.scatter(length.values, peak.values, s=12, alpha=0.5, color="tab:blue")
        ax.axhline(threshold, color="tab:red", ls="--", lw=1)
        ax.set(title=clip, xlabel="track length (frames)", ylabel="peak danger",
               ylim=(-0.02, 1.02))
        ax.grid(alpha=0.3)
    fig.suptitle("Peak danger value against track length")
    fig.tight_layout()
    _save(fig, "danger_vs_track_length.png")

    if show:
        plt.show()
    return written
