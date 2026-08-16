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
#     plot_count_accuracy, evaluate_sequence)
#   - ffmpeg odd-dimension encoding and video-inference memory bugs fixed
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
  * load_mot_counts                   — parse MOT-style GT CSV into per-frame counts
  * frames_to_video                    — assemble a numbered-frame folder into an MP4
  * convert_video                      — transcode a video to H.264 MP4 (ffmpeg)
  * plot_count_accuracy                — accuracy-vs-frame plot against ground truth
  * evaluate_sequence                 — one-call sequence evaluation (counter + results + plot)

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
import warnings

import cv2
from ultralytics import YOLO

warnings.filterwarnings("ignore")

__all__ = [
    "pick_device",
    "load_model",
    "infer_image",
    "infer_video_save",
    "draw_info_panels",
    "count_vehicles_in_video",
    "load_mot_counts",
    "frames_to_video",
    "convert_video",
    "plot_count_accuracy",
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
                            conf=0.4, imgsz=640, fps=20.0,
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


def frames_to_video(frames_dir, output_video, fps=14):
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


def plot_count_accuracy(gt_counts, pred_counts, show=True):
    """Compare per-frame predicted counts against ground-truth counts.

    pred_counts[i] is the model's count for video frame i (GT frame id i+1,
    as returned by count_vehicles_in_video(return_counts=True)). Plots
    accuracy vs frame plus the counts comparison; prints and returns
    (frame_ids, accuracies, mean_accuracy).
    """
    import matplotlib.pyplot as plt

    frame_ids, accs = [], []
    for g, gt in sorted(gt_counts.items()):
        if 1 <= g <= len(pred_counts):
            pred = pred_counts[g - 1]
            frame_ids.append(g)
            accs.append(max(0.0, 1.0 - abs(pred - gt) / max(gt, 1)))
    if not accs:
        raise RuntimeError("no ground-truth frames matched the prediction range")

    mean_acc = sum(accs) / len(accs)
    print(f"mean count accuracy over {len(accs)} annotated frames: {mean_acc:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(frame_ids, accs, "-", lw=1, alpha=0.6, label="per frame")
    ax1.plot(frame_ids, [mean_acc] * len(frame_ids), "--", color="tab:red",
             label=f"mean = {mean_acc:.3f}")
    ax1.set(xlabel="frame", ylabel="accuracy", title="Accuracy vs frame (count match)")
    ax1.set_ylim(0, 1.05); ax1.legend(); ax1.grid(alpha=0.3)

    gt_series = [gt_counts.get(f, 0) for f in frame_ids]
    pred_series = [pred_counts[f - 1] for f in frame_ids]
    ax2.plot(frame_ids, gt_series, label="ground truth", lw=1.5)
    ax2.plot(frame_ids, pred_series, label="predicted", lw=1.5, alpha=0.8)
    ax2.set(xlabel="frame", ylabel="vehicles", title="Vehicle counts per frame")
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()
    return frame_ids, accs, mean_acc


def evaluate_sequence(model, frames_dir, gt_path, out_dir,
                      classes=(4, 5, 6, 9), conf=0.4, fps=14,
                      video_name="sequence_test.mp4",
                      count_avi="sequence_count.avi",
                      results_mp4="sequence_test_results.mp4"):
    """End-to-end sequence evaluation: ground truth -> counter -> results.

    Assembles `frames_dir` into a video, runs count_vehicles_in_video with the
    detection slice disabled (aerial frames are not 1280x720), converts the
    annotated output into a viewable MP4, and plots accuracy vs frame.

    Returns (frame_ids, accuracies, mean_accuracy) from plot_count_accuracy.
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
    return plot_count_accuracy(gt_counts, pred_counts)
