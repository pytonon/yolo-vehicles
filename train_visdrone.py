#!/usr/bin/env python3
"""
train_visdrone.py — Fine-tune a YOLO26 model on the Ultralytics VisDrone-DET
dataset (vehicle classes only: car, van, truck, bus -> single class 0).

Standalone on purpose: upload THIS ONE FILE to Google Colab (or run locally)
and execute it. First run downloads the VisDrone dataset (~2 GB, cached
afterwards), then trains on the best available device (CUDA on Colab, MPS on
Apple Silicon, CPU otherwise).

Output:
  * best_visdrone.pt  — trained weights in the current directory
                       (auto-downloaded on Colab when training finishes)
  * a final metrics summary (precision / recall / mAP50 / mAP50-95)

Usage on Colab:  upload this file, then run:  !python train_visdrone.py
Usage locally:   python train_visdrone.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration — edit here if you want different settings
# ---------------------------------------------------------------------------
MODEL_NAME = "yolo26s.pt"   # pretrained base: yolo26n / s / m / l / x
EPOCHS = 30
IMGSZ = 640                 # 1280 helps for VisDrone's small objects, but is slower
BATCH = 8                   # fine on a Colab T4 / M1; lower on low-RAM machines
WORKERS = 4
PATIENCE = 20
VEHICLE_ONLY = True         # keep only car/van/truck/bus as a single class 0
DATA_ROOT = "datasets/VisDrone"
PROJECT = "train_visdrone"
RUN_NAME = "train"
OUTPUT_WEIGHTS = "best_visdrone.pt"
# ---------------------------------------------------------------------------

VEHICLE_CLASSES = {3, 4, 5, 8}   # VisDrone YOLO numbering: car, van, truck, bus


def ensure_ultralytics():
    """Make sure a YOLO26-capable ultralytics is installed (>= 8.4.0)."""
    try:
        import ultralytics
        if tuple(map(int, ultralytics.__version__.split(".")[:2])) < (8, 4):
            raise ImportError("ultralytics too old for YOLO26")
        return ultralytics
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "-U", "ultralytics"]
        )
        import ultralytics
        return ultralytics


def filter_visdrone_labels(src_dir, dst_dir, classes=VEHICLE_CLASSES):
    """Copy YOLO labels from src_dir to dst_dir, keeping only `classes`
    (renumbered to a single class 0). Returns the number of labels written."""
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for f in src_dir.glob("*.txt"):
        kept = []
        for line in f.read_text().splitlines():
            t = line.split()
            if t and int(t[0]) in classes:
                kept.append("0 " + " ".join(t[1:]))
        if kept:
            (dst_dir / f.name).write_text("\n".join(kept) + "\n")
            written += 1
    return written



def _build_vehicle_dataset(dataset, veh_root):
    """Build (or refresh) the vehicle-only dataset.

    IMPORTANT: images are HARD-LINKED into a real directory (no symlinks).
    Ultralytics resolves symlinked image dirs to their real path and then reads
    the labels next to THAT path - i.e. the original unfiltered 10-class
    labels. With nc=1 every non-vehicle label is then flagged 'corrupt' and
    the image is silently dropped (this caused training on ~100 images).
    Hard links are instant and use no extra disk (same filesystem).
    """
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
        n = filter_visdrone_labels(src_lbl, dst_lbl)
        print(f"  {split}: kept {n} label files (vehicles only)")

        # --- images: real dir of hard links, rebuilt fresh each run ---
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
        print(f"  {split}: linked {n_links} images")


def _dataset_counts(veh_root):
    counts = {}
    for split in ("train", "val"):
        img_dir = veh_root / "images" / split
        lbl_dir = veh_root / "labels" / split
        counts[split] = (len(list(img_dir.glob("*.jpg"))), len(list(lbl_dir.glob("*.txt"))))
        print(f"  {split}: {counts[split][0]} images, {counts[split][1]} vehicle label files")
    return counts


def prepare_visdrone_dataset():
    """Download VisDrone-DET via Ultralytics (~2 GB once) and build the
    vehicle-only single-class variant. Returns the dataset YAML path."""
    from ultralytics.data.utils import check_det_dataset

    print("Checking VisDrone dataset (first run downloads ~2 GB, cached afterwards)...")
    dataset = check_det_dataset("VisDrone.yaml")    # download + YOLO conversion

    if not VEHICLE_ONLY:
        return Path("VisDrone.yaml")

    data_root = Path(DATA_ROOT).resolve()
    veh_root = data_root.parent / "VisDroneVehicle"

    _build_vehicle_dataset(dataset, veh_root)
    counts = _dataset_counts(veh_root)

    min_ok = {"train": (100, 10), "val": (50, 10)}
    if any(counts[s][0] < min_ok[s][0] or counts[s][1] < min_ok[s][1] for s in ("train", "val")):
        print("Dataset looks incomplete or stale - rebuilding from scratch ...")
        import shutil
        shutil.rmtree(veh_root, ignore_errors=True)
        _build_vehicle_dataset(dataset, veh_root)
        counts = _dataset_counts(veh_root)
        if any(counts[s][0] < min_ok[s][0] or counts[s][1] < min_ok[s][1] for s in ("train", "val")):
            raise RuntimeError(
                "vehicle dataset is still too small after rebuild - "
                f"train={counts['train']}, val={counts['val']}. "
                "Check the VisDrone download under datasets/VisDrone")

    yaml_path = veh_root / "visdrone_vehicle.yaml"
    yaml_path.write_text(
        f"path: {veh_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: vehicle\n"
    )
    print(f"Dataset ready: train {counts['train'][0]} images / val {counts['val'][0]} images -> {yaml_path}")
    return yaml_path


def main():
    ultralytics = ensure_ultralytics()

    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Ultralytics {ultralytics.__version__} | device: {device}")

    data_yaml = prepare_visdrone_dataset()

    print(f"\nTraining {MODEL_NAME} on {data_yaml} ({EPOCHS} epochs, imgsz={IMGSZ}) ...")
    model = YOLO(MODEL_NAME)
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        workers=WORKERS,
        patience=PATIENCE,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
    )

    best = Path(getattr(results, "save_dir", None) or Path(PROJECT) / RUN_NAME) / "weights" / "best.pt"
    if not best.exists():
        raise RuntimeError(f"trained weights not found at {best}")
    shutil.copy(best, OUTPUT_WEIGHTS)
    print("\nTrained weights saved to:", Path(OUTPUT_WEIGHTS).resolve())
    print("Training outputs in:", best.parent.parent)

    print("\n===== Final metrics =====")
    for key, value in results.results_dict.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # On Colab: try to download the weights automatically when training
    # finishes. files.download() needs a live notebook kernel, so when the
    # script runs via !python (or on non-Colab) it fails - that is fine, the
    # weights are already saved next to this script either way.
    try:
        from google.colab import files  # noqa: F401
        print("\nColab detected - downloading", OUTPUT_WEIGHTS)
        files.download(str(Path(OUTPUT_WEIGHTS).resolve()))
        print("Download started - check your browser downloads.")
    except Exception as e:
        print("\nAuto-download unavailable (", type(e).__name__, "-", e, ")")
        print("Your trained weights are safe at:", Path(OUTPUT_WEIGHTS).resolve())
        print("Download them from a notebook cell instead:")
        print(f'  from google.colab import files; files.download("/content/{OUTPUT_WEIGHTS}")')
        print("(or right-click the file in the Files panel on the left)")


if __name__ == "__main__":
    main()
