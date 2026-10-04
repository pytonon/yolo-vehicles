# YOLO26 Vehicle Detection & Bounding-Box Danger Values — Drone Traffic Analysis

Reference implementation for a high-school research paper: fine-tune a
**YOLO26s** detector on drone footage (the Ultralytics VisDrone-DET dataset),
count vehicles per frame, and — the paper's core contribution — assign each
vehicle a **danger value** from how quickly its bounding box grows as it
approaches the camera. The system is evaluated on three human-annotated
VisDrone clips covering day, night and heavy traffic.

A bounding box that is growing is a vehicle that is closing in: for a
(nearly) stationary camera, the relative growth rate of a box is the visual
"looming" cue, which equals one over the time to collision. The danger value
in this repo maps that rate onto a 0–1 score (0 = safe, 1 = imminent), so a
camera-only detector can rank which vehicles are becoming dangerous without
any distance sensor.

## What it does

1. **Fine-tuning** — trains YOLO26s on Ultralytics VisDrone-DET (auto-downloaded
   on first use, ~2 GB), filtered to a single `vehicle` class
   (car / van / truck / bus), on CUDA / MPS / CPU automatically.
2. **Counting + detection quality** — counts vehicles per frame and measures how
   well the detections match the ground truth: per-clip mAP50 / mAP50-95 and
   precision / recall, plus a one-to-one true-positive / false-positive /
   false-negative correspondence (the paper's Table 1).
3. **Danger value** — follows each vehicle's box across frames and converts its
   size-growth rate into a 0–1 danger value (`analyze_danger_video`), the same
   way the paper does.
4. **Validation** — recomputes the danger value from the *annotated* boxes and
   compares the two, so the value's implementation (not just its output) is
   checked (Tables 3 and 4).

The pipeline is model-agnostic Ultralytics code — the same functions run with
the pre-trained `yolo26s.pt` (COCO) or the fine-tuned `best_visdrone.pt`.

## Repository structure

```
yolo-vehicles/
├── main_workflow26.ipynb   # step-by-step notebook (run top to bottom)
├── vd_lib26.py             # library: all functions behind the notebook
├── train_visdrone.py       # standalone Colab script for fine-tuning
├── best_visdrone.pt        # fine-tuned YOLO26s weights (included, ~19 MB)
├── yolo26s.pt              # pre-trained YOLO26s weights (included, ~19 MB)
├── requirements.txt
├── README.md
├── LICENSE                 # Apache-2.0
├── NOTICE.md               # attribution + list of changes (Apache-2.0 §4(d))
├── .gitignore
├── figures/                # paper figures written by sections 5-7
└── data_input/
    ├── sample_video_copy.mp4             # sample video (included)
    ├── test_video_visdrone/              # day clip (NOT committed)
    │   ├── uav0000077_00720_v/           #   780 frames
    │   └── uav0000077_00720_v.txt
    ├── test_video_visdrone_night/        # night clip (NOT committed)
    │   ├── uav0000119_02301_v/           #   179 frames
    │   └── uav0000119_02301_v.txt
    └── visdrone_heavy_traffic/           # heavy-traffic clip (NOT committed)
        ├── uav0000297_00000_v/           #   146 frames
        └── uav0000297_00000_v.txt
```

`data_out/` (annotated videos, CSVs, results) and `train_out26/` (training runs)
are generated locally and git-ignored. The VisDrone clips and the training
dataset are not committed — see "Test data" below.

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook main_workflow26.ipynb   # or VS Code / Colab
```

Run the cells in order. Sections 1-3 (load model, fine-tune, load the
fine-tuned weights) and section 4 (sample video) need no VisDrone data;
sections 5-7 (the paper's evaluation) need the three test clips below. The
`best_visdrone.pt` already in the repo folder was trained on Colab (30 epochs,
T4), so you can skip section 2 and run the rest directly with it.

## Data

**Training data** — Ultralytics VisDrone-DET, auto-downloaded on first use by
section 2 (see https://docs.ultralytics.com/datasets/detect/visdrone/), then
filtered to a single `vehicle` class (car / van / truck / bus).

**Test data** — three clips from the VisDrone2019-MOT benchmark (Task 4).
Download `VisDrone2019-MOT-test-dev` from the
[VisDrone dataset page](https://github.com/VisDrone/VisDrone-Dataset)
(Google Drive:
https://drive.google.com/open?id=14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu), which
contains the three sequences used here:
`uav0000077_00720_v`, `uav0000119_02301_v`, `uav0000297_00000_v`.

### Formatting a VisDrone clip for this repo

Each sequence ships as a folder of frames plus a ground-truth file in MOT
format. The notebook expects, for each clip:

```
data_input/<scenario_dir>/uavXXXXXXXX_XXXXX_v/     # the frames
data_input/<scenario_dir>/uavXXXXXXXX_XXXXX_v.txt  # the ground truth (beside the folder)
```

with `<scenario_dir>` one of `test_video_visdrone`, `test_video_visdrone_night`
or `visdrone_heavy_traffic` (see the tree above). Concretely:

1. **Unzip** the sequence; you get a folder `uavXXXXXXXX_XXXXX_v/` containing
   the frames (either directly, or in an `img1/` subfolder) and the annotations
   (either `gt.txt` in the folder, or `gt/gt.txt`).
2. **Frames** — keep them inside `uavXXXXXXXX_XXXXX_v/`. The loader searches
   recursively, so a subfolder such as `img1/` is fine. Filenames must be
   zero-padded 7-digit (`0000001.jpg` … `0000780.jpg`) so that alphabetical
   order equals frame order; VisDrone already names them this way.
3. **Annotations** — move/rename `gt/gt.txt` to `uavXXXXXXXX_XXXXX_v.txt` and
   place it *beside* the folder (not inside it).
4. Put the folder and its `.txt` under the right `data_input/<scenario_dir>/`.

The `.txt` is MOT format — 10 comma-separated values per line, no header:

```
<frame>, <id>, <x>, <y>, <w>, <h>, <score>, <class>, <truncation>, <occlusion>
```

where `x, y` is the top-left corner and `w, h` the width/height of the box.
The code reads classes `4` (car), `5` (van), `6` (truck) and `9` (bus) as
"vehicle"; every other class is ignored.

## Reproducing the paper

| Notebook section | Paper output |
|---|---|
| 5. VisDrone scenario evaluations (5.1 night, 5.2 heavy) | per-clip counts, Figure 1a-1c |
| 5.3 Detection quality | per-clip mAP50 / mAP50-95, precision, recall |
| 6. Per-vehicle danger value | danger CSVs + annotated videos |
| 6.1 Danger summary | Table 3, Figures 3-4 |
| 6.2 Checking the danger value against the annotations | Table 4 |
| 6.3 Table 1 (TP/FP/FN correspondence) | Table 1 |
| 7. Paper figures | copies every figure into `figures/` under its paper number |

## Outputs

Running the notebook produces, in `data_out/`:
- `danger/` — per-clip danger CSVs, validation CSVs, `danger_summary.csv`,
  `table1_correspondence.csv` and the paper figures.
- annotated videos (`*_sequence.mp4`, `*_count.avi`, `danger_*.avi`).

## Changes from the original notebook

`main_workflow26.ipynb` and `vd_lib26.py` are modified derivatives of the
Kaggle notebook
["Vehicles Detection and Counting"](https://www.kaggle.com/code/hakim11/vehicles-detection-and-counting)
by **hakim11** (Apache License 2.0). The main changes made in this repository:

1. **Detector upgraded** — YOLOv8 replaced with YOLO26 (`yolo26s.pt`
   pre-trained, `best_visdrone.pt` fine-tuned).
2. **Danger value added** — the core contribution: bounding-box growth is
   turned into a per-vehicle 0–1 danger value (looming → ~1/TTC), which the
   original notebook does not compute.
3. **Dataset changed** — fine-tuning moved from a top-view vehicle image set
   to Ultralytics VisDrone-DET (drone footage), collapsed to a single
   `vehicle` class.
4. **Evaluation extended** — added MOT ground-truth comparison: per-clip
   mAP50 / mAP50-95 and precision / recall, a one-to-one TP/FP/FN
   correspondence, and danger-value validation against the annotations.
5. **Refactored into a library** — the notebook's logic was extracted into
   `vd_lib26.py`; the code is model-agnostic, all absolute paths were removed,
   and the repo root is auto-detected.

## Acknowledgments

- **Ultralytics YOLO26** (AGPL-3.0) — detection/training framework and the
  pre-trained `yolo26s.pt` weights. https://github.com/ultralytics/ultralytics
- **VisDrone datasets** (VisDrone-DET and VisDrone2019-MOT) — training and
  evaluation data. Free for research use; cite:
  ```bibtex
  @article{zhu2021detection,
    title={Detection and tracking meet drones challenge},
    author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and
            Fan, Heng and Hu, Qinghua and Ling, Haibin},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    volume={44}, number={11}, pages={7380--7399}, year={2021}, publisher={IEEE}
  }
  ```
- **hakim11's Kaggle notebook** ("Vehicles Detection and Counting", Apache-2.0)
  — the notebook this repository is derived from. See NOTICE.md for the full
  attribution and list of changes.
  https://www.kaggle.com/code/hakim11/vehicles-detection-and-counting

## License

Apache License 2.0 — see [LICENSE](LICENSE). `main_workflow26.ipynb` and
`vd_lib26.py` are modified derivatives of hakim11's Apache-2.0-licensed
notebook; attribution and the list of modifications are in [NOTICE.md](NOTICE.md).
Third-party resources (Ultralytics AGPL-3.0, the VisDrone datasets) keep their
own licenses — see Acknowledgments above.

Repository: https://github.com/pytonon/yolo-vehicles
