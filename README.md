# YOLO26 Vehicle Detection & Counting — Top-View Traffic Analysis

Reference implementation for a high-school research project: counting vehicles
in top-view traffic video with a fine-tuned **YOLO26** model, estimating
real-time traffic intensity, and measuring per-frame counting accuracy against
a ground-truth-annotated test sequence.

The pipeline is model-agnostic Ultralytics code — it runs identically with the
pre-trained `yolo26s.pt` (COCO) or the fine-tuned `best26.pt`.

## What it does

1. **Pre-trained model demo** — run `yolo26s.pt` on a sample image and video
   (COCO classes, no training required).
2. **Fine-tuning** — trains YOLO26s on a public top-view vehicle dataset
   (auto-downloaded from Kaggle), on CUDA / MPS / CPU automatically.
3. **Real-time counting** — `count_vehicles_in_video()` counts vehicles per
   frame with a traffic-intensity overlay ("Heavy" / "Smooth") and saves an
   annotated video.
4. **Accuracy evaluation** — `evaluate_sequence()` runs the counter on a
   ground-truth-annotated frame sequence (MOT-style `frame,id,x,y,w,h,score,
   class,truncation,occlusion` CSV) and plots per-frame count accuracy.

**Count accuracy metric:** for each annotated frame,
`accuracy = max(0, 1 - |predicted − ground_truth| / max(ground_truth, 1))`,
then averaged over all annotated frames (1.0 = exact count match).

## Repository structure

```
github_yolo-vehicles/
├── main_workflow26.ipynb   # step-by-step notebook (run top to bottom)
├── vd_lib26.py             # library: all functions behind the notebook
├── yolo26s.pt              # pre-trained weights (included, 19 MB)
├── requirements.txt
├── README.md
├── LICENSE
├── .gitignore
└── data_input/
    ├── sample_video_copy.mp4          # sample video (included)
    └── test_video_visdrone/           # NOT committed — see "Test data"
        ├── uav0000077_00720_v/        #   780 frames (0000001.jpg …)
        └── uav0000077_00720_v.txt     #   ground-truth annotations
```

`data_out/` (annotated videos, results) and `train_out26/` (training runs) are
generated locally and git-ignored.

## Quickstart

```bash
pip install -r requirements.txt
jupyter notebook main_workflow26.ipynb   # or VS Code / Colab
```

Then run the cells in order. Sections 1–2 (pre-trained demo + dataset
inspection) need no training; section 3 (fine-tuning, ~30–60 min on an M1,
faster on Colab) writes `best26.pt`; section 6 (accuracy evaluation) needs the
test data below.

## Data

- **Training data** (top-view vehicles, auto-downloaded by the notebook):
  https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset
  (free Kaggle account; `kagglehub` prompts for credentials once).
- **Test data** (VisDrone MOT clip, 780 frames): download the clip
  `uav0000077_00720_v` from the VisDrone MOT benchmark
  (https://github.com/VisDrone/VisDrone-Dataset), then place it so the layout
  matches the tree above: a folder of sequentially numbered JPGs plus the
  `.txt` ground truth. The clip is ~280 MB, which is why it is not committed.

## Outputs

Running the notebook produces, in `data_out/`:

| File | Contents |
|---|---|
| `vehicle_count.avi` | sample video with per-frame count + intensity overlay |
| `sequence_test.mp4` / `sequence_count.avi` / `sequence_test_results.mp4` | VisDrone sequence assembled + annotated |
| accuracy plot | per-frame count accuracy vs ground truth (printed + plotted) of VisDrone sequence |

## Notes

- The detection slice (`X1`/`X2` in `vd_lib26.py`) is tuned for the 1280×720
  sample camera; it is disabled automatically for the VisDrone evaluation.
- `best26.pt` (your fine-tuned weights) is git-ignored on purpose — regenerate
  it with notebook section 3.

## Acknowledgments

This project builds on the following third-party resources (each keeps its own
license; the code in this repository is Apache 2.0):

- **Ultralytics YOLO26** (AGPL-3.0) — detection/training framework and the
  pre-trained `yolo26s.pt` weights. https://github.com/ultralytics/ultralytics
- **Top-view vehicle dataset** (Kaggle) — fine-tuning data, auto-downloaded by
  the notebook. Check the dataset page for its license terms and credit the
  author in the paper:
  https://www.kaggle.com/datasets/farzadnekouei/top-view-vehicle-detection-image-dataset
- **VisDrone MOT benchmark** — ground-truth-annotated test clip. Free for
  research use; cite as:
  ```bibtex
  @article{zhu2021detection,
    title={Detection and tracking meet drones challenge},
    author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and
            Fan, Heng and Hu, Qinghua and Ling, Haibin},
    journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
    volume={44}, number={11}, pages={7380--7399}, year={2021}, publisher={IEEE}
  }
  ```
- **Original notebook** — `vd_lib26.py` and `main_workflow26.ipynb` are derived
  from `vehicles-detection-and-counting.ipynb` (Apache License 2.0).
  Attribution and the modification list are in [NOTICE.md](NOTICE.md).

## Changes from the original notebook

`main_workflow26.ipynb` and `vd_lib26.py` are derived from
`vehicles-detection-and-counting.ipynb` (Apache License 2.0). The
modifications made in this repository:

1. **Detector upgraded** — YOLOv8 replaced with YOLO26 (`yolo26s.pt`
   pre-trained, `best26.pt` fine-tuned); the pipeline code is now
   model-agnostic Ultralytics code.
2. **Refactored into a library** — the notebook's logic was extracted into
   `vd_lib26.py` (model loading, image/video inference, counting, evaluation
   helpers); notebook cells now call library functions instead of holding
   inline code.
3. **Added ground-truth evaluation** — new `load_mot_counts`,
   `evaluate_sequence` and `plot_count_accuracy` compare predictions against
   MOT-style annotations (VisDrone test clip) and plot per-frame count
   accuracy (metric defined above).
4. **Robustness fixes** — video inference consumes the results stream
   (memory fix), and frame folders are assembled with an ffmpeg even-dimension
   scale filter (odd frame heights previously failed to encode).
5. **Portability** — all absolute paths removed; the repo root is
   auto-detected; dead dependencies (pandas, scipy, seaborn) dropped.

## License

This repository is **dual-licensed**:

- Code written for this project (your original additions) is **MIT** — see
  [LICENSE-MIT](LICENSE-MIT).
- Portions derived from `vehicles-detection-and-counting.ipynb` remain under
  the **Apache License 2.0** — see [LICENSE](LICENSE); modified files carry
  prominent notices and [NOTICE.md](NOTICE.md) lists the modifications, as
  required by the license.
- Third-party resources (Ultralytics AGPL-3.0, datasets) keep their own
  licenses — see Acknowledgments above.
