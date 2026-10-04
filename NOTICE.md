NOTICE
======

This file satisfies section 4(d) of the Apache License, Version 2.0. It records
the attribution notices that must be preserved with any distribution of this
derivative work.

This repository (`main_workflow26.ipynb` and `vd_lib26.py`) is a modified
derivative of the following work:

  "Vehicles Detection and Counting"
  Kaggle notebook by hakim11
  https://www.kaggle.com/code/hakim11/vehicles-detection-and-counting
  Licensed under the Apache License, Version 2.0.

Copyright for the original notebook remains with its author; this derivative
adds the changes listed below. The code in this repository is licensed under
the Apache License, Version 2.0 (see LICENSE).

Changes made in this repository
-------------------------------

1. Detector upgraded from YOLOv8 to YOLO26 (`yolo26s.pt` pre-trained,
   `best_visdrone.pt` fine-tuned).
2. Added a per-vehicle danger value computed from the rate of bounding-box
   growth (the visual "looming" cue, ~1 / time-to-collision) — the paper's
   core contribution; the original notebook does not compute this.
3. Changed the fine-tuning dataset from a top-view vehicle image set to
   Ultralytics VisDrone-DET (drone footage), collapsed to a single `vehicle`
   class (car / van / truck / bus).
4. Extended evaluation with MOT ground-truth comparison: per-clip
   mAP50 / mAP50-95 and precision / recall, a one-to-one TP/FP/FN
   correspondence, and danger-value validation against the annotations.
5. Extracted the notebook's logic into `vd_lib26.py` (model-agnostic),
   removed all absolute paths, and made the repo root auto-detected.

Third-party components (each keeps its own license)
---------------------------------------------------

- Ultralytics YOLO26 (AGPL-3.0) — detection/training framework and the
  pre-trained `yolo26s.pt` weights. https://github.com/ultralytics/ultralytics
- VisDrone datasets (VisDrone-DET and VisDrone2019-MOT) — training and
  evaluation data; free for research use, cite Zhu et al., "Detection and
  tracking meet drones challenge", IEEE TPAMI 44(11):7380-7399, 2021.
