# Figure manifest — which file is which figure

Every file lives in this folder (`/Users/k/.reasonix/global-workspace/figures/`). Filenames carry the figure number used in `veritas_results_draft.md`. Four numbered figures, **seven image files**.

| Figure | File | Clip / set | Pixels | Size | What the image contains |
|---|---|---|---|---|---|
| **1a** | `Figure1a_count_day.png` | daytime | 1959×1269 | 393 KB | Three panels. Top left: absolute count error in every frame, with the RMSE (dashed) and MAE (dotted) as reference lines. Top right: predicted count against annotated count, one dot per frame, with a y = x line. Bottom (full width): per-frame annotated counts (grey) and predicted counts (blue), the ten-frame rolling mean of the prediction (red) and its rolling standard deviation as a shaded band. |
| **1b** | `Figure1b_count_night.png` | night-time | 1959×1269 | 342 KB | The same three panels, night clip. |
| **1c** | `Figure1c_count_heavy.png` | heavy traffic | 1959×1269 | 309 KB | The same three panels, heavy-traffic clip. |
| **2** | `Figure2_PR_curve.png` | VisDrone validation split | 2250×1500 | 89 KB | Precision–recall curve written by Ultralytics `val()`. |
| **2** | `Figure2_confusion_matrix.png` | VisDrone validation split | 3000×2250 | 91 KB | Detection confusion matrix. `BoxP_curve.png`, `BoxR_curve.png` and `BoxF1_curve.png` also exist in `data_out/map_val/` if you prefer one of those. |
| — | `Figure2_confusion_matrix_normalized.png` | VisDrone validation split | 3000×2250 | 92 KB | Row-normalised confusion matrix. **Not referenced in the draft** — reads better than raw counts when the classes are as unbalanced as these. |
| **3** | `Figure3_danger_distribution.png` | all three clips | 1650×480 | 61 KB | Distribution of the danger values, by clip. **Earlier rendering** — see the note below. |
| **4** | `Figure4_danger_vs_track_length.png` | all three clips | 1650×510 | 95 KB | Peak danger value against track length, by clip — the chart behind the long-track versus short-track sentence. |
| — | `extra_danger_per_frame_max.png` | all three clips | 1500×810 | 236 KB | Highest danger value in each frame, over time. **Not referenced in the draft** — the natural third danger figure, and the only place the sustained episodes (13 / 7 / 9) show up as spikes. |

Each of Figures 1a–1c is one combined image per clip, so the paper uses one figure number for the three conditions and the reader compares them across 1a, 1b and 1c.

## Which notebook cell reproduces each figure

Every figure is produced by `main_workflow26.ipynb` — nothing depends on a throwaway script any more.

| Figure | Cell | Call |
|---|---|---|
| 1a, 1b, 1c | 21 (day), 23 (night), 25 (heavy) | `evaluate_sequence(..., save_plots_dir=FIGURES)` → writes `count_summary_<clip>.png`, the combined three-panel figure |
| 2 | 29, then 36 | `evaluate_map(...)` draws the curves into `data_out/map_val/`; cell 36 copies them out under paper names |
| 3, 4, extra | 31, then 34 | `analyze_danger_video(...)` writes the CSVs; `plot_danger_figures(...)` draws the three figures |
| all seven | 36 | renames and gathers everything into `figures/` |

The danger table behind Figures 3 and 4 is cell 33 (`summarise_danger(...)`), writing `data_out/danger/danger_summary.csv`. It reproduces the earlier `danger_summary.csv` **exactly** — all 18 numeric columns, all three clips — and adds the five columns the paper needs.

## `notebook_render/` — and why the danger figures have two versions

The three danger figures were originally drawn by a script that was run inline and no longer exists. To make them reproducible I reimplemented the plotting in `plot_danger_figures()`. The **numbers are identical** (same CSVs, same values), but the **pictures differ**: 16–23% of pixels differ between old and new, and the distribution panel has noticeably less ink (17.2% → 8.8%), so the old one plotted something denser — most likely more bins, or all rows rather than the scored ones.

Both versions are therefore kept:

- `Figure3_*`, `Figure4_*`, `extra_*` in this folder — the **earlier** rendering, from the deleted script.
- `notebook_render/` — the same three, as the **notebook** now draws them.

Open both and keep the pair you prefer; if you keep the notebook's, copy `notebook_render/*` over the top-level files so the folder matches what the notebook regenerates.

## Where each came from

| Figure | Produced by |
|---|---|
| 1a, 1b, 1c | The counting path on each clip: frames assembled into a video, `count_vehicles_in_video`, then `plot_count_summary`. The same run recomputed the counts independently — RMSE 5.266 / 7.434 / 15.770 against the notebook's 5.334 / 7.358 / 15.879, i.e. agreement to about 1%. |
| 2 | Ultralytics `val()` on the 548-image vehicle validation split → `data_out/map_val/`. Copied here from your Desktop repo. |
| 3, 4, extra | The danger run on all three clips, plotted afterwards. Originally copied from `danger_results/figures/`, which was never itself copied into your repo — Desktop writes are blocked for me. |

## Copying these into the repository

```bash
cp -r /Users/k/.reasonix/global-workspace/figures \
      "/Users/k/Desktop/summer/veritas/github_yolo-vehicles/"
```

## Two things to check by eye

1. The descriptions above come from the plotting code and its printed output, not from looking at the images — I cannot view pictures. **Open each file before submitting** and confirm the axis labels, legends and titles read the way you want.
2. Figure 2 spans two files (precision–recall curve and confusion matrix). If your paper template wants one image per figure number, that is the one place you still need to combine two panels, or split it into Figure 2 and Figure 3 and renumber.
