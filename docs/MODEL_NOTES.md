# Model notes: what the app runs, and how far to trust it

## Checkpoint

| | |
|---|---|
| File | `backend/models/geolitternet_v3_seed42.pth` (sha256 `a1965d35…eb73fcac7`) |
| Source | Kaggle notebook *GeoLitterNet New V2*, seed 42 of the 5-seed benchmark (selected on validation) |
| Architecture | High-resolution physics tokenizer → 4 Dynamic Local-Global SR-MHSA stages with MS-PGSS physics gates → multi-scale conv decoder |
| Size | 2,552,213 parameters (2.55 M including BatchNorm buffers) |
| Input | 256 × 256 × 16 = 11 bands (B1, B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12) + FDI, PI, NDVI, MNDWI, SWR |
| Output | 15 MARIDA classes per pixel; the app uses softmax P(class 0 = Marine Debris) |

### Measured accuracy (MARIDA official test split, from the training notebook)

| | Debris F1 | Debris IoU | Macro mIoU |
|---|---|---|---|
| **Seed 42 (shipped)** | 0.659 | 0.491 | 0.406 |
| 5-seed mean ± sd | 0.737 ± 0.057 | — | 0.405 ± 0.033 |
| Seed 42 validation | 0.855 | — | 0.448 |

Seed 42 was the best on **validation**, which is the fair way to pick a checkpoint. Seed 999 scored highest on test
(F1 0.818). To swap it in, set `GLN_MODEL_PATH` to that file; any seed of this architecture loads.

## Input scaling: reproduced exactly

The training dataset class read MARIDA patches, which already hold **reflectance (0–1)**, then divided by 10 000
again (`clip(raw / 10000, 0, 2)`) before computing the indices. The checkpoint confirms it: the first BatchNorm has
`running_var ≈ 1e-11`, which is only possible with inputs around 1e-5.

The app therefore converts every source to physical reflectance first (DN × scale + offset from STAC metadata,
DN / 10000 for GEE's harmonized collection, auto-detected for uploads) and then applies the **identical**
training transform (`app/ml/preprocess.py`). During development, automated tests confirmed bit-identical inputs and output
parity with a plain forward pass. Feeding plain reflectance would push the raw-band branch roughly 10 000× outside
its training range, so **do not "fix" the divisor for this checkpoint**.

What this means for the model: the 11 raw bands reach the network at tiny amplitude, and the FDI index computed
on them is close to zero, so the predictions are driven mainly by the scale-free ratio indices (PI, NDVI, MNDWI,
SWR) plus whatever the BatchNorm re-amplifies. A **retrain with correctly scaled inputs** (reflectance in, indices
on reflectance) is the single most promising accuracy improvement. When you retrain, change `RAW_DIVISOR` in
`preprocess.py` to `1.0` together with the new weights.

## Domain gap: ACOLITE (training) vs Sen2Cor L2A (live)

MARIDA patches were atmospherically corrected with **ACOLITE**; Earth Search and GEE serve **Sen2Cor L2A**. Over
water the two differ, especially in the blue and red-edge bands. Expect:

* probabilities that are calibrated differently from the paper's test numbers;
* more false positives on foam, wakes, sun-glint, turbid plumes and Sargassum (the classic look-alikes; the
  class map layer helps tell them apart).

Recommended practice before operational use:

1. Run the app on known MARIDA event dates and locations, and on a few clean-water scenes.
2. Pick the threshold that suits your use: 0.5 is balanced, 0.7–0.8 favours precision for alerts.
3. Treat clusters as *candidates for inspection*, not confirmed plastic.

## Masking (why some areas show no heatmap)

The model has no land class, so the app analyses open water only:

* **SCL** (Sen2Cor scene classification) defines water / cloud / shadow / land.
* Floating debris and Sargassum are often labelled *vegetation* by Sen2Cor. Small non-water holes fully enclosed
  by water (≤ 0.25 km², `GLN_MAX_HOLE_FILL_PX`) are put back into the analysis area so real detections are not
  masked out.
* Without SCL (some uploads), a spectral NDWI/MNDWI water test and a bright-cloud test are used.

## Limits of 10 m Sentinel-2

Only aggregations that change a 10 m pixel's spectrum are visible: windrows, accumulation patches, river plumes.
Isolated items are not. Detections are strongest in calm conditions, low cloud and outside sun-glint.
