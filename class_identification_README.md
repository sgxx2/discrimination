# `class_identification.py` — User Guide

An end-to-end image classification pipeline for the IE7615 object dataset. A single
run of the script:

1. **Cleans** the raw photos (validates, de-duplicates, resizes to 224×224 JPEG).
2. **Splits** them into stratified train / validation / test sets (70 / 15 / 15).
3. **Tunes** a custom CNN with Optuna.
4. **Trains and compares** four models: `custom_cnn`, `resnet18`, `efficientnet_b0` and `swin_t`.
5. **Grid-searches** hyperparameters for the winning model.
6. **Evaluates** the final model on the test set and saves reports, plots and checkpoints.


## 1. Data layout

The script reads images from `photo/IE7615 Deep Learning for AI/`, relative to the
script's own folder:

```
discrimination/
├── class_identification.py
├── photo/
│   └── IE7615 Deep Learning for AI/
│       ├── images_OBJ001/          ← one folder = one class
│       │   ├── OBJ001_001.jpg
│       │   ├── OBJ001_bg_001.jpg   ← "_bg_" in the name = photo with background
│       │   └── ...
│       ├── images_OBJ002/
│       └── ...
└── results_ie7615/                 ← created automatically; one subfolder per run
```

Rules:

- Each **direct subfolder** is one class. The folder name becomes the class name.
  Classes are sorted in natural order, so `OBJ2` comes before `OBJ10`.
- Only files **directly inside** a class folder are used. Files in deeper subfolders
  are skipped and logged as `nested_file`.
- Any format Pillow can open works: JPG, PNG, HEIC and others.
- Each class needs **at least 3 unique, valid images** after cleaning.

---

## 2. Configuration

All settings are constants at the top of the file. 

### Run control

| Setting | Default | Description |
|---|---|---|
| `INCLUDE_BG` | `True` | Include images with `_bg_` in the file name. `False` drops them during cleaning. |
| `RUN_CNN_SEARCH` | `True` | Run the Optuna search for `custom_cnn` before the comparison. |
| `RUN_BEST_MODEL_SEARCH` | `True` | After the comparison, grid-search hyperparameters for the winning model. |
| `EPOCHS_OVERRIDE` | `None` | When set, replaces the epoch count in **every** stage (Optuna trials, comparison, grid search). |
| `EARLY_STOP_PATIENCE` | `4` | Stop training after this many epochs without a new best validation accuracy. `None` disables early stopping. |
| `MODEL_NAMES` | all 4 | Models to train and compare. Remove a name to skip that model. |
| `NUM_WORKERS` | `0` | DataLoader worker processes. `0` is the safest choice on Windows. |
| `RESUME_DIR` | `None` | Name of an existing run folder in `results_ie7615/` to continue. See §6. |
| `DATA_DIR` / `OUT_DIR` | see §2 | Input dataset folder and output folder. |

### Data processing

| Setting | Default | Description |
|---|---|---|
| `SIZE` | `224` | Output image size (square). |
| `SEED` | `42` | Random seed for the split, the DataLoader shuffling and model initialization. |
| `VAL_FRAC`, `TEST_FRAC` | `0.15`, `0.15` | Fraction of each class used for validation and test. |
| `MIN_IMAGE_SIDE` | `32` | Images whose shorter side is below this are rejected. |
| `JPEG_QUALITY` | `95` | Quality of the cleaned JPEGs. |
| `MAX_SOURCE_PIXELS` | `300_000_000` | Largest source image allowed (protects against decompression bombs). |

### Training hyperparameters

`DEFAULTS` holds the training settings for each model:

| Key | Meaning |
|---|---|
| `epochs` | Maximum number of epochs (early stopping may end training sooner). |
| `lr` | Learning rate of the classification **head**. |
| `backbone_mult` | Backbone learning rate = `lr × backbone_mult`. `0` freezes the backbone (linear probe). |
| `weight_decay` | AdamW weight decay. |
| `batch_size` | Training batch size. Lower it if you run out of GPU memory. |

Other tuning settings:

| Setting | Default | Description |
|---|---|---|
| `CNN_ARCH` | 5 blocks, width 32 | Custom-CNN architecture used when no Optuna result exists. |
| `CNN_TRIALS` | `3` | Maximum number of Optuna trials. |
| `CNN_TRIAL_EPOCHS` | `3` | Epochs per Optuna trial. |
| `CNN_TIMEOUT_S` | `600` | Optuna time budget in seconds. It is checked between trials, so a running trial always finishes. |
| `TUNE_GRID` | `lr ∈ {3e-4, 1e-3}`, `backbone_mult ∈ {0.1}`, `weight_decay ∈ {1e-4}` | Grid for stage 5. |
| `TUNE_EPOCHS` | `None` | Epochs per grid configuration. Uses the model's `DEFAULTS` value when `None`. |

---

## 3. What each stage does

### Stage 1 — Cleaning (`preprocess_dataset`)

Every source file is checked and gets one status in `cleaning_manifest.csv`:

| Status | Meaning |
|---|---|
| `kept` | Valid and unique. Written to `cleaned_images/<class>/<sha256>.jpg`. |
| `duplicate` | Same normalized pixels, or same output JPEG, as an image that was already kept. |
| `label_conflict` | The same image appears in more than one class. **All** copies are dropped. |
| `too_small` | Shorter side is below `MIN_IMAGE_SIDE`. |
| `unreadable` | Corrupted, unsupported or too large. The `reason` column has the error. |
| `nested_file` | Not directly inside a class folder. |
| `excluded_background` | Has `_bg_` in its name while `INCLUDE_BG = False`. |

Each kept image is processed in this order:

1. Rotated according to its EXIF orientation.
2. Transparent areas are filled with white.
3. Converted to RGB.
4. Resized to 224×224 **keeping the aspect ratio**, with white padding.
5. Saved as a JPEG.

### Stage 2 — Split (`build_splits`)

The split is stratified by class with a fixed seed. Each class gets about 15% validation,
15% test and the rest training, with at least one validation image and one test image per
class. The split is saved to `split_manifest.csv`.

### Stage 3 — Custom CNN search (`cmd_tune_cnn`)

- **Architecture:** `n_blocks` blocks of (Conv 3×3 → BatchNorm → ReLU → MaxPool), then
  global average pooling, dropout 0.3 and a linear layer. Channels start at `base_width`,
  double in each block and are capped at 512.
- **Search space:** `n_blocks ∈ {4, 5}`, `base_width ∈ {16, 32}`, `lr ∈ {3e-4, 1e-3}`.
- **Objective:** `val_acc − 0.01 × val_loss`, maximized with a TPE sampler and a median pruner.
- Three configurations are queued first. With the default `CNN_TRIALS = 3`, the search
  therefore evaluates only those three. Increase `CNN_TRIALS` to let TPE explore further.
- The best configuration is saved to `cnn_best_config.json` with the full training length
  (`DEFAULTS["custom_cnn"]["epochs"]`, 40 by default). The best trial's model is saved to
  `custom_cnn_optuna.pt`. Stage 4 reuses that model only if its config matches, for example
  when `EPOCHS_OVERRIDE` is set. Otherwise the CNN is retrained for the full 40 epochs.

### Stage 4 — Model comparison (`cmd_compare`)

| Model | Initialization | Replaced layer | Input normalization |
|---|---|---|---|
| `custom_cnn` | From scratch | — | Mean/std computed from the training images |
| `resnet18` | ImageNet | `fc` | ImageNet mean/std |
| `efficientnet_b0` | ImageNet | `classifier[-1]` | ImageNet mean/std |
| `swin_t` | ImageNet | `head` | ImageNet mean/std |

Training setup, the same for all models:

- **Optimizer:** AdamW with two learning rates, one for the head and one for the backbone.
- **Schedule:** cosine annealing.
- **Loss:** cross-entropy with label smoothing 0.1 for training. Plain cross-entropy for evaluation.
- **Augmentation (training only):** RandomResizedCrop (scale 0.6–1.0), horizontal flip,
  ±15° rotation and color jitter. Validation and test images are only normalized.
- **Best epoch:** the weights from the epoch with the highest validation accuracy are kept.
  Ties go to the lower validation loss.

Models are ranked by validation accuracy, then lower validation loss, then validation
macro-F1. **The test set is never used to choose a model.** The winner is copied to
`best_model.pt` and its metrics are written to `best_summary.json`.

### Stage 5 — Hyperparameter grid search (`cmd_tune`)

- Every combination in `TUNE_GRID` is trained on the winning model, except the model's
  default configuration, which stage 4 already trained. With the default grid, that leaves
  one new configuration: `lr = 3e-4`.
- The best grid model is saved as `best_grid_model.pt`. It replaces `best_model.pt` only
  if its validation score is **strictly better** than the stage 4 result.
- If the winner is `custom_cnn`, this stage is skipped because Optuna already tuned it.

### Stage 6 — Final evaluation

The script reloads `best_model.pt` and evaluates it on the validation and test sets. It
writes `report_best_*.csv`, `confusion_best_*.csv`, `predictions_best_*.csv` and
`confusion_best.png`, then prints the test accuracy and macro-F1.


---

## 4. Output files

All outputs are written to `results_ie7615/<run>/`.

| File | Contents |
|---|---|
| `cleaned_images/` | Cleaned 224×224 JPEGs, one subfolder per class |
| `cleaning_manifest.csv` | Status of every source file (see §5, Stage 1) |
| `cleaning_summary.json` | Counts by status, source format and class |
| `class_to_idx.json` | Class name → label index |
| `split_manifest.csv` | Path, class, label and split of every kept image |
| `cnn_optuna.csv` / `.png` | All Optuna trials and a plot of the search progress |
| `cnn_param_importance.json` | Optuna hyperparameter importance (needs ≥ 2 completed trials) |
| `cnn_best_config.json` | Selected custom-CNN configuration |
| `custom_cnn_optuna.pt` | Model from the best Optuna trial |
| `<model>.pt` | Checkpoint of each compared model |
| `history_<model>.csv` | Training and validation loss/accuracy for each epoch |
| `report_<model>_<split>.csv` | Precision, recall and F1 for each class |
| `confusion_<model>_<split>.csv` / `confusion_<model>.png` | Confusion matrices |
| `predictions_<model>_<split>.csv` | Prediction and confidence for each image, and whether it was correct |
| `comparison.csv` / `.png` | Ranked comparison of the models (accuracy, F1, parameters, latency, training time) |
| `curves.png` | Validation (solid) and training (dashed) curves for all models |
| `grid_<model>_<i>.pt` | Checkpoint for each grid configuration |
| `tuning.csv` / `.png` | Ranked grid-search results |
| `best_grid_model.pt` | Best grid-search model |
| `confusion_best_tuned.png` | Confusion matrix of the tuned winner |
| **`best_model.pt`** | **Final selected model** |
| **`best_summary.json`** | **Final model name, configuration and metrics** |
| `*_best_*.csv`, `confusion_best.png` | Final evaluation of `best_model.pt` |

The `report_*` and `predictions_*` files use the model name for the comparison models,
`best_grid` for the grid winner and `best` for the final model.

Each `.pt` checkpoint is a dict with these keys: `model` (architecture name), `classes`,
`config`, `norm` (mean/std), `state_dict`, and usually `result` (metrics).

### Reference results

From run `run_10032026_143832_651357`, which used the default settings:

| Model | Val acc | Test acc | Test macro-F1 | Params (M) | Latency (ms) |
|---|---|---|---|---|---|
| **swin_t** | 0.963 | 0.948 | 0.948 | 27.6 | 72.0 |
| efficientnet_b0 | 0.955 | 0.946 | 0.945 | 4.1 | 12.8 |
| resnet18 | 0.949 | 0.942 | 0.942 | 11.2 | 10.7 |
| custom_cnn | 0.779 | 0.788 | 0.783 | 1.6 | 7.8 |

---

## 5. Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: optuna` | Install `optuna`, or set `RUN_CNN_SEARCH = False`. |
| `ModuleNotFoundError: pillow_heif` | Install `pillow-heif`. It is needed for cleaning. |
| `Each class needs >= 3 unique valid images` | Look up the listed classes in `cleaning_manifest.csv` to see why their images were rejected. |
| `INCLUDE_BG=... differs from the run being resumed` | Set `INCLUDE_BG` back to the value used for that run, or start a new run. |
| `Rebuilt split differs from ...; cannot resume this run` | The dataset, `SEED` or split fractions changed. Start a new run (`RESUME_DIR = None`). |
| `RESUME_DIR not found` | Use only the folder name inside `results_ie7615/`, not a full path. |
| `FileExistsError` for `cleaned_images` | An earlier run with the same `RESUME_DIR` crashed during cleaning. Delete that run folder, or its `cleaned_images/` subfolder, and run again. |
| `Device: cpu` although you have a GPU | Your PyTorch build is CPU-only. Install the CUDA build from pytorch.org. |
| CUDA out of memory | Lower `batch_size` in `DEFAULTS` (Swin-T uses the most memory). |
| Pretrained weight download fails | Connect to the internet for the first run, or copy the weight files into the torch hub cache. |
