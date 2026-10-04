import hashlib
import io
import itertools
import json
import random
import re
import shutil
import time
import warnings
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

INCLUDE_BG = True                   
RUN_CNN_SEARCH = True               
RUN_BEST_MODEL_SEARCH = True        
EPOCHS_OVERRIDE = None              
EARLY_STOP_PATIENCE = 4             # stop after this many epochs without a higher val acc; None = off
MODEL_NAMES = ("custom_cnn", "resnet18", "efficientnet_b0", "swin_t")
NUM_WORKERS = 0                     # DataLoader worker processes; 0 = main process 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESUME_DIR = None  #"run_10032026_132740_055755"  

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "photo" / "IE7615 Deep Learning for AI"
OUT_DIR = HERE / "results_ie7615" / (RESUME_DIR or datetime.now().strftime("run_%m%d%Y_%H%M%S_%f"))

SIZE = 224
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15
MIN_IMAGE_SIDE = 32                 
JPEG_QUALITY = 95
MAX_SOURCE_PIXELS = 300_000_000     

# Default training settings per model (lr = head lr; backbone gets lr * backbone_mult)
DEFAULTS = {
    "custom_cnn":      dict(epochs=40, lr=1e-3, backbone_mult=1.0, weight_decay=1e-4, batch_size=32),
    "resnet18":        dict(epochs=15, lr=1e-3, backbone_mult=0.1, weight_decay=1e-4, batch_size=32),
    "efficientnet_b0": dict(epochs=15, lr=1e-3, backbone_mult=0.1, weight_decay=1e-4, batch_size=32),
    "swin_t":          dict(epochs=15, lr=1e-3, backbone_mult=0.1, weight_decay=1e-4, batch_size=32),
}

CNN_ARCH = dict(n_blocks=5, base_width=32, convs_per_block=1, dropout=0.3)

CNN_TRIALS = 3
CNN_TRIAL_EPOCHS = 3
CNN_TIMEOUT_S = 10 * 60             
CNN_CONFIG_FILE = "cnn_best_config.json"
CNN_TUNED_CKPT = "custom_cnn_optuna.pt"     # best Optuna trial's model; compare reuses it only if
                                            # its config matches (i.e. same epochs), else retrains

TUNE_GRID = {
    "lr": [3e-4, 1e-3],
    "backbone_mult": [0.1],
    "weight_decay": [1e-4],
}
TUNE_EPOCHS = None                  

MODEL_COLORS = {
    "custom_cnn":      "#2a78d6",   # blue
    "resnet18":        "#eb6834",   # orange
    "efficientnet_b0": "#1baf7a",   
    "swin_t":          "#eda100",   
}
INK = "#0b0b0b"                     
INK_MUTED = "#52514e"               
GRID = "#e4e3df"                    


# ---------------------------------------------------------------- data

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", path.name)]


@contextmanager
def open_image(path):
    """Apply a bounded pixel limit for these local photos, restoring Pillow's default."""
    old_limit = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS
        with Image.open(path) as img:
            yield img
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit


def load_rgb(path):
    with open_image(path) as img:
        if img.format == "JPEG":
            img.draft("RGB", (SIZE * 2, SIZE * 2))
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            img = Image.new("RGB", rgba.size, "white")
            img.paste(rgba, mask=rgba.getchannel("A"))
        else:
            img = img.convert("RGB")
        if img.size != (SIZE, SIZE):
            img = ImageOps.pad(img, (SIZE, SIZE), Image.Resampling.LANCZOS, color="white")
        return img.copy()


def load_cleaned():
    class_to_idx = json.loads((OUT_DIR / "class_to_idx.json").read_text(encoding="utf-8"))
    manifest = pd.read_csv(OUT_DIR / "cleaning_manifest.csv", encoding="utf-8-sig", dtype=str)
    kept = manifest[manifest["status"] == "kept"]
    bg_excluded = (manifest["status"] == "excluded_background").any()
    bg_kept = kept["source"].map(lambda s: "_bg_" in Path(s).name.lower()).any()
    if (INCLUDE_BG and bg_excluded) or (not INCLUDE_BG and bg_kept):
        raise ValueError(f"INCLUDE_BG={INCLUDE_BG} differs from the run being resumed ({OUT_DIR.name})")
    paths = [Path(p) for p in kept["output"]]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} cleaned images are missing, e.g. {missing[0]}")
    print(f"Resume: reusing {len(paths)} cleaned images from {OUT_DIR.name}", flush=True)
    classes = sorted(class_to_idx, key=class_to_idx.get)
    return classes, paths, [class_to_idx[name] for name in kept["class_name"]]


def preprocess_dataset():
    if (OUT_DIR / "class_to_idx.json").exists():    # written last, so cleaning finished
        return load_cleaned()

    class_dirs = sorted((d for d in DATA_DIR.iterdir() if d.is_dir()), key=natural_key)

    from pillow_heif import register_heif_opener
    register_heif_opener()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_dir = OUT_DIR / "cleaned_images"
    clean_dir.mkdir(exist_ok=False)
    classes = [d.name for d in class_dirs]
    records, groups, pixel_groups, jpeg_groups = [], [], {}, {}
    for folder in class_dirs:
        (clean_dir / folder.name).mkdir()
        for path in sorted((p for p in folder.rglob("*") if p.is_file()),
                           key=lambda p: p.relative_to(folder).as_posix().lower()):
            row = dict(source=str(path), class_name=folder.name, status="", reason="",
                       source_format="", width="", height="", output="", duplicate_of="")
            records.append(row)
            if path.parent != folder:
                row.update(status="nested_file", reason="Only direct files belong to this class folder")
                continue
            if not INCLUDE_BG and "_bg_" in path.name.lower():
                row.update(status="excluded_background", reason="INCLUDE_BG=False")
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", Image.DecompressionBombWarning)
                    with open_image(path) as img:
                        row.update(source_format=img.format, width=img.width, height=img.height)
                        img.verify()
                    if min(row["width"], row["height"]) < MIN_IMAGE_SIDE:
                        row.update(status="too_small", reason=f"Minimum side < {MIN_IMAGE_SIDE}")
                        continue
                    img = load_rgb(path)
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError,
                    Image.DecompressionBombWarning) as exc:
                row.update(status="unreadable", reason=f"{type(exc).__name__}: {exc}")
                continue

            pixels = hashlib.sha256(img.tobytes()).hexdigest()
            buffer = io.BytesIO()
            img.save(buffer, "JPEG", quality=JPEG_QUALITY, subsampling=0)
            encoded = buffer.getvalue()
            digest = hashlib.sha256(encoded).hexdigest()
            group_id = pixel_groups.get(pixels, jpeg_groups.get(digest))
            if group_id is None:
                output = clean_dir / folder.name / f"{digest}.jpg"
                output.write_bytes(encoded)
                row.update(status="kept", output=str(output))
                group_id = len(groups)
                groups.append([row])
            else:
                first = groups[group_id][0]
                row.update(status="duplicate", duplicate_of=first["source"])
                groups[group_id].append(row)
            pixel_groups[pixels] = jpeg_groups[digest] = group_id
        print(f"  scanned {folder.name}", flush=True)

    for group in groups:
        if len({row["class_name"] for row in group}) > 1:
            Path(group[0]["output"]).unlink()
            for row in group:
                row.update(status="label_conflict", output="",
                           reason="Identical normalized image appears in multiple classes")
    kept = [row for row in records if row["status"] == "kept"]
    pd.DataFrame(records).to_csv(OUT_DIR / "cleaning_manifest.csv", index=False, encoding="utf-8-sig")
    counts = Counter(row["class_name"] for row in kept)
    summary = {
        "source": str(DATA_DIR), "cleaned_directory": str(clean_dir),
        "target": f"{SIZE}x{SIZE} RGB JPEG", "resize": "preserve aspect ratio, white padding",
        "min_image_side": MIN_IMAGE_SIDE, "jpeg_quality": JPEG_QUALITY,
        "max_source_pixels": MAX_SOURCE_PIXELS, "class_layout": "direct files in immediate class folders",
        "deduplication": "exact normalized pixels or identical output JPEG;",
        "status_counts": dict(Counter(row["status"] for row in records)),
        "source_formats": dict(Counter(row["source_format"] for row in records if row["source_format"])),
        "class_counts": {name: counts[name] for name in classes},
    }
    (OUT_DIR / "cleaning_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Cleaning:", summary["status_counts"], flush=True)
    insufficient = {name: counts[name] for name in classes if counts[name] < 3}
    if insufficient:
        raise ValueError(f"Each class needs >= 3 unique valid images for train/val/test: {insufficient}. "
                         f"See {OUT_DIR / 'cleaning_manifest.csv'}")
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    (OUT_DIR / "class_to_idx.json").write_text(
        json.dumps(class_to_idx, indent=2, ensure_ascii=False), encoding="utf-8")
    return classes, [Path(row["output"]) for row in kept], [class_to_idx[row["class_name"]] for row in kept]


class ImageDataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(load_rgb(self.images[i])), self.labels[i]


def dataset_stats(images):
    total, squared, count = np.zeros(3), np.zeros(3), 0
    for path in images:
        x = np.asarray(load_rgb(path), dtype=np.float64).reshape(-1, 3) / 255
        total += x.sum(axis=0)
        squared += (x * x).sum(axis=0)
        count += len(x)
    mean = total / count
    std = np.sqrt(np.maximum(squared / count - mean * mean, 0))
    return tuple(mean.tolist()), tuple(np.maximum(std, 1e-6).tolist())


def norm_stats(name, train_images=None):
    if name in WEIGHTS:
        t = WEIGHTS[name].transforms()
        return tuple(t.mean), tuple(t.std)
    if train_images is None:
        raise ValueError(f"{name}: no pretrained weights, training images needed for mean/std")
    return dataset_stats(train_images)


def make_transforms(mean, std):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    return train_tf, eval_tf


def build_splits():
    classes, paths, labels = preprocess_dataset()
    rng = np.random.default_rng(SEED)
    indices = {name: [] for name in ("train", "val", "test")}
    for label in range(len(classes)):
        ids = np.flatnonzero(np.asarray(labels) == label)
        rng.shuffle(ids)
        n_val = min(max(1, round(len(ids) * VAL_FRAC)), len(ids) - 2)
        n_test = min(max(1, round(len(ids) * TEST_FRAC)), len(ids) - n_val - 1)
        indices["val"].extend(ids[:n_val])
        indices["test"].extend(ids[n_val:n_val + n_test])
        indices["train"].extend(ids[n_val + n_test:])
    rows, splits = [], {}
    for name, ids in indices.items():
        rng.shuffle(ids)
        splits[name] = ([paths[i] for i in ids], [labels[i] for i in ids])
        rows.extend(dict(path=str(paths[i]), class_name=classes[labels[i]], label=labels[i], split=name)
                    for i in ids)
    df = pd.DataFrame(rows)
    split_file = OUT_DIR / "split_manifest.csv"
    if split_file.exists():                     # resume: the split must not change mid-run
        cols = ["path", "label", "split"]
        saved = pd.read_csv(split_file, encoding="utf-8-sig", dtype=str)
        if not saved[cols].equals(df[cols].astype(str)):
            raise ValueError(f"Rebuilt split differs from {split_file}; cannot resume this run")
    else:
        df.to_csv(split_file, index=False, encoding="utf-8-sig")
    print("Split:", {name: len(value[1]) for name, value in splits.items()}, flush=True)
    return classes, splits


def make_loaders(splits, batch_size, mean, std):
    train_tf, eval_tf = make_transforms(mean, std)
    g = torch.Generator().manual_seed(SEED)
    opts = dict(num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    loaders = {"train": DataLoader(ImageDataset(*splits["train"], train_tf), batch_size,
                                   shuffle=True, generator=g, **opts)}
    for name in ("val", "test"):
        if splits[name][0]:
            loaders[name] = DataLoader(ImageDataset(*splits[name], eval_tf), 64, **opts)
    return loaders


# ---------------------------------------------------------------- models

def conv_block(cin, cout, n_convs=1):
    layers = []
    for i in range(n_convs):
        layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1, bias=False),
                   nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers, nn.MaxPool2d(2))


class CustomCNN(nn.Module):

    def __init__(self, num_classes, n_blocks=5, base_width=32, convs_per_block=1, dropout=0.3):
        super().__init__()
        chans = [3] + [min(base_width * 2 ** i, 512) for i in range(n_blocks)]
        self.features = nn.Sequential(*[conv_block(a, b, convs_per_block)
                                        for a, b in zip(chans, chans[1:])])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout),
                                  nn.Linear(chans[-1], num_classes))

    def forward(self, x):
        return self.head(self.features(x))

WEIGHTS = {
    "resnet18": models.ResNet18_Weights.DEFAULT,
    "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
    "swin_t": models.Swin_T_Weights.DEFAULT,
}

def build_model(name, num_classes, arch=None, pretrained=True):
    if name == "custom_cnn":
        m = CustomCNN(num_classes, **(arch or CNN_ARCH))
        return m, list(m.head.parameters())
    if name == "resnet18":
        m = models.resnet18(weights=WEIGHTS[name] if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m, list(m.fc.parameters())
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=WEIGHTS[name] if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m, list(m.classifier.parameters())
    if name == "swin_t":
        m = models.swin_t(weights=WEIGHTS[name] if pretrained else None)
        m.head = nn.Linear(m.head.in_features, num_classes)
        return m, list(m.head.parameters())
    raise ValueError(name)


def make_optimizer(model, head_params, lr, backbone_mult, weight_decay):
    head_ids = {id(p) for p in head_params}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]
    groups = [{"params": head_params, "lr": lr}]
    if backbone_mult > 0:
        groups.append({"params": backbone, "lr": lr * backbone_mult})
    else:                                       # linear probe: freeze backbone
        for p in backbone:
            p.requires_grad = False
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


# ---------------------------------------------------------------- train / eval

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    device = next(model.parameters()).device
    loss, preds, ys, confidence = 0.0, [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += criterion(out, y).item() * len(y)
        preds += out.argmax(1).cpu().tolist()
        ys += y.cpu().tolist()
        confidence += out.softmax(1).amax(1).cpu().tolist()
    preds, ys = np.array(preds), np.array(ys)
    return {"loss": loss / len(ys), "acc": float((preds == ys).mean()),
            "f1": float(f1_score(ys, preds, labels=range(out.shape[1]), average="macro", zero_division=0)),
            "preds": preds, "labels": ys, "confidence": confidence}


def val_score(acc, loss):
    return acc - 0.01 * loss


def train_model(name, splits, classes, epochs, lr, backbone_mult, weight_decay, batch_size,
                arch=None, trial=None, tag=None, verbose=True):
    set_seed(SEED)
    mean, std = norm_stats(name, splits["train"][0])
    loaders = make_loaders(splits, batch_size, mean, std)
    model, head = build_model(name, len(classes), arch)
    model.to(DEVICE)
    model.norm_stats = {"mean": mean, "std": std}   # saved with the checkpoint for predict
    opt = make_optimizer(model, head, lr, backbone_mult, weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    eval_criterion = nn.CrossEntropyLoss()

    history, best, best_state = [], None, None
    top_acc, top_ep = -1.0, 0                   # early stopping looks at val acc only
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        if backbone_mult == 0:                  # linear probe: keep frozen BatchNorm stats fixed
            model.eval()
        tl, tc, n = 0.0, 0, 0
        for x, y in loaders["train"]:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            n += len(y)
        sched.step()
        v = evaluate(model, loaders["val"], eval_criterion)
        history.append({"epoch": ep, "train_loss": tl / n, "train_acc": tc / n,
                        "val_loss": v["loss"], "val_acc": v["acc"]})
        if best is None or (v["acc"], -v["loss"]) > (best["val_acc"], -best["val_loss"]):
            best = {"epoch": ep, "val_acc": v["acc"], "val_loss": v["loss"], "val_f1": v["f1"]}
            best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if v["acc"] > top_acc:
            top_acc, top_ep = v["acc"], ep
        stale = ep - top_ep
        if trial is not None:
            trial.report(val_score(best["val_acc"], best["val_loss"]), ep)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned(f"pruned at epoch {ep}")
        if verbose:
            print(f"  [{tag or name}] ep {ep:2d}/{epochs}  train loss {tl / n:.3f} acc {tc / n:.3f}"
                  f" | val loss {v['loss']:.3f} acc {v['acc']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if EARLY_STOP_PATIENCE and stale >= EARLY_STOP_PATIENCE and ep < epochs:
            if verbose:
                print(f"  [{tag or name}] early stop: no val acc improvement for {stale} epochs"
                      f" (best ep {best['epoch']})", flush=True)
            break
    train_time = time.time() - t0

    model.load_state_dict(best_state)
    result = {"model": name, "epochs": epochs, "epochs_run": len(history), **best, "train_time_s": round(train_time, 1),
              "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
              "latency_ms": round(measure_latency(model), 1)}
    evals = {}
    for split in ("val", "test"):
        if split in loaders:
            r = evaluate(model, loaders[split], eval_criterion)
            result[f"{split}_acc"], result[f"{split}_f1"] = r["acc"], r["f1"]
            evals[split] = r
    return model, result, pd.DataFrame(history), evals


@torch.no_grad()
def measure_latency(model, runs=20):
    model.eval()
    device = next(model.parameters()).device
    x = torch.randn(1, 3, SIZE, SIZE, device=device)
    for _ in range(3):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / runs * 1000


def save_evaluation(evals, splits, classes, name):
    for split, result in evals.items():
        report = classification_report(result["labels"], result["preds"],
                                       labels=list(range(len(classes))), target_names=classes,
                                       output_dict=True, zero_division=0)
        pd.DataFrame(report).T.to_csv(OUT_DIR / f"report_{name}_{split}.csv", encoding="utf-8-sig")
        cm = confusion_matrix(result["labels"], result["preds"], labels=range(len(classes)))
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            OUT_DIR / f"confusion_{name}_{split}.csv", encoding="utf-8-sig")
        pd.DataFrame({
            "path": [str(p) for p in splits[split][0]],
            "true_class": [classes[i] for i in result["labels"]],
            "predicted_class": [classes[i] for i in result["preds"]],
            "confidence": result["confidence"],
            "correct": result["labels"] == result["preds"],
        }).to_csv(OUT_DIR / f"predictions_{name}_{split}.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------- plots

def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(histories, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for metric, ax in zip(("val_acc", "val_loss"), axes):
        for name, h in histories.items():
            ax.plot(h["epoch"], h[metric], color=MODEL_COLORS.get(name, INK), linewidth=2, label=name)
            ax.plot(h["epoch"], h[metric.replace("val", "train")], color=MODEL_COLORS.get(name, INK),
                    linewidth=1, linestyle="--", alpha=0.6)
        style_axes(ax)
        ax.set_xlabel("epoch", color=INK_MUTED)
        ax.set_title(f"{metric.replace('_', ' ')} (dashed = train)", color=INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_comparison(df, path):
    metrics = [c for c in ("val_acc", "test_acc") if c in df]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(df)
    x = np.arange(len(metrics))
    for i, row in enumerate(df.itertuples()):
        vals = [getattr(row, m) for m in metrics]
        bars = ax.bar(x + (i - (len(df) - 1) / 2) * width, vals, width * 0.92,
                      color=MODEL_COLORS.get(row.model, INK), label=row.model)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, color=INK_MUTED, padding=2)
    ax.set_xticks(x, [m.replace("_", " ") for m in metrics])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy", color=INK_MUTED)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=9, ncol=len(df), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title("Model comparison", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(evals, classes, title, path):
    side = max(5, min(18, 0.23 * len(classes)))
    fig, axes = plt.subplots(1, len(evals), figsize=(side * len(evals), side), squeeze=False)
    for ax, (split, r) in zip(axes[0], evals.items()):
        cm = confusion_matrix(r["labels"], r["preds"], labels=range(len(classes)))
        ax.imshow(cm, cmap="Blues")
        if len(classes) <= 15:
            for i, j in itertools.product(range(len(classes)), repeat=2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                        color="white" if cm[i, j] > cm.max() / 2 else INK)
        ax.set_xticks(range(len(classes)), classes, fontsize=6, rotation=90)
        ax.set_yticks(range(len(classes)), classes, fontsize=6)
        ax.set_xlabel("predicted", color=INK_MUTED)
        ax.set_ylabel("true", color=INK_MUTED)
        ax.set_title(f"{split} acc {r['acc']:.3f}", color=INK, fontsize=10, loc="left")
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tuning(df, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    labels = [f"lr={r.lr:g}\nbb×{r.backbone_mult:g}\nwd={r.weight_decay:g}" for r in df.itertuples()]
    bars = ax.bar(range(len(df)), df["val_acc"], color="#2a78d6", width=0.7)
    bars[0].set_color("#eb6834")               # df is sorted; first = selected config
    ax.bar_label(bars, fmt="%.3f", fontsize=7, color=INK_MUTED, padding=2)
    ax.set_xticks(range(len(df)), labels, fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("val accuracy", color=INK_MUTED)
    style_axes(ax)
    ax.set_title("Hyperparameter search (sorted; orange = selected)", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cnn_search(df, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    done, pruned = df[df["state"] == "COMPLETE"], df[df["state"] == "PRUNED"]
    ax.scatter(pruned["number"], pruned["value"], s=40, color="#c3c2b7", label="pruned", zorder=3)
    ax.scatter(done["number"], done["value"], s=48, color="#2a78d6", label="complete", zorder=3,
               edgecolors="white", linewidths=1.5)
    ax.plot(done["number"], done["value"].cummax(), color="#eb6834", linewidth=2,
            drawstyle="steps-post", label="best so far")
    style_axes(ax)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("trial", color=INK_MUTED)
    ax.set_ylabel("val acc - 0.01 x val loss", color=INK_MUTED)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Custom CNN: Optuna search", color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- commands

def rank(df):
    return df.sort_values(["val_acc", "val_loss", "val_f1"], ascending=[False, True, False]) \
             .reset_index(drop=True)


def save_checkpoint(model, name, classes, config, path, extra=None):
    torch.save({"model": name, "classes": classes, "config": config,
                "norm": getattr(model, "norm_stats", None),
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                **(extra or {})}, path)


def load_saved(path, cfg, num_classes):
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck["config"] != cfg:
        print(f"  {path.name} was saved with a different config, retraining")
        return None
    model, _ = build_model(ck["model"], num_classes, cfg.get("arch"), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.norm_stats = ck["norm"]
    return model.to(DEVICE), ck


def evaluate_splits(model, splits, batch_size):
    loaders = make_loaders(splits, batch_size, model.norm_stats["mean"], model.norm_stats["std"])
    return {split: evaluate(model, loaders[split], nn.CrossEntropyLoss())
            for split in ("val", "test") if split in loaders}


def load_finished(name, cfg, splits, classes):
    ckpt_file, hist_file = OUT_DIR / f"{name}.pt", OUT_DIR / f"history_{name}.csv"
    saved = load_saved(ckpt_file, cfg, len(classes)) if hist_file.exists() else None
    if saved is None:
        return None
    model, ck = saved
    hist = pd.read_csv(hist_file)
    if "result" in ck:
        print(f"  {name}: already finished, reusing {ckpt_file.name}")
        return ck["result"], hist
    # Checkpoints from before resume support carry no result: rebuild it (train time unknown).
    evals = evaluate_splits(model, splits, cfg["batch_size"])
    best = max(hist.itertuples(), key=lambda r: (r.val_acc, -r.val_loss))     # first best, as train_model
    res = {"model": name, "epochs": cfg["epochs"], "epochs_run": len(hist), "epoch": int(best.epoch),
           "val_acc": evals["val"]["acc"], "val_loss": float(best.val_loss), "val_f1": evals["val"]["f1"],
           "train_time_s": None, "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
           "latency_ms": round(measure_latency(model), 1)}
    if "test" in evals:
        res["test_acc"], res["test_f1"] = evals["test"]["acc"], evals["test"]["f1"]
    save_checkpoint(model, name, classes, cfg, ckpt_file, {"result": res})
    print(f"  {name}: already finished, rebuilt its metrics from {ckpt_file.name}")
    return res, hist


def data_key(splits, classes):
    payload = {k: [(p.name, int(y)) for p, y in zip(*v)] for k, v in splits.items()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return {"classes": list(classes), "sha256": digest, "seed": SEED, "size": SIZE}


def save_tuned_cnn(model, cfg, res, hist, evals, splits, classes):
    extra = {"data": data_key(splits, classes), "result": res, "history": hist.to_dict("records"),
             "evals": {k: dict(v, preds=v["preds"].tolist(), labels=v["labels"].tolist())
                       for k, v in evals.items()}}
    save_checkpoint(model, "custom_cnn", classes, cfg, OUT_DIR / CNN_TUNED_CKPT, extra)


def load_tuned_cnn(cfg, splits, classes):
    f = OUT_DIR / CNN_TUNED_CKPT
    if not f.exists():
        return None
    ck = torch.load(f, map_location="cpu", weights_only=True)
    if ck.get("config") != cfg or ck.get("data") != data_key(splits, classes):
        print(f"  custom_cnn: {f.name} was trained with a different config / data"
              f" (e.g. fewer search epochs), retraining")
        return None
    model, _ = build_model("custom_cnn", len(classes), cfg["arch"])
    model.load_state_dict(ck["state_dict"])
    model.norm_stats = ck["norm"]
    evals = {k: dict(v, preds=np.array(v["preds"]), labels=np.array(v["labels"]))
             for k, v in ck["evals"].items()}
    print(f"  custom_cnn: reusing trained model from {f.name} (no retraining)")
    return model, ck["result"], pd.DataFrame(ck["history"]), evals


def load_cnn_config():
    f = OUT_DIR / CNN_CONFIG_FILE
    if f.exists():
        print(f"  custom_cnn: using Optuna config from {f.name}")
        return json.loads(f.read_text())
    print("  custom_cnn: no Optuna config found, using defaults")
    return dict(DEFAULTS["custom_cnn"], arch=CNN_ARCH)


def cmd_tune_cnn(splits, classes):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    epochs = EPOCHS_OVERRIDE or CNN_TRIAL_EPOCHS
    best_value = [float("-inf")]                

    def objective(trial):
        arch = dict(
            n_blocks=trial.suggest_categorical("n_blocks", [4, 5]),
            base_width=trial.suggest_categorical("base_width", [16, 32]),
            convs_per_block=1,
            dropout=0.3,
        )
        cfg = dict(epochs=epochs, backbone_mult=1.0,
                   lr=trial.suggest_categorical("lr", [3e-4, 1e-3]),
                   weight_decay=1e-4,
                   batch_size=32)
        model, res, hist, evals = train_model("custom_cnn", splits, classes, **cfg, arch=arch,
                                              trial=trial, verbose=False)
        for k in ("epoch", "val_acc", "val_loss", "params_M", "train_time_s"):
            trial.set_user_attr(k, res[k])
        value = val_score(res["val_acc"], res["val_loss"])
        if value > best_value[0]:               
            best_value[0] = value
            save_tuned_cnn(model, dict(cfg, arch=arch), res, hist, evals, splits, classes)
        return value

    def log_trial(study, t):
        msg = (f"value {t.value:.4f} (val acc {t.user_attrs['val_acc']:.3f}, "
               f"{t.user_attrs['params_M']}M params, {t.user_attrs['train_time_s']:.0f}s)"
               if t.state == optuna.trial.TrialState.COMPLETE else t.state.name.lower())
        print(f"  [trial {t.number:2d}] {t.params}  -> {msg}")

    print(f"\n=== Optuna search: up to {CNN_TRIALS} trials x {epochs} epochs"
          f" (timeout {CNN_TIMEOUT_S}s)", flush=True)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=1))
    study.enqueue_trial(dict(n_blocks=4, base_width=16, lr=1e-3))
    study.enqueue_trial(dict(n_blocks=4, base_width=16, lr=3e-4))
    study.enqueue_trial(dict(n_blocks=5, base_width=32, lr=1e-3))
    study.optimize(objective, n_trials=CNN_TRIALS, timeout=CNN_TIMEOUT_S, callbacks=[log_trial])

    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state", "duration"))
    df.columns = [c.removeprefix("params_").removeprefix("user_attrs_") for c in df.columns]
    df["duration"] = df["duration"].dt.total_seconds().round(1)
    df["value"] = [t.value if t.value is not None                  # pruned -> last reported score
                   else next(reversed(t.intermediate_values.values()), np.nan) for t in study.trials]
    df.to_csv(OUT_DIR / "cnn_optuna.csv", index=False)
    plot_cnn_search(df, OUT_DIR / "cnn_optuna.png")
    try:
        imp = optuna.importance.get_param_importances(study)
        (OUT_DIR / "cnn_param_importance.json").write_text(json.dumps(imp, indent=2))
        print("  param importance:", {k: round(v, 3) for k, v in imp.items()})
    except Exception as e:                      # needs >= 2 complete trials
        print(f"  [skip] param importance: {e}")

    bp = study.best_params
    cfg = dict(epochs=DEFAULTS["custom_cnn"]["epochs"], backbone_mult=1.0, lr=bp["lr"], weight_decay=1e-4,
               batch_size=32, arch=dict(n_blocks=bp["n_blocks"], base_width=bp["base_width"],
                                        convs_per_block=1, dropout=0.3))
    (OUT_DIR / CNN_CONFIG_FILE).write_text(json.dumps(cfg, indent=2))
    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    print(f"\nBest CNN (trial {study.best_trial.number}, {n_done}/{len(study.trials)} completed): {cfg}")
    return cfg


def cmd_compare(splits, classes):
    rows, histories = [], {}
    for name in MODEL_NAMES:
        cfg = load_cnn_config() if name == "custom_cnn" else dict(DEFAULTS[name])
        if EPOCHS_OVERRIDE:
            cfg["epochs"] = EPOCHS_OVERRIDE
        print(f"\n=== {name}  {cfg}")
        finished = load_finished(name, cfg, splits, classes)
        if finished:
            res, hist = finished
        else:
            cached = load_tuned_cnn(cfg, splits, classes) if name == "custom_cnn" else None
            model, res, hist, evals = cached or train_model(name, splits, classes, **cfg)
            hist.to_csv(OUT_DIR / f"history_{name}.csv", index=False)
            plot_confusion(evals, classes, name, OUT_DIR / f"confusion_{name}.png")
            save_evaluation(evals, splits, classes, name)
            # Saved last, with its result, so a resumed run knows this model is finished.
            save_checkpoint(model, name, classes, cfg, OUT_DIR / f"{name}.pt", {"result": res})
        rows.append(res)
        histories[name] = hist
        print("  ->", {k: v for k, v in res.items() if k != "model"})

    df = rank(pd.DataFrame(rows))
    df.to_csv(OUT_DIR / "comparison.csv", index=False)
    plot_curves(histories, OUT_DIR / "curves.png")
    plot_comparison(df, OUT_DIR / "comparison.png")
    print("\nModel comparison (ranked by val acc, then val loss):")
    print(df.to_string(index=False))
    best = df.loc[0, "model"]
    shutil.copyfile(OUT_DIR / f"{best}.pt", OUT_DIR / "best_model.pt")
    ckpt = torch.load(OUT_DIR / "best_model.pt", map_location="cpu", weights_only=True)
    write_summary(best, ckpt["config"], df.iloc[0].to_dict())
    print(f"\nBest model: {best}")
    return best


def write_summary(model_name, config, metrics):
    summary = {"model": model_name, "config": config,
               **{k: v for k, v in metrics.items() if k != "model"}}
    (OUT_DIR / "best_summary.json").write_text(json.dumps(summary, indent=2))


def cmd_tune(splits, classes, model_name):
    if model_name == "custom_cnn":              # already tuned by Optuna in tune_cnn
        ckpt = torch.load(OUT_DIR / "custom_cnn.pt", map_location="cpu", weights_only=True)
        torch.save(ckpt, OUT_DIR / "best_model.pt")
        comp = pd.read_csv(OUT_DIR / "comparison.csv")
        write_summary(model_name, ckpt["config"], comp[comp["model"] == model_name].iloc[0].to_dict())
        shutil.copyfile(OUT_DIR / "confusion_custom_cnn.png", OUT_DIR / "confusion_best_tuned.png")
        print(f"\nBest model is custom_cnn; no repeated grid search. Saved -> {OUT_DIR / 'best_model.pt'}")
        return
    grid = [dict(zip(TUNE_GRID, v)) for v in itertools.product(*TUNE_GRID.values())]
    baseline = {k: DEFAULTS[model_name][k] for k in TUNE_GRID}
    grid = [hp for hp in grid if hp != baseline]
    comp = pd.read_csv(OUT_DIR / "comparison.csv")
    rows = [dict(comp[comp["model"] == model_name].iloc[0].to_dict(), **baseline, source="compare")]
    epochs = EPOCHS_OVERRIDE or TUNE_EPOCHS or DEFAULTS[model_name]["epochs"]
    print(f"\n=== Hyperparameter search on {model_name}: {len(grid)} new configs x up to {epochs} epochs"
          f" (baseline {baseline} reused from comparison)")
    best_model, best_key = None, None
    for i, hp in enumerate(grid, 1):
        cfg = dict(epochs=epochs, batch_size=DEFAULTS[model_name]["batch_size"], **hp)
        grid_file = OUT_DIR / f"grid_{model_name}_{i}.pt"
        saved = load_saved(grid_file, cfg, len(classes))
        if saved:
            model, ck = saved
            res = ck["result"]
            evals = evaluate_splits(model, splits, cfg["batch_size"])
            print(f"  [{i:2d}/{len(grid)}] {hp}  already finished, reusing {grid_file.name}")
        else:
            model, res, _, evals = train_model(model_name, splits, classes, **cfg,
                                               tag=f"{i}/{len(grid)}", verbose=False)
            res.update(hp, source="grid")
            save_checkpoint(model, model_name, classes, cfg, grid_file, {"result": res})
        rows.append(res)
        print(f"  [{i:2d}/{len(grid)}] {hp}  val acc {res['val_acc']:.3f} loss {res['val_loss']:.3f}"
              f"  (best ep {res['epoch']}/{res['epochs_run']} run, {res['train_time_s']:.0f}s)")
        key = (res["val_acc"], -res["val_loss"], res["val_f1"])
        if best_key is None or key > best_key:
            best_key, best_model, best_cfg, best_evals = key, model, cfg, evals

    df = rank(pd.DataFrame(rows))
    cols = ["source", "lr", "backbone_mult", "weight_decay", "epoch", "val_acc", "val_loss", "val_f1",
            "test_acc", "test_f1", "train_time_s"]
    df = df[[c for c in cols if c in df]]
    df.to_csv(OUT_DIR / "tuning.csv", index=False)
    plot_tuning(df, OUT_DIR / "tuning.png")
    print("\nTuning results (ranked):")
    print(df.to_string(index=False))
    if best_model is None:                      # grid held only the baseline config
        shutil.copyfile(OUT_DIR / f"confusion_{model_name}.png", OUT_DIR / "confusion_best_tuned.png")
        print(f"\nNo new grid configs; keeping comparison checkpoint -> {OUT_DIR / 'best_model.pt'}")
        return
    plot_confusion(best_evals, classes, f"{model_name} (tuned)", OUT_DIR / "confusion_best_tuned.png")
    save_evaluation(best_evals, splits, classes, "best_grid")
    save_checkpoint(best_model, model_name, classes, best_cfg, OUT_DIR / "best_grid_model.pt")
    baseline_summary = json.loads((OUT_DIR / "best_summary.json").read_text(encoding="utf-8"))
    baseline_key = (baseline_summary["val_acc"], -baseline_summary["val_loss"], baseline_summary["val_f1"])
    if best_key > baseline_key:
        shutil.copyfile(OUT_DIR / "best_grid_model.pt", OUT_DIR / "best_model.pt")
        write_summary(model_name, best_cfg, df.iloc[0].to_dict())
        print(f"\nGrid config {best_cfg} beats the comparison run.")
    else:
        print("\n  Keeping comparison checkpoint: its validation score is at least as good.")
    print(f"Final model -> {OUT_DIR / 'best_model.pt'}")


def predict_images(images, checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model, _ = build_model(ckpt["model"], len(ckpt["classes"]), ckpt["config"].get("arch"), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    norm = ckpt.get("norm") or dict(zip(("mean", "std"), norm_stats(ckpt["model"])))
    _, eval_tf = make_transforms(norm["mean"], norm["std"])
    for p in images:
        with torch.no_grad():
            prob = model(eval_tf(load_rgb(p)).unsqueeze(0)).softmax(1)[0]
        k = int(prob.argmax())
        print(f"{p}: {ckpt['classes'][k]}  (p={prob[k]:.3f})")


def main():
    if RUN_CNN_SEARCH and "custom_cnn" in MODEL_NAMES:
        import optuna  
    if RESUME_DIR and not OUT_DIR.is_dir():
        raise FileNotFoundError(f"RESUME_DIR not found: {OUT_DIR}")
    print(f"Device: {DEVICE}\nInput: {DATA_DIR}\nOutput: {OUT_DIR}", flush=True)
    classes, splits = build_splits()
    if RUN_CNN_SEARCH and "custom_cnn" in MODEL_NAMES:
        if (OUT_DIR / CNN_CONFIG_FILE).exists():     # written when the search finishes
            print(f"\nResume: Optuna search already finished, using {CNN_CONFIG_FILE}")
        else:
            cmd_tune_cnn(splits, classes)
    best = cmd_compare(splits, classes)
    if RUN_BEST_MODEL_SEARCH:
        cmd_tune(splits, classes, best)
    ckpt = torch.load(OUT_DIR / "best_model.pt", map_location="cpu", weights_only=True)
    model, _ = build_model(ckpt["model"], len(classes), ckpt["config"].get("arch"), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    loaders = make_loaders(splits, 32, ckpt["norm"]["mean"], ckpt["norm"]["std"])
    evals = {split: evaluate(model, loaders[split], nn.CrossEntropyLoss()) for split in ("val", "test")}
    save_evaluation(evals, splits, classes, "best")
    plot_confusion(evals, classes, ckpt["model"], OUT_DIR / "confusion_best.png")
    print(f"\nCompleted. Best model: {ckpt['model']}; test accuracy: {evals['test']['acc']:.4f}; "
          f"macro F1: {evals['test']['f1']:.4f}\nResults: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
