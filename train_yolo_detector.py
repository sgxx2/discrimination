"""
RUN_MODE values:
    "all": prepare the dataset, train, and evaluate the best checkpoint on the test set.
    "prepare": check dataset paths and prepare the optional mixed training set.
    "eval": evaluate the checkpoint in WEIGHTS on the multi-object test set.
    "eval-single": evaluate WEIGHTS on SINGLE_DIR's EVAL_SPLIT ("val" or "test").
    "predict": use WEIGHTS to detect objects in the paths listed in PREDICT_IMAGES.

"""
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

# Edit these settings, then click Run in the IDE.
RUN_MODE = "all"                   # select a workflow described in the module docstring above
WEIGHTS = None                     # trained best.pt path (Path or string) for evaluation or prediction
PREDICT_IMAGES = []                 # image paths, folders or wildcard patterns; e.g. ["photo/test/*.jpg"]
INCLUDE_SINGLE = False             # add SINGLE_DIR's training photos and evaluate its test photos
EVAL_SPLIT = "test"                # single-object evaluation split when RUN_MODE = "eval-single"
RESUME = None                      # interrupted run's last.pt path when RUN_MODE = "all"; None starts a new run

MODEL = "yolov8s.pt"                # COCO-pretrained start point (n/s/m/l/x)
EPOCHS = 50
PATIENCE = 10                       # early stop after this many epochs without val improvement
IMGSZ = 640                         # composites have a 640 px long edge
BATCH = 16
FRACTION = 1.0                     # fraction of training images; smaller values are useful for quick runs
WORKERS = min(8, os.cpu_count() or 1)  # image loading / augmentation processes
SEED = 20261003
CONF = 0.25                         # prediction confidence threshold
IOU_MATCH = 0.5                     # IoU needed for a prediction to count as a correct location
DEVICE = "0" if torch.cuda.is_available() else "cpu"

HERE = Path(__file__).resolve().parent
YOLO_DIR = HERE / "photo" / "multi_object" / "multi_object_det_10000"
SINGLE_DIR = HERE / "photo" / "single_object_labels"   # one box per photo; train/val/test.txt follow the same split
RUNS_DIR = HERE / "results_yolo"
MOSAIC = 1.0                        # share of training images built as 4-image mosaics (Ultralytics default)


# ---------------------------------------------------------------- dataset
def prepare_dataset():
    data_yaml = YOLO_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found: run annotate_single_object.py, then "
                                "generate_multi_object.py")
    text = data_yaml.read_text(encoding="utf-8")
    fixed = re.sub(r"^path:.*$", f"path: {YOLO_DIR.as_posix()}", text, count=1, flags=re.M)
    if fixed != text:                       # dataset copied to another folder / machine
        data_yaml.write_text(fixed, encoding="utf-8")
    print(f"YOLO dataset: {YOLO_DIR}", flush=True)
    return data_yaml


def with_single(data_yaml, single_dir):
    import yaml
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    single = yaml.safe_load((single_dir / "data.yaml").read_text(encoding="utf-8"))
    if single["names"] != cfg["names"]:
        raise ValueError(f"{single_dir / 'data.yaml'} and {data_yaml} have different class lists")
    cfg["train"] = [(YOLO_DIR / cfg["train"]).as_posix(), (single_dir / single["train"]).as_posix()]
    out = YOLO_DIR / "data_with_single.yaml"
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    n = sum(1 for line in (single_dir / single["train"]).read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"Training also on {n} single-object photos: {single_dir}", flush=True)
    return out


# ---------------------------------------------------------------- train / evaluate
def train(data_yaml):
    from ultralytics import YOLO
    if RESUME:                              # continue an interrupted run with its own settings
        model = YOLO(str(Path(RESUME).expanduser()))
        model.train(resume=True)
        return Path(model.trainer.save_dir) / "weights" / "best.pt"
    model = YOLO(MODEL)
    name = datetime.now().strftime("run_%m%d%Y_%H%M%S") + ("_single" if INCLUDE_SINGLE else "")
    model.train(data=str(data_yaml), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ,
                batch=BATCH, workers=WORKERS, device=DEVICE, seed=SEED, deterministic=True,
                project=str(RUNS_DIR), name=name, pretrained=True, optimizer="auto", cos_lr=True,
                fraction=FRACTION, mosaic=MOSAIC, plots=True, exist_ok=False)
    return Path(model.trainer.save_dir) / "weights" / "best.pt"


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gts, preds):
    used, hits = set(), []
    for p in sorted(preds, key=lambda p: -p["conf"]):
        best, best_iou = None, IOU_MATCH
        for j, g in enumerate(gts):
            if j not in used and g["class_id"] == p["class_id"]:
                v = iou(g["box"], p["box"])
                if v >= best_iou:
                    best, best_iou = j, v
        if best is not None:
            used.add(best)
            hits.append(best_iou)
    return hits


def detect(model, sources, batch=32):
    """Yield (path, [ {class_id, class_name, conf, box} ]) for each image."""
    sources = [str(s) for s in sources]
    for i in range(0, len(sources), batch):
        for r in model.predict(sources[i:i + batch], imgsz=IMGSZ, conf=CONF, iou=0.5, agnostic_nms=True,
                               device=DEVICE, verbose=False):
            yield Path(r.path), [dict(class_id=int(c), class_name=r.names[int(c)], conf=float(s),
                                      box=[float(v) for v in b])
                                 for b, c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(),
                                                    r.boxes.conf.tolist())]


def evaluate(weights, data_yaml, out_dir):
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), split="test", imgsz=IMGSZ, batch=BATCH, device=DEVICE,
                  project=str(out_dir), name="val_test", exist_ok=True, plots=True)
    metrics = {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
               "precision": float(m.box.mp), "recall": float(m.box.mr)}

    # Task-level check: did we get every object's ID and location in each test image?
    gt = pd.read_csv(YOLO_DIR / "gt_boxes.csv", encoding="utf-8-sig")
    gt = gt[gt["split"] == "test"]
    gt_by_image = {name: [dict(class_id=int(r.class_id), box=[r.x0, r.y0, r.x1, r.y1]) for r in g.itertuples()]
                   for name, g in gt.groupby("image")}
    rows, tp, n_pred, n_gt, ious = [], 0, 0, 0, []
    images = sorted((YOLO_DIR / "images" / "test").glob("*.jpg"))
    for path, preds in detect(model, images):
        gts = gt_by_image.get(path.name, [])
        hits = match(gts, preds)
        tp, n_pred, n_gt = tp + len(hits), n_pred + len(preds), n_gt + len(gts)
        ious += hits
        ids_ok = sorted(g["class_id"] for g in gts) == sorted(p["class_id"] for p in preds)
        rows.append(dict(image=path.name, n_gt=len(gts), n_pred=len(preds), correct=len(hits),
                         ids_correct=ids_ok, all_correct=ids_ok and len(hits) == len(gts),
                         gt_ids=" ".join(str(g["class_id"]) for g in gts),
                         predictions=json.dumps([{**p, "box": [round(v, 1) for v in p["box"]]} for p in preds])))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    precision, recall = tp / max(n_pred, 1), tp / max(n_gt, 1)
    metrics.update({
        f"object_precision@iou{IOU_MATCH}": precision,
        f"object_recall@iou{IOU_MATCH}": recall,
        "object_f1": 2 * precision * recall / max(precision + recall, 1e-9),
        "mean_iou_of_correct": float(np.mean(ious)) if ious else 0.0,
        "image_id_set_accuracy": float(df["ids_correct"].mean()),
        "image_all_objects_correct": float(df["all_correct"].mean()),
        "test_images": len(df), "conf": CONF, "weights": str(weights),
    })
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def single_ground_truth(single_dir, split="test"):
    gt = {}
    for line in filter(str.strip, (single_dir / f"{split}.txt").read_text(encoding="utf-8").splitlines()):
        image = single_dir / line.strip().removeprefix("./")     # file names may contain spaces
        label = single_dir / "labels" / image.relative_to(single_dir / "images").with_suffix(".txt")
        with Image.open(image) as im:
            w, h = im.size
        boxes = []
        for row in (label.read_text(encoding="utf-8").splitlines() if label.exists() else []):
            if row.strip():
                c, x, y, bw, bh = (float(v) for v in row.split())
                x, y, bw, bh = x * w, y * h, bw * w, bh * h
                boxes.append(dict(class_id=int(c), box=[x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2]))
        gt[image.resolve()] = boxes
    return gt


def evaluate_single(weights, single_dir, out_dir, split="test"):
    from ultralytics import YOLO
    out_dir.mkdir(parents=True, exist_ok=True)
    gt = single_ground_truth(single_dir, split)
    rows = []
    for path, preds in detect(YOLO(str(weights)), list(gt)):
        gts = gt[path.resolve()]
        top = max(preds, key=lambda p: p["conf"], default=None)
        rows.append(dict(image=path.name, background=not gts, n_pred=len(preds),
                         top_correct=bool(gts and top and top["class_id"] == gts[0]["class_id"]
                                          and iou(top["box"], gts[0]["box"]) >= IOU_MATCH),
                         id_among_boxes=bool(gts) and any(p["class_id"] == gts[0]["class_id"] for p in preds),
                         predictions=json.dumps([{**p, "box": [round(v, 1) for v in p["box"]]} for p in preds])))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"single_{split}_predictions.csv", index=False, encoding="utf-8-sig")
    obj, bg = df[~df.background], df[df.background]
    metrics = {
        "object_photos": len(obj),
        "no_box": float((obj.n_pred == 0).mean()), "one_box": float((obj.n_pred == 1).mean()),
        "two_or_more_boxes": float((obj.n_pred >= 2).mean()),
        f"top_box_correct@iou{IOU_MATCH}": float(obj.top_correct.mean()),
        "right_id_among_boxes": float(obj.id_among_boxes.mean()),
        "background_photos": len(bg), "background_without_box": float((bg.n_pred == 0).mean()),
        "split": split, "conf": CONF, "weights": str(weights),
    }
    (out_dir / f"single_{split}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


# ---------------------------------------------------------------- inference
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RED, GREEN = (230, 40, 40), (40, 190, 70)


def expand_images(sources):
    """Expand configured image paths, folders and wildcard patterns."""
    out = []
    for source in sources:
        p = Path(source).expanduser()
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        elif any(c in str(p) for c in "*?["):
            out += sorted(f for f in p.parent.glob(p.name) if f.suffix.lower() in IMAGE_EXTS)
        elif p.exists():
            out.append(p)
        else:
            print(f"not found: {source}")
    return out


def ground_truth():
    """Ground-truth boxes of the dataset images, keyed by resolved image path."""
    f = YOLO_DIR / "gt_boxes.csv"
    if not f.exists():
        return {}
    gt = {}
    for r in pd.read_csv(f, encoding="utf-8-sig").itertuples():
        key = (YOLO_DIR / "images" / r.split / r.image).resolve()
        gt.setdefault(key, []).append(dict(class_id=int(r.class_id), class_name=r.class_name,
                                           box=[r.x0, r.y0, r.x1, r.y1]))
    return gt


def draw(path, preds, out_path, gts=()):
    with Image.open(path) as im:
        im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=max(12, im.width // 30))
    width = max(2, im.width // 200)
    for g in gts:                                     # ground truth: green, name under the box
        x0, y0, x1, y1 = g["box"]
        d.rectangle([x0, y0, x1, y1], outline=GREEN, width=width)
        tb = d.textbbox((x0, y1), "GT " + g["class_name"], font=font)
        d.rectangle([tb[0], tb[1], tb[2] + 4, tb[3] + 2], fill=GREEN)
        d.text((x0 + 2, y1), "GT " + g["class_name"], fill="white", font=font)
    for p in preds:                                   # prediction: red, name and confidence above
        x0, y0, x1, y1 = p["box"]
        label = f"{p['class_name']} {p['conf']:.2f}"
        d.rectangle([x0, y0, x1, y1], outline=RED, width=width)
        tb = d.textbbox((x0, y0), label, font=font)
        d.rectangle([tb[0], tb[1], tb[2] + 4, tb[3] + 2], fill=RED)
        d.text((x0 + 2, y0), label, fill="white", font=font)
    im.save(out_path, quality=95)


def predict(weights, images, out_dir):
    from ultralytics import YOLO
    images = expand_images(images)
    if not images:
        raise SystemExit("no images to predict")
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    gt_all = ground_truth()
    results, n_gt, n_hit = {}, 0, 0
    for path, preds in detect(model, images):
        gts = gt_all.get(path.resolve(), [])
        print(f"\n{path}: {len(preds)} object(s)")
        for p in sorted(preds, key=lambda p: (p["box"][1], p["box"][0])):
            x0, y0, x1, y1 = p["box"]
            print(f"  ID {p['class_name']:<8} conf={p['conf']:.3f}  box(x0,y0,x1,y1)="
                  f"({x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f})")
        if gts:
            hits = match(gts, preds)
            n_gt, n_hit = n_gt + len(gts), n_hit + len(hits)
            print(f"  ground truth: {' '.join(sorted(g['class_name'] for g in gts))}  -> "
                  f"{len(hits)}/{len(gts)} found with the right ID and IoU >= {IOU_MATCH}")
        draw(path, preds, out_dir / f"{path.stem}_pred.jpg", gts)
        results[str(path)] = dict(predictions=preds, ground_truth=gts)
    (out_dir / "predictions.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if n_gt:
        print(f"\nDataset images: {n_hit}/{n_gt} ground-truth objects found ({n_hit / n_gt:.1%})")
    print(f"\nAnnotated images (red = prediction, green = ground truth) and predictions.json -> {out_dir}")


def main():
    """Execute the workflow selected by the configuration variables above."""
    global YOLO_DIR, RUNS_DIR
    mode = RUN_MODE
    if mode not in ("all", "prepare", "eval", "eval-single", "predict"):
        raise ValueError("RUN_MODE must be 'all', 'prepare', 'eval', 'eval-single' or 'predict'")
    if mode in ("eval", "eval-single", "predict") and not WEIGHTS:
        raise ValueError(f"Set WEIGHTS to a trained best.pt file before running {mode!r}")
    if mode == "eval-single" and EVAL_SPLIT not in ("val", "test"):
        raise ValueError("EVAL_SPLIT must be 'val' or 'test'")
    if mode == "predict":
        if isinstance(PREDICT_IMAGES, (str, Path)):
            raise TypeError('PREDICT_IMAGES must be a list, e.g. ["photo/test/*.jpg"]')
        if not PREDICT_IMAGES:
            raise ValueError("Set PREDICT_IMAGES to at least one image, folder or wildcard pattern")
    YOLO_DIR = Path(YOLO_DIR).expanduser().resolve()
    RUNS_DIR = Path(RUNS_DIR).expanduser().resolve()
    single_dir = Path(SINGLE_DIR).expanduser().resolve()
    single = single_dir if INCLUDE_SINGLE else None
    weights = Path(WEIGHTS).expanduser() if WEIGHTS else None
    print(f"Mode: {mode}\nDevice: {DEVICE}  workers: {WORKERS}\nDataset: {YOLO_DIR}\nRuns: {RUNS_DIR}", flush=True)

    if mode == "predict":
        predict(weights, PREDICT_IMAGES, RUNS_DIR / "predictions" / datetime.now().strftime("%m%d%Y_%H%M%S"))
        return
    if mode == "eval-single":
        evaluate_single(weights, single_dir, weights.parent.parent / "test_eval", EVAL_SPLIT)
        return
    data_yaml = prepare_dataset()
    train_yaml = with_single(data_yaml, single) if single else data_yaml
    if mode == "prepare":
        return
    if mode == "eval":
        best = weights
    else:
        best = train(train_yaml)
    evaluate(best, data_yaml, best.parent.parent / "test_eval")         # always the multi-object test split
    if single:
        evaluate_single(best, single, best.parent.parent / "test_eval")
    print(f"\nBest weights: {best}", flush=True)


if __name__ == "__main__":
    main()
