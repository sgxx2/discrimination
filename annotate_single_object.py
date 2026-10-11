"""Auto-label every single-object photo with one tight object box, using YOLOE-26 (Ultralytics).

Boxes are drawn on the cleaned 224x224 images of class_identification.py (white padding,
duplicates removed), reusing a class_identification.py run's cleaning and train/val/test split, so
generate_multi_object.py can paste the same images and keep the same split.

"""
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont, ImageOps

import class_identification as ci

RUN_MODE = "all"
ONLY_CLASSES = []                   # [] = all classes / review queue; e.g. ["OBJ001", "OBJ009"] to filter

DET_MODEL = "yoloe-26l-seg-pf.pt"   # prompt-free YOLOE-26 (4585-name vocabulary), n/s/m/l/x
TEXT_MODEL = "yoloe-26l-seg.pt"     # text-promptable YOLOE-26 for the refine pass
CLASSIFIER = "efficientnet_b0.pt"   # class_identification checkpoint used to recognise each crop
DET_IMGSZ = 640                     # 224 px photos are upsampled so small objects are found
DET_CONF = 0.05                     # keep weak proposals; the selection decides
DET_BATCH = 16
MAX_CANDIDATES = 12                 # proposals kept per photo after class-agnostic NMS
CROP_MARGIN = 0.08                  # context added around a box before classifying the crop
CONSISTENCY_ITERS = 2               # co-localisation rounds (matches weighted by the last round)
EMBEDDER = "dinov2_vits14"          # torch.hub facebookresearch/dinov2, context-free crop features
REFINE = True                       # text-prompt pass for photos that are still flagged
REVIEW_FRAC = 0.05                  # share of object photos copied to review/
REVIEW_RANDOM_SHARE = 0.2           # part of the review set drawn at random (estimates error rate)
EXCLUDE_UNREVIEWED_HIGH_RISK = True  # high-risk review photos are only used once reviewed (spot checks are)
SEED = 20261003
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HERE = Path(__file__).resolve().parent
CLEAN_RUN = HERE / "results_ie7615" / "run_10032026_143832_651357"   # classifier run (background kept)
OUT_DIR = HERE / "photo" / "single_object_labels"
WEIGHTS_DIR = HERE / "weights"
OVERRIDES = OUT_DIR / "manual_overrides.csv"
REVIEW_DIR = OUT_DIR / "review"
OVERRIDE_COLS = ["image", "action", "x0", "y0", "x1", "y1", "time"]

GREEN, YELLOW, RED, GRAY = (40, 200, 70), (250, 200, 30), (230, 40, 40), (150, 150, 150)


# ---------------------------------------------------------------- photos
def load_records():
    if output_complete():
        return load_records_from_output()
    if not (CLEAN_RUN / "class_to_idx.json").exists():
        # never let class_identification start a new cleaning inside someone else's run folder
        raise FileNotFoundError(f"{CLEAN_RUN} has no finished cleaning (class_to_idx.json missing). "
                                "Restore that run folder or point CLEAN_RUN at another finished run.")
    ci.INCLUDE_BG = True
    ci.OUT_DIR = CLEAN_RUN
    classes, splits = ci.build_splits()          # reuses the run's cleaning and verifies its split
    manifest = pd.read_csv(CLEAN_RUN / "cleaning_manifest.csv", encoding="utf-8-sig", dtype=str)
    source = dict(zip(manifest["output"], manifest["source"]))
    records, used = [], set()
    for split, (paths, labels) in splits.items():
        for path, label in zip(paths, labels):
            original = Path(source[str(path)])
            stem = original.stem
            while (classes[label], stem) in used:   # same stem with another extension
                stem += "_2"
            used.add((classes[label], stem))
            records.append(dict(key=f"{classes[label]}/{stem}", cleaned=Path(path), source=original,
                                class_id=label, class_name=classes[label], split=split, stem=stem,
                                is_bg="_bg_" in original.name.lower()))
    records.sort(key=lambda r: (r["class_id"], ci.natural_key(Path(r["stem"]))))
    return classes, records


def output_complete():
    if not (OUT_DIR / "boxes.csv").exists() or not (OUT_DIR / "classes.txt").exists():
        return False
    n_classes = len((OUT_DIR / "classes.txt").read_text(encoding="utf-8").split())
    return pd.read_csv(OUT_DIR / "boxes.csv", encoding="utf-8-sig", usecols=["class_name"])["class_name"].nunique() == n_classes


def load_records_from_output():
    classes = ["images_" + c for c in (OUT_DIR / "classes.txt").read_text(encoding="utf-8").split()]
    df = pd.read_csv(OUT_DIR / "boxes.csv", encoding="utf-8-sig")
    records = []
    for r in df.to_dict("records"):
        key = r["image"].removeprefix("images/").removesuffix(".jpg")
        image = OUT_DIR / r["image"]
        if not image.exists():
            raise FileNotFoundError(f"{image} is missing from the annotation folder")
        records.append(dict(key=key, cleaned=image, source=Path(r["source"]), class_id=int(r["class_id"]),
                            class_name="images_" + r["class_name"], split=r["split"],
                            stem=key.split("/", 1)[1], is_bg=bool(r["is_background"])))
    records.sort(key=lambda r: (r["class_id"], ci.natural_key(Path(r["stem"]))))
    return classes, records


def short(class_name):
    return class_name.removeprefix("images_")


def content_box(im):
    a = np.asarray(im.convert("L")) >= 240
    w, h = im.size

    def run(lines):
        n = 0
        for line in lines:
            if line.mean() < 0.99:
                break
            n += 1
        return n

    left, right = run(a.T), run(a.T[::-1])
    top, bottom = run(a), run(a[::-1])
    if left and right and abs(left - right) <= 2 and left + right < w - 8:
        return [float(left), 0.0, float(w - right), float(h)]
    if top and bottom and abs(top - bottom) <= 2 and top + bottom < h - 8:
        return [0.0, float(top), float(w), float(h - bottom)]
    return [0.0, 0.0, float(w), float(h)]


def box_area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------- models
def yoloe(name):
    from ultralytics import YOLOE
    WEIGHTS_DIR.mkdir(exist_ok=True)
    return YOLOE(str(WEIGHTS_DIR / name))       # downloaded there on first use


class CropScorer:
    """class_identification classifier: P(class | crop) and its top class."""

    def __init__(self, classes):
        ck = torch.load(CLEAN_RUN / CLASSIFIER, map_location="cpu", weights_only=True)
        if list(ck["classes"]) != list(classes):
            raise ValueError(f"{CLASSIFIER} was trained on different classes")
        self.model, _ = ci.build_model(ck["model"], len(classes), ck["config"].get("arch"), pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval().to(DEVICE)
        norm = ck.get("norm") or dict(zip(("mean", "std"), ci.norm_stats(ck["model"])))
        _, self.tf = ci.make_transforms(norm["mean"], norm["std"])

    @staticmethod
    def crop(im, box, content):
        x0, y0, x1, y1 = box
        mx, my = (x1 - x0) * CROP_MARGIN, (y1 - y0) * CROP_MARGIN
        x0, y0 = max(content[0], x0 - mx), max(content[1], y0 - my)
        x1, y1 = min(content[2], x1 + mx), min(content[3], y1 + my)
        c = im.crop((round(x0), round(y0), max(round(x1), round(x0) + 1), max(round(y1), round(y0) + 1)))
        return ImageOps.pad(c, (ci.SIZE, ci.SIZE), Image.Resampling.LANCZOS, color="white")

    @torch.no_grad()
    def __call__(self, crops, class_id, batch=64):
        probs, tops = [], []
        for i in range(0, len(crops), batch):
            x = torch.stack([self.tf(c) for c in crops[i:i + batch]]).to(DEVICE)
            p = self.model(x).softmax(1)
            probs.append(p[:, class_id].cpu())
            tops.append(p.argmax(1).cpu())
        return torch.cat(probs).numpy(), torch.cat(tops).numpy()


class Embedder:
    """DINOv2 CLS embedding of a crop (L2-normalised): similar for the same object in any room."""

    def __init__(self):
        self.model = torch.hub.load("facebookresearch/dinov2", EMBEDDER, verbose=False).eval().to(DEVICE)
        self.tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    @torch.no_grad()
    def __call__(self, crops, batch=64):
        out = []
        for i in range(0, len(crops), batch):
            x = torch.stack([self.tf(c) for c in crops[i:i + batch]]).to(DEVICE)
            out.append(torch.nn.functional.normalize(self.model(x), dim=1).cpu())
        return torch.cat(out).half().numpy()


# ---------------------------------------------------------------- detect
def proposals(result, content):
    """Class-agnostic NMS over YOLOE's boxes, clipped to the photo area, best MAX_CANDIDATES."""
    b = result.boxes
    if b is None or len(b) == 0:
        return []
    xyxy, conf = b.xyxy.cpu(), b.conf.cpu()
    keep = torchvision.ops.nms(xyxy, conf, 0.7)[:MAX_CANDIDATES]
    out = []
    for i in keep.tolist():
        x0, y0, x1, y1 = xyxy[i].tolist()
        box = [max(x0, content[0]), max(y0, content[1]), min(x1, content[2]), min(y1, content[3])]
        if box[2] - box[0] >= 4 and box[3] - box[1] >= 4:
            out.append((box, float(conf[i]), result.names[int(b.cls[i])]))
    return out


def detect(model, scorer, embedder, recs, source, cache_file):
    """Propose boxes for the given photos (all from one class) and score every crop."""
    rows = defaultdict(list)
    for i in range(0, len(recs), DET_BATCH):
        chunk = recs[i:i + DET_BATCH]
        results = model.predict([str(r["cleaned"]) for r in chunk], imgsz=DET_IMGSZ, conf=DET_CONF,
                                max_det=100, device=DEVICE, verbose=False, retina_masks=False)
        for rec, res in zip(chunk, results):
            with Image.open(rec["cleaned"]) as im:
                im = im.convert("RGB")
            content = content_box(im)
            cands = proposals(res, content)
            if source == "pf":                      # whole photo, the fallback candidate
                cands.append((content, 0.0, "(whole photo)"))
            if not cands:
                continue
            crops = [scorer.crop(im, b, content) for b, _, _ in cands]
            p, top = scorer(crops, rec["class_id"])
            emb = embedder(crops)
            for (box, conf, name), pi, ti, ei in zip(cands, p, top, emb):
                for k, v in dict(stem=rec["stem"], box=box, content=content, conf=conf, name=name,
                                 src="full" if name == "(whole photo)" else source,
                                 p=pi, top=ti, dino=ei).items():
                    rows[k].append(v)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, **{k: np.asarray(v) for k, v in rows.items()},
                        stems_done=np.asarray([r["stem"] for r in recs]))


def add_embeddings(f, cleaned_by_stem, embedder):
    """Give a cache written without DINOv2 embeddings its embeddings (no re-detection)."""
    with np.load(f, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files if k != "emb"}
    crops = []
    for stem, box, content in zip(data["stem"], data["box"], data["content"]):
        with Image.open(cleaned_by_stem[str(stem)]) as im:
            crops.append(CropScorer.crop(im.convert("RGB"), box, content))
    data["dino"] = embedder(crops)
    np.savez_compressed(f, **data)


def load_cache(class_name):
    """All candidates of a class (prompt-free + text pass) as arrays."""
    parts = []
    for tag in ("pf", "text"):
        f = OUT_DIR / "cache" / f"{class_name}_{tag}.npz"
        if f.exists():
            with np.load(f, allow_pickle=False) as z:
                if "stem" in z.files:
                    parts.append({k: z[k] for k in z.files if k not in ("stems_done", "emb")})
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# ---------------------------------------------------------------- select
def area_prior(frac):
    return np.clip(np.where(frac > 0.4, 1 - (frac - 0.4) / 0.6 * 0.5, 1.0), 0.5, 1.0) * \
        np.where(frac < 0.003, 0.3, 1.0)


def score_class(c):
    """Score every candidate; returns (score, chosen index per stem, groups, area fraction, consistency)."""
    content_area = (c["content"][:, 2] - c["content"][:, 0]) * (c["content"][:, 3] - c["content"][:, 1])
    frac = (c["box"][:, 2] - c["box"][:, 0]) * (c["box"][:, 3] - c["box"][:, 1]) / content_area
    full = c["src"] == "full"
    # the classifier saw whole scenes, never small objects blown up, so its vote is softened
    base = c["p"] ** 0.25 * np.where(full, 0.15, 0.3 + 0.7 * c["conf"]) * area_prior(frac)
    groups = defaultdict(list)
    for i, s in enumerate(c["stem"]):
        groups[s].append(i)
    groups = {s: np.asarray(ix) for s, ix in groups.items()}
    own = np.empty(len(c["stem"]), dtype=int)
    for g, ix in enumerate(groups.values()):
        own[ix] = g
    emb = c["dino"].astype(np.float32)
    sim = emb @ emb.T

    def consistency(weight):
        """Mean over the class's other photos of the best (weighted) match in that photo."""
        if len(groups) < 2:
            return np.ones(len(own))
        best = np.stack([(sim[:, ix] * weight[ix]).max(1) for ix in groups.values()], 1)
        best[np.arange(len(own)), own] = np.nan
        return np.clip(np.nanmean(best, 1), 0, 1)

    def choose(score):
        return {s: int(ix[np.argmax(score[ix])]) for s, ix in groups.items()}

    weight = np.where(full, 0.0, 1.0)         # whole-photo boxes never serve as evidence
    for _ in range(CONSISTENCY_ITERS):
        cons = consistency(weight)
        score = base * cons ** 2
        best_in_photo = np.zeros_like(score)
        for ix in groups.values():
            best_in_photo[ix] = score[ix].max()
        weight = np.where(full, 0.0, score / np.maximum(best_in_photo, 1e-9))
    chosen = choose(score)
    # "floor + object" boxes: a box inside the pick that recurs more across the class is the object
    # (a logo or other part recurs less than the whole object, so parts are not picked this way)
    for s, i in chosen.items():
        bi = c["box"][i]
        inner = [j for j in groups[s] if j != i and not full[j] and frac[j] < 0.5 * frac[i]
                 and cons[j] > cons[i] + 0.02 and contained(c["box"][j], bi)]
        if inner:
            chosen[s] = max(inner, key=lambda j: cons[j] * (0.3 + 0.7 * c["conf"][j]))
    return score, chosen, groups, frac, cons


def contained(inner, outer, share=0.85):
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy >= share * box_area(inner)


def flag_photo(i, ix, score, c, frac, class_id, log_median):
    flags = []
    if c["src"][i] == "full":
        flags.append("fallback_whole_photo")
    if c["top"][i] != class_id:
        flags.append("classifier_disagrees")
    if c["src"][i] != "full" and c["conf"][i] < 0.25:
        flags.append("low_yolo_conf")
    others = [j for j in ix if j != i and iou(c["box"][i], c["box"][j]) < 0.3
              and c["src"][j] != "full"]
    if others and max(score[j] for j in others) >= 0.8 * score[i]:
        flags.append("ambiguous")
    if abs(np.log(frac[i]) - log_median) > 1.6:
        flags.append("size_outlier")
    if frac[i] > 0.8 and c["src"][i] != "full":
        flags.append("very_large")
    return flags


FLAG_WEIGHT = {"fallback_whole_photo": 1.0, "classifier_disagrees": 0.6, "ambiguous": 0.35,
               "size_outlier": 0.2, "low_yolo_conf": 0.2, "very_large": 0.2}


def read_overrides():
    if not OVERRIDES.exists():
        return {}
    df = pd.read_csv(OVERRIDES, encoding="utf-8-sig")
    return {r["image"]: r for r in df.to_dict("records")}


def select(classes, records, partial=False):
    """Pick a box for every photo; write labels, visuals, boxes.csv, names and the review set."""
    overrides = read_overrides()
    rows, names = [], []
    by_class = defaultdict(list)
    for r in records:
        by_class[r["class_name"]].append(r)
    for class_name, recs in by_class.items():
        c = load_cache(class_name)
        objs = [r for r in recs if not r["is_bg"]]
        picked = {}
        if c is not None:
            score, chosen, groups, frac, sim = score_class(c)
            ok = [i for i in chosen.values() if c["src"][i] != "full"]
            log_median = float(np.median(np.log(frac[ok]))) if ok else 0.0
            ref = float(np.quantile(score[list(chosen.values())], 0.9)) or 1.0
            for stem, i in chosen.items():
                flags = flag_photo(i, groups[stem], score, c, frac, recs[0]["class_id"], log_median)
                risk = (1 - min(score[i] / ref, 1.0)) + sum(FLAG_WEIGHT[f] for f in flags)
                picked[stem] = dict(box=[float(v) for v in c["box"][i]], method=f"auto_{c['src'][i]}",
                                    score=float(score[i]), conf=float(c["conf"][i]), p_class=float(c["p"][i]),
                                    consistency=float(sim[i]), yoloe_name=str(c["name"][i]),
                                    flags=flags, risk=float(risk))
            votes = Counter()
            for i in chosen.values():
                if c["src"][i] == "pf":
                    votes[str(c["name"][i])] += float(score[i])
            total = sum(votes.values()) or 1.0
            top = votes.most_common(3)
            names.append(dict(class_name=class_name, object_id=short(class_name),
                              inferred_name=top[0][0] if top else "",
                              share=round(top[0][1] / total, 3) if top else 0,
                              alternatives="; ".join(f"{n} ({v / total:.2f})" for n, v in top[1:])))
        for r in recs:
            row = dict(image=f"images/{r['key']}.jpg", key=r["key"], class_id=r["class_id"],
                       class_name=short(r["class_name"]), split=r["split"], is_background=r["is_bg"],
                       source=str(r["source"]), cleaned=str(r["cleaned"]))
            p = picked.get(r["stem"])
            if r["is_bg"]:
                row.update(method="background", flags="", risk=0.0)
            elif p is None:
                row.update(method="not_detected", flags="not_detected", risk=9.0)
            else:
                row.update({k: v for k, v in p.items() if k != "box"})
                row.update(dict(zip(("x0", "y0", "x1", "y1"), p["box"])), flags=" ".join(p["flags"]))
            o = overrides.get(r["key"])
            if o is not None:
                if o["action"] == "none":
                    row.update(method="manual_no_object", x0=None, y0=None, x1=None, y1=None)
                elif o["action"] == "exclude":
                    row.update(method="manual_excluded", x0=None, y0=None, x1=None, y1=None)
                else:
                    same = all(row.get(k) is not None and abs(float(o[k]) - float(row[k])) < 0.5
                               for k in ("x0", "y0", "x1", "y1"))
                    row.update({k: float(o[k]) for k in ("x0", "y0", "x1", "y1")},
                               method="verified" if same else "manual")
                row.update(reviewed=True)
            rows.append(row)
    df = pd.DataFrame(rows)
    for col in ("x0", "y0", "x1", "y1", "score", "conf", "p_class", "consistency", "yoloe_name", "reviewed"):
        if col not in df:
            df[col] = None
    df["reviewed"] = df["reviewed"].fillna(False).astype(bool)
    df["has_box"] = df["x0"].notna()
    names_df = pd.DataFrame(names)
    write_outputs(df, names_df, classes, partial)
    return df


# ---------------------------------------------------------------- outputs
def yolo_line(class_id, box, w=ci.SIZE, h=ci.SIZE):
    x0, y0, x1, y1 = box
    return f"{class_id} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}"


def row_box(row):
    return [float(row[k]) for k in ("x0", "y0", "x1", "y1")] if row["has_box"] else None


def link_or_copy(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def font(size):
    return ImageFont.load_default(size=size)


def draw_box(d, box, color, label, width, fnt):
    d.rectangle(box, outline=color, width=width)
    tb = d.textbbox((box[0], box[1]), label, font=fnt)
    y = box[1] - (tb[3] - tb[1]) - 4 if box[1] > tb[3] - tb[1] + 4 else box[1]
    tb = d.textbbox((box[0], y), label, font=fnt)
    d.rectangle([tb[0] - 1, tb[1] - 1, tb[2] + 3, tb[3] + 2], fill=color)
    d.text((box[0] + 1, y), label, fill="black" if color == YELLOW else "white", font=fnt)


def render(row, names, scale=2, alternatives=()):
    """Photo enlarged `scale` times with its box (green; red when flagged) and optional numbered alternatives."""
    with Image.open(row["cleaned"]) as im:
        im = im.convert("RGB").resize((ci.SIZE * scale, ci.SIZE * scale), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(im)
    fnt = font(7 * scale)
    for n, box in alternatives:
        draw_box(d, [v * scale for v in box], YELLOW, str(n), 1, fnt)
    box = row_box(row)
    name = names.get(row["class_name"], "")
    if box:
        color = RED if row["flags"] else GREEN
        label = f"{row['class_name']} {name}".strip()
        draw_box(d, [v * scale for v in box], color, label, max(2, scale), fnt)
    else:
        text = "background (no object)" if row["is_background"] else f"{row['method']}"
        d.text((6, 6), text, fill=RED, font=fnt, stroke_width=2, stroke_fill="white")
    return im


def exclusion(df, chosen):
    """Why a photo is left out of training ("" = used). Background photos are always used."""
    pending = set(chosen.loc[chosen["reason"] == "high_risk", "key"]) if EXCLUDE_UNREVIEWED_HIGH_RISK else set()
    reason = pd.Series("", index=df.index)
    reason[~df["has_box"]] = "no_box"
    reason[df["key"].isin(pending) & ~df["reviewed"]] = "high_risk_not_reviewed"
    reason[df["method"] == "manual_no_object"] = "no_object"
    reason[df["method"] == "manual_excluded"] = "do_not_use"
    reason[df["is_background"]] = ""
    return reason


def write_outputs(df, names_df, classes, partial):
    names = dict(zip(names_df["object_id"], names_df["inferred_name"])) if len(names_df) else {}
    chosen = review_selection(df, partial)
    df["exclude_reason"] = exclusion(df, chosen)
    df["use"] = df["exclude_reason"] == ""
    for row in df.to_dict("records"):
        key = row["key"]
        img = OUT_DIR / "images" / f"{key}.jpg"
        lbl = OUT_DIR / "labels" / f"{key}.txt"
        img.parent.mkdir(parents=True, exist_ok=True)
        lbl.parent.mkdir(parents=True, exist_ok=True)
        link_or_copy(Path(row["cleaned"]), img)
        box = row_box(row)
        lbl.write_text(yolo_line(row["class_id"], box) + "\n" if box else "", encoding="utf-8")
        vis = OUT_DIR / "vis" / f"{key}.jpg"
        vis.parent.mkdir(parents=True, exist_ok=True)
        render(row, names).save(vis, quality=88)
    contact_sheets(df, names)

    # splits: excluded photos are left out (an empty label would teach "background", an unchecked
    # high-risk box may be wrong); generate_multi_object.py follows the same "use" column
    usable = df[df["use"]]
    for split in ("train", "val", "test"):
        lines = [f"./{p}" for p in usable.loc[usable["split"] == split, "image"]]
        (OUT_DIR / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    yaml_names = "\n".join(f"  {i}: {short(n)}" for i, n in enumerate(classes))
    (OUT_DIR / "data.yaml").write_text(f"path: {OUT_DIR.as_posix()}\ntrain: train.txt\nval: val.txt\n"
                                       f"test: test.txt\nnames:\n{yaml_names}\n", encoding="utf-8")
    df.drop(columns=["key"]).to_csv(OUT_DIR / "boxes.csv", index=False, encoding="utf-8-sig")
    names_df.to_csv(OUT_DIR / "class_names_inferred.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "classes.txt").write_text("\n".join(short(n) for n in classes) + "\n", encoding="utf-8")
    write_review(df, names, chosen)

    objs = df[~df["is_background"]]
    summary = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "detector": DET_MODEL, "text_detector": TEXT_MODEL if REFINE else None,
        "classifier": str(CLEAN_RUN / CLASSIFIER), "cleaning": str(CLEAN_RUN),
        "photos": len(df), "object_photos": len(objs), "background_photos": int(df["is_background"].sum()),
        "method_counts": dict(Counter(df["method"])),
        "flag_counts": dict(Counter(f for fl in objs["flags"].fillna("") for f in fl.split())),
        "flagged_photos": int((objs["flags"].fillna("") != "").sum()),
        "reviewed_photos": int(df["reviewed"].sum()),
        "used_photos": int(df["use"].sum()),
        "excluded_photos": dict(Counter(df.loc[~df["use"], "exclude_reason"])),
        "box_area_fraction_median": float(((objs["x1"] - objs["x0"]) * (objs["y1"] - objs["y0"]) / ci.SIZE ** 2).median()),
        "photos_per_split": dict(Counter(usable["split"])),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


def contact_sheets(df, names, cols=10, cell=160):
    out = OUT_DIR / "sheets"
    out.mkdir(exist_ok=True)
    fnt = font(11)
    for class_name, g in df.groupby("class_name", sort=False):
        g = g.sort_values(["is_background", "risk"], ascending=[True, False])
        rows = -(-len(g) // cols)
        sheet = Image.new("RGB", (cols * cell, rows * cell + 30), (237, 239, 242))
        d = ImageDraw.Draw(sheet)
        d.text((8, 8), f"{class_name}  {names.get(class_name, '')}   red = flagged, sorted by risk",
               fill=(25, 30, 40), font=font(14))
        for k, row in enumerate(g.to_dict("records")):
            x, y = (k % cols) * cell, (k // cols) * cell + 30
            thumb = render(row, {}, scale=1).resize((cell - 6, cell - 18))
            sheet.paste(thumb, (x + 3, y + 2))
            if row["flags"]:
                d.rectangle([x + 1, y, x + cell - 2, y + cell - 15], outline=RED, width=2)
            d.text((x + 4, y + cell - 15), Path(row["key"]).name[:22], fill=(25, 30, 40), font=fnt)
        sheet.save(out / f"{class_name}.jpg", quality=85)


# ---------------------------------------------------------------- review set

def review_selection(df, partial):
    """Fixed review set: built once from a full run and kept, so the list never changes under the reviewer.
    A trial run on a few classes (partial) rebuilds its own list every time. Returns rank, key, reason."""
    list_file, scope_file = REVIEW_DIR / "review_list.csv", REVIEW_DIR / "list_scope.txt"
    if REVIEW_DIR.exists() and (partial or not scope_file.exists() or scope_file.read_text().strip() != "full"):
        shutil.rmtree(REVIEW_DIR)
    REVIEW_DIR.mkdir(exist_ok=True)
    scope_file.write_text("partial" if partial else "full")
    objs = df[~df["is_background"]].copy()
    if list_file.exists():
        chosen = pd.read_csv(list_file, encoding="utf-8-sig")[["rank", "key", "reason"]]
    else:
        n = max(1, round(len(objs) * REVIEW_FRAC))
        n_rand = round(n * REVIEW_RANDOM_SHARE)
        risky = objs.sort_values("risk", ascending=False).head(n - n_rand)
        rest = objs.drop(risky.index)
        rand = rest.sample(n=min(n_rand, len(rest)), random_state=SEED)
        chosen = pd.concat([risky.assign(reason="high_risk"), rand.assign(reason="random_spot_check")])
        chosen = chosen.assign(rank=range(1, len(chosen) + 1))[["rank", "key", "reason"]]
    return chosen


def write_review(df, names, chosen):
    list_file = REVIEW_DIR / "review_list.csv"
    merged = chosen.merge(df, on="key", how="left")
    merged["decision"] = merged["method"].where(merged["reviewed"], "")
    cols = ["rank", "key", "reason", "class_name", "split", "flags", "risk", "score", "method",
            "x0", "y0", "x1", "y1", "reviewed", "decision", "use", "image"]
    merged[cols].to_csv(list_file, index=False, encoding="utf-8-sig")
    vis = REVIEW_DIR / "vis"
    vis.mkdir(exist_ok=True)
    for row in merged.to_dict("records"):
        render(row, names).save(vis / f"{row['rank']:04d}_{row['key'].replace('/', '_')}.jpg", quality=90)
    n_risk = int((merged["reason"] == "high_risk").sum())


# ---------------------------------------------------------------- review tool
def save_override(key, action, box):
    rows = [r for r in read_overrides().values() if r["image"] != key]
    rows.append(dict(image=key, action=action, **dict(zip(("x0", "y0", "x1", "y1"),
                     [round(v, 1) for v in box] if box else [None] * 4)),
                     time=datetime.now().isoformat(timespec="seconds")))
    pd.DataFrame(rows, columns=OVERRIDE_COLS).to_csv(OVERRIDES, index=False, encoding="utf-8-sig")


def disable_ime(win):
    """Windows: switch the input method off for the review window, so that letter keys (n, b, s, q)
    reach OpenCV even when a Chinese IME is active (otherwise the IME swallows them)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        user32, imm32 = ctypes.windll.user32, ctypes.windll.imm32
        top = user32.FindWindowW(None, win)
        hwnds = [top]
        proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(
            lambda h, _l: hwnds.append(h) or True)
        user32.EnumChildWindows(top, proc, 0)
        for h in hwnds:
            imm32.ImmAssociateContext(wintypes.HWND(h), None)
    except (OSError, AttributeError):
        pass


def review(classes, records, only_classes):
    import cv2
    df = pd.read_csv(OUT_DIR / "boxes.csv", encoding="utf-8-sig")
    df["key"] = df["image"].str.removeprefix("images/").str.removesuffix(".jpg")
    df["has_box"] = df["x0"].notna()
    df["flags"] = df["flags"].fillna("")
    names_df = pd.read_csv(OUT_DIR / "class_names_inferred.csv", encoding="utf-8-sig")
    names = dict(zip(names_df["object_id"], names_df["inferred_name"].fillna("")))
    if only_classes:
        todo = df[df["class_name"].isin(only_classes) & ~df["is_background"]].sort_values("risk", ascending=False)
    else:
        order = pd.read_csv(REVIEW_DIR / "review_list.csv", encoding="utf-8-sig")
        todo = order[["key"]].merge(df, on="key")
        todo = pd.concat([todo[~todo["reviewed"]], todo[todo["reviewed"]]])      # unreviewed first
    todo = todo.to_dict("records")
    caches = {}
    scale, top = 3, 44
    win = "review - drag = new box | Enter OK | 1-9 proposal | n no object | x do not use | b back | s skip | q quit"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, np.zeros((top + ci.SIZE * scale, ci.SIZE * scale, 3), np.uint8))
    cv2.waitKey(1)
    disable_ime(win)

    # the mouse draws a new box at any time: no key needed first
    drag = {"start": None, "end": None, "active": False, "finished": False}

    def on_mouse(event, x, y, _flags, _param):
        y = max(y, top)                         # not into the header bar
        if event == cv2.EVENT_LBUTTONDOWN:
            drag.update(start=(x, y), end=(x, y), active=True, finished=False)
        elif event == cv2.EVENT_MOUSEMOVE and drag["active"]:
            drag["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag["active"]:
            drag.update(end=(x, y), active=False, finished=True)

    cv2.setMouseCallback(win, on_mouse)
    i, done = 0, 0
    while 0 <= i < len(todo):
        row = todo[i]
        class_dir = "images_" + row["class_name"]
        if class_dir not in caches:
            caches[class_dir] = load_cache(class_dir)
        c = caches[class_dir]
        current = row_box(row)
        alts = []
        if c is not None:
            ix = [j for j in np.flatnonzero(c["stem"] == Path(row["key"]).name) if c["src"][j] != "full"]
            ix.sort(key=lambda j: -c["p"][j] * (0.3 + 0.7 * c["conf"][j]))
            for j in ix:
                b = [float(v) for v in c["box"][j]]
                if (current is None or iou(b, current) < 0.9) and all(iou(b, a) < 0.9 for _, a in alts):
                    alts.append((len(alts) + 1, b))
                if len(alts) == 9:
                    break
        excluded = row["method"] == "manual_excluded"
        frame = None
        drag.update(start=None, active=False, finished=False)
        while True:
            if frame is None:                   # rebuilt only when the current box changes
                if excluded:
                    shown = dict(row, has_box=False, method="DO NOT USE this photo (Enter to save)")
                elif current:
                    shown = dict(row, x0=current[0], y0=current[1], x1=current[2], y1=current[3],
                                 has_box=True, flags="")
                else:
                    shown = dict(row, has_box=False, method="no object (Enter to save)")
                im = render(shown, names, scale=scale, alternatives=alts)
                canvas = Image.new("RGB", (im.width, im.height + top), (30, 30, 30))
                canvas.paste(im, (0, top))
                d = ImageDraw.Draw(canvas)
                d.text((6, 4), f"[{i + 1}/{len(todo)}] {row['key']}  split={row['split']}  "
                               f"{'reviewed' if row['reviewed'] else ''}", fill="white", font=font(14))
                d.text((6, 24), f"flags: {row['flags'] or '-'}   |   drag with the mouse to draw a new box",
                       fill=(255, 180, 180), font=font(13))
                frame = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
            if drag["finished"]:
                (xa, ya), (xb, yb) = drag["start"], drag["end"]
                drag.update(start=None, finished=False)
                if abs(xb - xa) > 2 and abs(yb - ya) > 2:
                    excluded = False
                    current = [min(max(v, 0.0), float(ci.SIZE)) for v in
                               (min(xa, xb) / scale, (min(ya, yb) - top) / scale,
                                max(xa, xb) / scale, (max(ya, yb) - top) / scale)]
                    frame = None
                    continue
            view = frame
            if drag["active"]:
                view = frame.copy()
                cv2.rectangle(view, drag["start"], drag["end"], (255, 255, 0), 2)     # cyan (BGR)
            cv2.imshow(win, view)
            k = cv2.waitKeyEx(15)
            if k == -1:
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    i = -1
                    break
                continue
            ch = chr(k & 0xFF).lower() if k & 0xFF else ""
            if k in (13, 32):
                action = "exclude" if excluded else "box" if current else "none"
                save_override(row["key"], action, current if action == "box" else None)
                box = current if action == "box" else [None] * 4
                row.update(dict(zip(("x0", "y0", "x1", "y1"), box)), has_box=action == "box",
                           method={"exclude": "manual_excluded", "none": "manual_no_object"}.get(action, "manual"))
                row["reviewed"], done, i = True, done + 1, i + 1
                break
            if ch.isdigit() and 1 <= int(ch) <= len(alts):
                current, excluded, frame = alts[int(ch) - 1][1], False, None
            elif ch == "n":
                current, excluded, frame = None, False, None
            elif ch == "x":
                excluded, frame = True, None
            elif ch == "b":
                i = max(0, i - 1)
                break
            elif ch == "s":
                i += 1
                break
            elif ch == "q" or k == 27:
                i = -1
                break
    cv2.destroyAllWindows()
    print(f"Saved {done} decision(s) to {OVERRIDES}; rebuilding outputs...", flush=True)
    scope = (REVIEW_DIR / "list_scope.txt").read_text().strip() if (REVIEW_DIR / "list_scope.txt").exists() else "full"
    if scope == "partial":
        done_classes = set(pd.read_csv(OUT_DIR / "boxes.csv", encoding="utf-8-sig")["class_name"])
        records = [r for r in records if short(r["class_name"]) in done_classes]
    select(classes, records, partial=scope == "partial")


# ---------------------------------------------------------------- main
def run_detect(classes, records, only):
    recs_by_class = defaultdict(list)
    for r in records:
        if not r["is_bg"] and (not only or short(r["class_name"]) in only):
            recs_by_class[r["class_name"]].append(r)
    model = scorer = embedder = None
    for n, (class_name, recs) in enumerate(recs_by_class.items(), 1):
        cache = OUT_DIR / "cache" / f"{class_name}_pf.npz"
        if cache.exists():
            with np.load(cache) as z:
                has_dino = "dino" in z.files
            if not has_dino:
                embedder = embedder or Embedder()
                add_embeddings(cache, {r["stem"]: r["cleaned"] for r in recs}, embedder)
                print(f"[embed] {cache.name}", flush=True)
            continue
        model = model or yoloe(DET_MODEL)
        scorer = scorer or CropScorer(classes)
        embedder = embedder or Embedder()
        t = datetime.now()
        detect(model, scorer, embedder, recs, "pf", cache)
        print(f"[detect {n}/{len(recs_by_class)}] {class_name}: {len(recs)} photos "
              f"in {(datetime.now() - t).total_seconds():.0f}s", flush=True)


def run_refine(classes, records, df, only):
    """Text-prompt YOLOE with the inferred class name on photos that are still flagged."""
    names = pd.read_csv(OUT_DIR / "class_names_inferred.csv", encoding="utf-8-sig")
    flagged = df[(df["flags"].fillna("") != "") & ~df["reviewed"] & ~df["is_background"]]
    by_key = {r["key"]: r for r in records}
    model = scorer = embedder = None
    for row in names.to_dict("records"):
        class_name, prompts = row["class_name"], []
        if only and short(class_name) not in only:
            continue
        cache = OUT_DIR / "cache" / f"{class_name}_text.npz"
        if cache.exists():
            continue
        for text in [row["inferred_name"]] + [a.rsplit(" (", 1)[0] for a in str(row["alternatives"]).split("; ")][:1]:
            if isinstance(text, str) and text and text != "nan" and text not in prompts:
                prompts.append(text)
        recs = [by_key[k] for k in flagged.loc[flagged["class_name"] == short(class_name), "key"]]
        if not recs or not prompts:
            continue
        model = model or yoloe(TEXT_MODEL)
        scorer = scorer or CropScorer(classes)
        embedder = embedder or Embedder()
        model.set_classes(prompts)
        detect(model, scorer, embedder, recs, "text", cache)
        print(f"[refine] {class_name}: {len(recs)} flagged photos, prompts {prompts}", flush=True)


def main():
    mode = RUN_MODE
    if mode not in ("all", "detect", "select", "refine", "review"):
        raise ValueError("RUN_MODE must be 'all', 'detect', 'select', 'refine' or 'review'")
    if isinstance(ONLY_CLASSES, str):
        raise TypeError('ONLY_CLASSES must be a list, e.g. [] or ["OBJ001", "OBJ009"]')
    only = {short(c) for c in ONLY_CLASSES}
    random.seed(SEED)
    print(f"Mode: {mode}\nClasses: {sorted(only) or 'all'}\nDevice: {DEVICE}\nOutput: {OUT_DIR}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    complete = output_complete()
    classes, records = load_records()
    if mode == "review":
        review(classes, records, only)
        return
    if mode in ("all", "detect"):
        run_detect(classes, records, only)
    if mode == "detect":
        return
    # once every class is labelled, outputs are always rebuilt for all classes (ONLY_CLASSES then only
    # limits detection / refinement), so boxes.csv and the review list never shrink to a subset
    partial = bool(only) and not complete
    sel = [r for r in records if not partial or short(r["class_name"]) in only]
    df = select(classes, sel, partial=partial)
    if mode in ("all", "refine") and REFINE:
        run_refine(classes, sel, df, only)
        select(classes, sel, partial=partial)


if __name__ == "__main__":
    main()
