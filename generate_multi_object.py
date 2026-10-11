import hashlib
import io
import json
import random
import shutil
from collections import Counter, deque
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

import class_identification as ci
import generate_dataset as layouts         # make_layout() / rectangles() of the 100-image generator

NUM_IMAGES = 10_000
TARGET_COUNTS = (2, 3, 4)           # objects per image, drawn uniformly
LONG_EDGE = 640                     # output long side in px (YOLO's default input size)
BG_TILE_PROB = 0.15                 # chance of one extra background-only tile (when < 4 objects)
FLIP_PROB = 0.5                     # per-tile horizontal mirror
AUGMENT_PROB = 0.45                 # mild brightness/contrast jitter, as in generate_dataset.py
SEED = 20261003
PREVIEW_PAGES = 4
VIS_COUNT = 300                     # composites also saved with their boxes drawn

HERE = Path(__file__).resolve().parent
LABEL_DIR = HERE / "photo" / "single_object_labels"
OUT_DIR = HERE / "photo" / "multi_object" / f"multi_object_det_{NUM_IMAGES}"
SPLITS = {"train": 1 - ci.VAL_FRAC - ci.TEST_FRAC, "val": ci.VAL_FRAC, "test": ci.TEST_FRAC}
PALETTE = [(230, 40, 40), (40, 120, 230), (40, 180, 70), (230, 140, 20), (150, 60, 200)]


def load_photos():
    boxes = pd.read_csv(LABEL_DIR / "boxes.csv", encoding="utf-8-sig")
    classes = (LABEL_DIR / "classes.txt").read_text(encoding="utf-8").split()
    if boxes["class_name"].nunique() != len(classes):
        raise ValueError(f"{LABEL_DIR / 'boxes.csv'} covers {boxes['class_name'].nunique()} of {len(classes)} "
                         "classes; run annotate_single_object.py on all classes first")
    if "use" not in boxes:
        raise ValueError("boxes.csv has no 'use' column: set RUN_MODE = 'select' in "
                         "annotate_single_object.py and run it first")
    objs = boxes[~boxes["is_background"] & boxes["use"]]
    excluded = Counter(boxes.loc[~boxes["use"], "exclude_reason"])
    pools = {s: {c: [] for c in range(len(classes))} for s in SPLITS}
    bg = {s: [] for s in SPLITS}
    for r in objs.to_dict("records"):
        pools[r["split"]][r["class_id"]].append(r)
    for r in boxes[boxes["is_background"]].to_dict("records"):
        bg[r["split"]].append(r)
    empty = [(s, classes[c]) for s, by in pools.items() for c, ps in by.items() if not ps]
    if empty:
        raise ValueError(f"No usable photos for {empty[:5]}")
    return classes, pools, bg, excluded


class Deck:
    def __init__(self, items, rng):
        self.items, self.rng, self.queue = list(items), rng, deque()

    def draw(self, exclude=()):
        if len(self.queue) <= len(exclude) + 1:
            refill = list(self.items)
            self.rng.shuffle(refill)
            self.queue.extend(refill)
        for i, item in enumerate(self.queue):
            if item not in exclude:
                del self.queue[i]
                return item
        raise ValueError("Not enough distinct items to draw from")


def compose(gen, items, rng):
    tree, layout = gen.make_layout(items, rng)
    aspect = tree["aspect"]
    w, h = (LONG_EDGE, round(LONG_EDGE / aspect)) if aspect >= 1 else (round(LONG_EDGE * aspect), LONG_EDGE)
    rects = {}
    gen.rectangles(tree, 0, 0, w, h, rects)
    canvas = Image.new("RGB", (w, h))
    tiles = []
    for index, item in enumerate(items):
        x, y, pw, ph = rects[index]
        flip = rng.random() < FLIP_PROB
        with Image.open(LABEL_DIR / item["image"]) as im:
            im = im.convert("RGB")
            if flip:
                im = ImageOps.mirror(im)
            canvas.paste(im.resize((pw, ph), Image.Resampling.LANCZOS), (x, y))
        tile = {"source": item["source"], "image": item["image"], "class_id": item["class_id"],
                "class_name": item["class_name"], "background_only": item["is_background"], "mirrored": flip,
                "source_photo_rectangle_xywh": [x, y, pw, ph], "box_xyxy": None}
        if not item["is_background"]:
            x0, y0, x1, y1 = item["x0"], item["y0"], item["x1"], item["y1"]
            if flip:
                x0, x1 = ci.SIZE - x1, ci.SIZE - x0
            sx, sy = pw / ci.SIZE, ph / ci.SIZE
            tile["box_xyxy"] = [round(x + x0 * sx, 2), round(y + y0 * sy, 2),
                                round(x + x1 * sx, 2), round(y + y1 * sy, 2)]
        tiles.append(tile)
    augmented = rng.random() < AUGMENT_PROB
    brightness = round(rng.uniform(0.94, 1.06), 4) if augmented else 1.0
    contrast = round(rng.uniform(0.94, 1.06), 4) if augmented else 1.0
    if augmented:
        canvas = ImageEnhance.Brightness(canvas).enhance(brightness)
        canvas = ImageEnhance.Contrast(canvas).enhance(contrast)
    return canvas, layout, tiles, {"brightness": brightness, "contrast": contrast}


def draw_boxes(im, tiles, width=3, size=16):
    im = im.copy()
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=size)
    for k, t in enumerate(t for t in tiles if t["box_xyxy"]):
        color = PALETTE[k % len(PALETTE)]
        x0, y0, x1, y1 = t["box_xyxy"]
        d.rectangle([x0, y0, x1, y1], outline=color, width=width)
        tb = d.textbbox((x0, y0), t["class_name"], font=font)
        d.rectangle([tb[0], tb[1], tb[2] + 4, tb[3] + 2], fill=color)
        d.text((x0 + 2, y0), t["class_name"], fill="white", font=font)
    return im


def contact_sheets(records, work):
    preview_dir = work / "previews"
    preview_dir.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=14)
    for page in range(PREVIEW_PAGES):
        canvas = Image.new("RGB", (1500, 1500), (237, 239, 242))
        draw = ImageDraw.Draw(canvas)
        for k, record in enumerate(records[page * 25:(page + 1) * 25]):
            x, y = (k % 5) * 300, (k // 5) * 300
            with Image.open(work / record["filename"]) as im:
                boxed = draw_boxes(im.convert("RGB"), record["tiles"], width=4, size=22)
            thumb = ImageOps.contain(boxed, (284, 260), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x + (300 - thumb.width) // 2, y + 6 + (260 - thumb.height) // 2))
            names = ",".join(t["class_name"] for t in record["tiles"] if t["box_xyxy"])
            bg = " +bg" if any(t["background_only"] for t in record["tiles"]) else ""
            draw.text((x + 8, y + 274), f"{record['id']:05d} {names}{bg}", fill=(25, 30, 40), font=font)
        canvas.save(preview_dir / f"overview_{page + 1:02d}.jpg", quality=90)


def replace_dir(new, final):
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    try:
        if final.exists():
            final.rename(old)
        new.rename(final)
    except OSError as exc:
        if old.exists() and not final.exists():
            old.rename(final)
        raise RuntimeError(f"Could not replace {final} ({exc}). Close programs using it and run again; "
                           f"the new dataset is in {new}") from exc
    if old.exists():
        shutil.rmtree(old)


def main():
    # build in a side folder; the finished dataset replaces OUT_DIR only at the very end
    work = OUT_DIR.with_name(OUT_DIR.name + ".partial")
    if work.exists():
        shutil.rmtree(work)                 # left over from an interrupted run
    work.mkdir(parents=True)
    classes, pools, bg_pools, excluded = load_photos()
    if max(TARGET_COUNTS) > len(classes):
        raise ValueError("TARGET_COUNTS exceeds the number of classes")
    gen = layouts
    rng = random.Random(SEED)

    counts = {"val": round(NUM_IMAGES * ci.VAL_FRAC), "test": round(NUM_IMAGES * ci.TEST_FRAC)}
    counts = {"train": NUM_IMAGES - counts["val"] - counts["test"], **counts}
    records, gt_rows, seen = [], [], set()
    image_id = 0
    for split, n_images in counts.items():
        class_deck = Deck(range(len(classes)), rng)
        source_decks = {c: Deck(range(len(ps)), rng) for c, ps in pools[split].items()}
        bg_deck = Deck(range(len(bg_pools[split])), rng) if bg_pools[split] else None
        for d in ("images", "labels"):
            (work / d / split).mkdir(parents=True, exist_ok=True)
        for _ in range(n_images):
            for _attempt in range(100):
                k = rng.choice(TARGET_COUNTS)
                class_ids = []
                for _ in range(k):
                    class_ids.append(class_deck.draw(exclude=class_ids))
                items = [pools[split][c][source_decks[c].draw()] for c in class_ids]
                if bg_deck and k < 4 and rng.random() < BG_TILE_PROB:
                    items.insert(rng.randrange(k + 1), bg_pools[split][bg_deck.draw()])
                key = tuple(sorted(item["image"] for item in items))
                if key not in seen:
                    seen.add(key)
                    break
            else:
                raise RuntimeError("Could not find a new source combination after 100 attempts")
            for item in items:      # gen.make_layout reads the photo shape
                item["source_width"] = item["source_height"] = ci.SIZE

            image_id += 1
            canvas, layout, tiles, augmentation = compose(gen, items, rng)
            stem = f"multi_{image_id:05d}"
            filename = f"images/{split}/{stem}.jpg"
            encoded = io.BytesIO()
            canvas.save(encoded, format="JPEG", quality=ci.JPEG_QUALITY, subsampling=0)
            (work / filename).write_bytes(encoded.getvalue())
            W, H = canvas.size
            lines = []
            for t in tiles:
                if not t["box_xyxy"]:
                    continue
                x0, y0, x1, y1 = t["box_xyxy"]
                lines.append(f"{t['class_id']} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                             f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
                gt_rows.append(dict(image=f"{stem}.jpg", split=split, class_id=t["class_id"],
                                    class_name=t["class_name"], x0=x0, y0=y0, x1=x1, y1=y1,
                                    box_w=round(x1 - x0, 2), box_h=round(y1 - y0, 2), source=t["image"]))
            (work / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            record = {"id": image_id, "filename": filename, "split": split, "width": W, "height": H,
                      "target_count": k, "background_tiles": len(tiles) - k, "layout": layout,
                      "tiles": tiles, "augmentation": augmentation,
                      "sha256": hashlib.sha256(encoded.getvalue()).hexdigest()}
            records.append(record)
            if image_id <= VIS_COUNT:
                (work / "vis").mkdir(exist_ok=True)
                draw_boxes(canvas, tiles).save(work / "vis" / f"{stem}.jpg", quality=90)
            if image_id % 500 == 0:
                print(f"Generated {image_id}/{NUM_IMAGES}", flush=True)

    assert len({r["sha256"] for r in records}) == len(records), "Duplicate output images"
    gt = pd.DataFrame(gt_rows)
    gt.to_csv(work / "gt_boxes.csv", index=False, encoding="utf-8-sig")
    with (work / "composition_manifest.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    (work / "classes.json").write_text(
        json.dumps({name: idx for idx, name in enumerate(classes)}, indent=2, ensure_ascii=False), encoding="utf-8")
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(classes))
    (work / "data.yaml").write_text(f"path: {OUT_DIR.as_posix()}\ntrain: images/train\nval: images/val\n"
                                       f"test: images/test\nnames:\n{names}\n", encoding="utf-8")

    occurrences = Counter(gt["class_name"])
    source_uses = Counter(gt["source"])
    short_side = gt[["box_w", "box_h"]].min(axis=1)
    summary = {
        "image_count": len(records), "seed": SEED, "long_edge_px": LONG_EDGE,
        "box_source": str(LABEL_DIR / "boxes.csv"),
        "photos_excluded": dict(excluded),   # high_risk_not_reviewed / no_object / do_not_use
        "images_per_split": dict(Counter(r["split"] for r in records)),
        "object_photos_per_split": {s: sum(map(len, by.values())) for s, by in pools.items()},
        "background_photos_per_split": {s: len(v) for s, v in bg_pools.items()},
        "target_count_distribution": {k: v for k, v in sorted(Counter(r["target_count"] for r in records).items())},
        "images_with_background_tile": sum(r["background_tiles"] > 0 for r in records),
        "layout_distribution": dict(Counter(r["layout"] for r in records)),
        "boxes": len(gt),
        "class_occurrences_min_max": [min(occurrences.values()), max(occurrences.values())],
        "source_photos_used": len(source_uses),
        "source_reuse_min_max": [min(source_uses.values()), max(source_uses.values())],
        "box_short_side_px_percentiles_5_50_95": [round(float(short_side.quantile(q)), 1) for q in (0.05, 0.5, 0.95)],
        "boxes_short_side_under_8px": int((short_side < 8).sum()),
        "mirrored_tiles": sum(t["mirrored"] for r in records for t in r["tiles"]),
        "photometrically_augmented_images": sum(r["augmentation"]["brightness"] != 1 for r in records),
    }
    (work / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    contact_sheets(records, work)
    replace_dir(work, OUT_DIR)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Done: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
