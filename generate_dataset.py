#!/usr/bin/env python3
import argparse
import csv
import hashlib
import io
import itertools
import json
import random
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps, __version__ as PILLOW_VERSION

CLASSES = [
    ("cup", "杯子", "cup_dataset_100.zip", ["1(5).jpg", "2(7).jpg", "3(7).jpg", "4(5).jpg", "5(5).jpg", "6(4).jpg"]),
    ("mouse", "鼠标", "mouse_dataset_100.zip", ["1(6).jpg", "2(8).jpg", "3(8).jpg", "5(6).jpg", "6(5).jpg"]),
    ("medicine_box", "药盒", "dages_dataset.zip", ["1(7).jpg", "2(9).jpg", "3(9).jpg", "4(7).jpg", "5(7).jpg", "6(6).jpg"]),
    ("phone_stand", "手机支架", "phone_stand_dataset_100_revised.zip", ["1(8).jpg", "2(10).jpg", "3(10).jpg", "4(8).jpg", "5(8).jpg", "5(9).jpg"]),
]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_catalog(archives, originals):
    pools = defaultdict(list)
    duplicates, negatives, archive_hashes = [], [], {}
    seen_hashes = {}
    for class_id, (name, zh, archive, filenames) in enumerate(CLASSES):
        archive_path = archives / archive
        archive_hashes[archive] = sha(archive_path.read_bytes())
        with ZipFile(archive_path) as z:
            manifest = next(n for n in z.namelist() if n.endswith(".csv"))
            folder = PurePosixPath(manifest).parent
            rows = csv.DictReader(io.StringIO(z.read(manifest).decode("utf-8-sig")))
            for row in rows:
                filename = row.get("filename") or row["file"]
                member = str(folder / (PurePosixPath(filename) if "/" in filename else PurePosixPath("images") / filename))
                if row["target_present"] != "1":
                    negatives.append({"archive": archive, "member": member})
                    continue
                data = z.read(member)
                digest = sha(data)
                if row.get("sha256") and row["sha256"] != digest:
                    raise ValueError(f"Source checksum mismatch: {member}")
                with Image.open(io.BytesIO(data)) as im:
                    w, h = ImageOps.exif_transpose(im).size
                item = {
                    "source_id": f"{name}_synthetic_{int(row['id']):03d}",
                    "class_id": class_id, "class_name": name, "class_zh": zh,
                    "source_kind": "ai_generated", "archive": archive, "archive_member": member,
                    "original_filename": "", "source_width": w, "source_height": h,
                    "distance": {"near": "close"}.get(row.get("distance_group"), row.get("distance_group") or row.get("distance", "")),
                    "scene": row.get("scene") or row.get("scene_id", ""),
                    "view_design": row.get("angle") or row.get("requested_view") or row.get("requested_pose", ""),
                    "lighting_design": row.get("lighting") or row.get("lighting_prompt") or row.get("requested_lighting", ""),
                    "sha256": digest,
                }
                if digest in seen_hashes:
                    duplicates.append({"excluded": item["source_id"], "same_as": seen_hashes[digest]})
                else:
                    seen_hashes[digest] = item["source_id"]
                    pools[class_id].append(item)
        for filename in filenames:
            data = (originals / filename).read_bytes()
            digest = sha(data)
            if digest in seen_hashes:
                duplicates.append({"excluded": filename, "same_as": seen_hashes[digest]})
                continue
            with Image.open(io.BytesIO(data)) as im:
                w, h = ImageOps.exif_transpose(im).size
            item = {
                "source_id": f"{name}_original_{filename}",
                "class_id": class_id, "class_name": name, "class_zh": zh,
                "source_kind": "original_photo", "archive": "", "archive_member": "",
                "original_filename": filename, "source_width": w, "source_height": h,
                "distance": "unmeasured_original", "scene": "original_photo_background",
                "view_design": "", "lighting_design": "", "sha256": digest,
            }
            seen_hashes[digest] = item["source_id"]
            pools[class_id].append(item)
    return pools, duplicates, negatives, archive_hashes


def scene_balanced_pick(items, count, rng):
    buckets = defaultdict(list)
    for item in items:
        buckets[item["scene"]].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < count:
        scenes = [scene for scene, bucket in buckets.items() if bucket]
        if not scenes:
            raise ValueError("Not enough distinct source photos")
        rng.shuffle(scenes)
        for scene in scenes:
            if len(selected) == count:
                break
            selected.append(buckets[scene].pop())
    return selected


def leaf(index, item):
    return {"leaf": index, "aspect": item["source_width"] / item["source_height"]}


def join(axis, children):
    aspect = sum(c["aspect"] for c in children) if axis == "h" else 1 / sum(1 / c["aspect"] for c in children)
    return {"axis": axis, "children": children, "aspect": aspect}


def make_layout(items, rng):
    leaves = [leaf(i, item) for i, item in enumerate(items)]
    n = len(leaves)
    if n == 2:
        axis = rng.choice(["h", "h", "h", "v"])
        return join(axis, leaves), "two_columns" if axis == "h" else "two_rows"
    if n == 3:
        axis = rng.choice(["h", "v"])
        children = [leaves[0], join("v" if axis == "h" else "h", leaves[1:])]
        if rng.random() < 0.5:
            children.reverse()
        return join(axis, children), "three_asymmetric_" + axis
    axis = rng.choice(["h", "v"])
    other = "v" if axis == "h" else "h"
    return join(axis, [join(other, leaves[:2]), join(other, leaves[2:])]), "four_grid_" + axis


def rectangles(tree, x, y, w, h, rects):
    if "leaf" in tree:
        rects[tree["leaf"]] = [x, y, w, h]
        return
    horizontal = tree["axis"] == "h"
    weights = [c["aspect"] if horizontal else 1 / c["aspect"] for c in tree["children"]]
    length, total, offset, cumulative = w if horizontal else h, sum(weights), 0, 0
    for child, weight in zip(tree["children"], weights):
        cumulative += weight
        boundary = round(length * cumulative / total)
        extent = boundary - offset
        rectangles(child, x + offset if horizontal else x, y if horizontal else y + offset,
                   extent if horizontal else w, h if horizontal else extent, rects)
        offset = boundary


def get_photo(item, archives, originals):
    if item["source_kind"] == "original_photo":
        data = (originals / item["original_filename"]).read_bytes()
    else:
        with ZipFile(archives / item["archive"]) as z:
            data = z.read(item["archive_member"])
    with Image.open(io.BytesIO(data)) as im:
        return ImageOps.exif_transpose(im).convert("RGB")


def contact_sheets(output, records):
    preview_dir = output / "previews"
    preview_dir.mkdir(exist_ok=True)
    font = ImageFont.load_default(size=18)
    for page in range(4):
        canvas = Image.new("RGB", (2000, 1850), (237, 239, 242))
        draw = ImageDraw.Draw(canvas)
        for k, record in enumerate(records[page * 25:(page + 1) * 25]):
            x, y = (k % 5) * 400, (k // 5) * 370
            with Image.open(output / record["filename"]) as im:
                thumb = ImageOps.contain(im, (384, 326), Image.Resampling.LANCZOS)
                canvas.paste(thumb, (x + (400 - thumb.width) // 2, y + 8 + (326 - thumb.height) // 2))
            draw.text((x + 10, y + 340), f"{record['id']:03d} | {record['target_count']} targets", fill=(25, 30, 40), font=font)
        canvas.save(preview_dir / f"overview_{page + 1:02d}.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zips", type=Path, required=True)
    parser.add_argument("--originals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    output = args.output
    (output / "images").mkdir(parents=True, exist_ok=True)
    if list((output / "images").glob("*.jpg")):
        raise ValueError("Output images directory must be empty to avoid overwriting a dataset")
    pools, duplicates, negatives, archive_hashes = load_catalog(args.source_zips, args.originals)
    queues = {}
    for class_id in range(4):
        originals = [r for r in pools[class_id] if r["source_kind"] == "original_photo"]
        chosen = list(originals)
        quotas = {"close": 26, "medium": 26, "far": 75 - len(originals) - 52}
        for distance, count in quotas.items():
            eligible = [r for r in pools[class_id] if r["source_kind"] == "ai_generated" and r["distance"] == distance]
            chosen.extend(scene_balanced_pick(eligible, count, rng))
        rng.shuffle(chosen)
        queues[class_id] = chosen
    selected = [r for queue in queues.values() for r in queue]
    assert len(selected) == len({r["sha256"] for r in selected}) == 300
    combinations = []
    for pair in itertools.combinations(range(4), 2):
        combinations.extend([pair] * (5 if pair in [(0, 1), (2, 3)] else 6))
    for triple in itertools.combinations(range(4), 3):
        combinations.extend([triple] * 8)
    combinations.extend([(0, 1, 2, 3)] * 34)
    rng.shuffle(combinations)
    assert len(combinations) == 100
    assert Counter(c for group in combinations for c in group) == Counter({c: 75 for c in range(4)})
    records, labels = [], []
    for image_id, group in enumerate(combinations, 1):
        class_ids = list(group)
        rng.shuffle(class_ids)
        items = [queues[c].pop() for c in class_ids]
        tree, layout_name = make_layout(items, rng)
        aspect = tree["aspect"]
        w, h = (2048, round(2048 / aspect)) if aspect >= 1 else (round(2048 * aspect), 2048)
        rects = {}
        rectangles(tree, 0, 0, w, h, rects)
        canvas = Image.new("RGB", (w, h))
        tiles = []
        for index, item in enumerate(items):
            x, y, pw, ph = rects[index]
            im = get_photo(item, args.source_zips, args.originals)
            # Dimensions arise from proportional concatenation. Integer rounding
            # introduces at most a pixel or two, never a crop or content-aware warp.
            assert abs((pw / ph) / (im.width / im.height) - 1) < 0.005
            canvas.paste(im.resize((pw, ph), Image.Resampling.LANCZOS), (x, y))
            tiles.append({"source_id": item["source_id"], "class_id": item["class_id"],
                          "class_name": item["class_name"], "source_kind": item["source_kind"],
                          "source_photo_rectangle_xywh": [x, y, pw, ph]})
        # Mild whole-composite photometric augmentation; text is never mirrored.
        augmented = rng.random() < 0.45
        brightness = round(rng.uniform(0.94, 1.06), 4) if augmented else 1.0
        contrast = round(rng.uniform(0.94, 1.06), 4) if augmented else 1.0
        if augmented:
            canvas = ImageEnhance.Brightness(canvas).enhance(brightness)
            canvas = ImageEnhance.Contrast(canvas).enhance(contrast)
        filename = f"images/multi_{image_id:03d}.jpg"
        destination = output / filename
        encoded = io.BytesIO()
        canvas.save(encoded, format="JPEG", quality=93, subsampling=0, optimize=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(encoded.getvalue())
        temporary.replace(destination)
        record = {"id": image_id, "filename": filename, "width": w, "height": h,
                  "target_count": len(items), "layout": layout_name, "tiles": tiles,
                  "augmentation": {"brightness": brightness, "contrast": contrast, "mirror": False, "crop": False},
                  "sha256": sha(destination.read_bytes())}
        records.append(record)
        label = {"filename": filename, "width": w, "height": h, "target_count": len(items)}
        label.update({name: int(c in class_ids) for c, (name, _, _, _) in enumerate(CLASSES)})
        labels.append(label)
        if image_id % 20 == 0:
            print(f"Generated {image_id}/100", flush=True)
    assert not any(queues.values())
    write_csv(output / "image_labels.csv", labels)
    write_csv(output / "selected_sources.csv", sorted(selected, key=lambda r: r["source_id"]))
    with (output / "composition_manifest.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(output / "classes.json", [{"id": c, "name": name, "name_zh": zh} for c, (name, zh, _, _) in enumerate(CLASSES)])
    write_json(output / "source_audit.json", {"archives_sha256": archive_hashes, "excluded_duplicates": duplicates,
                "excluded_background_only": negatives,
                "available_positive_photos_per_class": {CLASSES[c][0]: len(pools[c]) for c in range(4)}})
    summary = {"image_count": len(records), "seed": args.seed, "pillow_version": PILLOW_VERSION,
               "long_edge_px": 2048, "preserve_full_source_frame": True,
               "target_count_distribution": dict(Counter(r["target_count"] for r in records)),
               "target_occurrences_per_class": {CLASSES[c][0]: 75 for c in range(4)},
               "selected_unique_source_photos": len(selected),
               "source_kind_distribution": dict(Counter(r["source_kind"] for r in selected)),
               "synthetic_distance_distribution": dict(Counter(r["distance"] for r in selected if r["source_kind"] == "ai_generated")),
               "photometrically_augmented_images": sum(r["augmentation"]["brightness"] != 1 for r in records),
               "has_object_bounding_boxes": False}
    write_json(output / "dataset_summary.json", summary)
    contact_sheets(output, records)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
