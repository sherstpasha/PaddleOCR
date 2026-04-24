import json
import math
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml


# =========================
# Edit only this section
# =========================

# Add as many COCO datasets as needed. Each item becomes part of one shared
# PaddleOCR detection label file. Use split="train" for training data and
# split="eval" for validation/test data.
DATASETS = [
    {
        "name": "archives020525_train",
        "annotation": r"C:\shared\data02065\d2\Archives020525\train.json",
        "image_dir": r"C:\shared\data02065\d2\Archives020525\train_images",
        "split": "train",
    },
    {
        "name": "archives020525_eval",
        "annotation": r"C:\shared\data02065\d2\Archives020525\test.json",
        "image_dir": r"C:\shared\data02065\d2\Archives020525\test_images",
        "split": "eval",
    },
    {
        "name": "ddi_100_train",
        "annotation": r"C:\shared\data02065\d2\DDI_100\train.json",
        "image_dir": r"C:\shared\data02065\d2\DDI_100\train_images",
        "split": "train",
    },
    {
        "name": "ddi_100_eval",
        "annotation": r"C:\shared\data02065\d2\DDI_100\test.json",
        "image_dir": r"C:\shared\data02065\d2\DDI_100\test_images",
        "split": "eval",
    },
    {
        "name": "icdar2015_train",
        "annotation": r"C:\shared\data02065\d2\ICDAR2015\train.json",
        "image_dir": r"C:\shared\data02065\d2\ICDAR2015\train_images",
        "split": "train",
    },
    {
        "name": "icdar2015_eval",
        "annotation": r"C:\shared\data02065\d2\ICDAR2015\test.json",
        "image_dir": r"C:\shared\data02065\d2\ICDAR2015\test_images",
        "split": "eval",
    },
    {
        "name": "school_notebooks_ru_train",
        "annotation": r"C:\shared\data02065\d2\school_notebooks_RU\train.json",
        "image_dir": r"C:\shared\data02065\d2\school_notebooks_RU\train_images",
        "split": "train",
    },
    {
        "name": "school_notebooks_ru_eval",
        "annotation": r"C:\shared\data02065\d2\school_notebooks_RU\test.json",
        "image_dir": r"C:\shared\data02065\d2\school_notebooks_RU\test_images",
        "split": "eval",
    },
    {
        "name": "totaltext_train",
        "annotation": r"C:\shared\data02065\d2\TotalText\train.json",
        "image_dir": r"C:\shared\data02065\d2\TotalText\train_images",
        "split": "train",
    },
    {
        "name": "totaltext_eval",
        "annotation": r"C:\shared\data02065\d2\TotalText\test.json",
        "image_dir": r"C:\shared\data02065\d2\TotalText\test_images",
        "split": "eval",
    },
]

# mobile is lighter and usually easier to start with. Change to "server" for
# the larger PP-OCRv5 detector.
MODEL_SIZE = "mobile"  # "mobile" or "server"

PREPARED_DATA_DIR = "train_data/custom_ppocrv5_det_coco"
SAVE_MODEL_DIR = "output/custom_ppocrv5_det_coco"

TRAIN_RATIO_IF_NO_EVAL = 0.9
SEED = 42
TRAIN_IMAGE_SIZE = 1408

# Leave empty for normal single-process training. Set "0" or "0,1" to run
# through paddle.distributed.launch.
GPUS = ""

EPOCH_NUM = 200
BATCH_SIZE_PER_CARD = 1
TRAIN_NUM_WORKERS = None
EVAL_NUM_WORKERS = None
LEARNING_RATE = 0.0005
EVAL_BATCH_STEP = 1500
PRETRAINED_MODEL = None
CHECKPOINTS = None

# If True, the script prepares labels and immediately starts training.
# If False, it only prepares files and prints the command.
RUN_TRAIN = True


# =========================
# Internal code
# =========================

REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / path).resolve()


def to_override_path(path_obj):
    path_obj = path_obj.resolve()
    try:
        return path_obj.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path_obj.as_posix()


def polygon_area(points):
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, (x1, y1) in enumerate(points):
        x2, y2 = points[(idx + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def order_quad_points(points):
    points = [[float(x), float(y)] for x, y in points]
    ordered = [[0.0, 0.0] for _ in range(4)]
    sums = [x + y for x, y in points]
    diffs = [x - y for x, y in points]

    ordered[0] = points[sums.index(min(sums))]
    ordered[2] = points[sums.index(max(sums))]
    ordered[1] = points[diffs.index(max(diffs))]
    ordered[3] = points[diffs.index(min(diffs))]
    return [[round(x, 2), round(y, 2)] for x, y in ordered]


def normalize_polygon_points(points):
    if len(points) == 4:
        points = order_quad_points(points)
    if polygon_area(points) <= 1.0:
        return []
    return points


def flat_polygon_to_points(poly):
    if not isinstance(poly, list) or len(poly) < 8 or len(poly) % 2 != 0:
        return []
    points = []
    for idx in range(0, len(poly), 2):
        try:
            x = float(poly[idx])
            y = float(poly[idx + 1])
        except (TypeError, ValueError):
            return []
        if not math.isfinite(x) or not math.isfinite(y):
            return []
        points.append([round(x, 2), round(y, 2)])
    return normalize_polygon_points(points)


def bbox_to_points(bbox):
    if not isinstance(bbox, list) or len(bbox) < 4:
        return []
    try:
        x, y, w, h = [float(v) for v in bbox[:4]]
    except (TypeError, ValueError):
        return []
    if w <= 1 or h <= 1:
        return []
    return [
        [round(x, 2), round(y, 2)],
        [round(x + w, 2), round(y, 2)],
        [round(x + w, 2), round(y + h, 2)],
        [round(x, 2), round(y + h, 2)],
    ]


def annotation_to_points(annotation):
    segmentation = annotation.get("segmentation")
    candidates = []
    if isinstance(segmentation, list):
        for segment in segmentation:
            points = flat_polygon_to_points(segment)
            if points:
                candidates.append(points)
    if candidates:
        return max(candidates, key=polygon_area)
    return bbox_to_points(annotation.get("bbox", []))


def annotation_to_text(annotation):
    if annotation.get("iscrowd"):
        return "###"
    for key in ("transcription", "text", "utf8_string", "label"):
        value = annotation.get(key)
        if value is not None and str(value).strip():
            return str(value).replace("\t", " ").replace("\n", " ").strip()
    return "text"


def is_ignored_annotation(annotation):
    ignore_keys = (
        "ignore",
        "ignored",
        "illegible",
        "illegibility",
        "do_not_care",
    )
    return any(bool(annotation.get(key)) for key in ignore_keys)


def find_image_path(image_dir, file_name):
    file_path = Path(file_name)
    candidates = []
    if file_path.is_absolute():
        candidates.append(file_path)
    candidates.append(image_dir / file_name)
    candidates.append(image_dir / file_path.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    stem = file_path.stem
    if stem:
        for suffix in IMAGE_SUFFIXES:
            candidate = image_dir / f"{stem}{suffix}"
            if candidate.is_file():
                return candidate.resolve()
    return (image_dir / file_name).resolve()


def load_coco_entries(dataset_cfg):
    annotation_path = resolve_path(dataset_cfg["annotation"])
    image_dir = resolve_path(dataset_cfg["image_dir"])
    if not annotation_path.is_file():
        raise FileNotFoundError(f"COCO annotation not found: {annotation_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    with annotation_path.open("r", encoding="utf-8-sig") as file:
        coco = json.load(file)

    images = {image["id"]: image for image in coco.get("images", [])}
    annotations_by_image = defaultdict(list)
    stats = {
        "images": len(images),
        "annotations": len(coco.get("annotations", [])),
        "missing_image_id": 0,
        "bad_geometry": 0,
        "missing_image_file": 0,
        "kept_images": 0,
        "kept_boxes": 0,
    }

    for annotation in coco.get("annotations", []):
        image_id = annotation.get("image_id")
        if image_id not in images:
            stats["missing_image_id"] += 1
            continue

        points = annotation_to_points(annotation)
        if not points:
            stats["bad_geometry"] += 1
            continue

        text = "###" if is_ignored_annotation(annotation) else annotation_to_text(annotation)
        annotations_by_image[image_id].append({"transcription": text, "points": points})

    entries = []
    for image_id, labels in annotations_by_image.items():
        image = images[image_id]
        file_name = image.get("file_name")
        if not file_name:
            continue
        image_path = find_image_path(image_dir, file_name)
        if not image_path.is_file():
            stats["missing_image_file"] += 1
            continue
        entries.append((image_path, labels))
        stats["kept_images"] += 1
        stats["kept_boxes"] += len(labels)

    return entries, stats


def split_train_eval(entries):
    shuffled = list(entries)
    random.Random(SEED).shuffle(shuffled)
    split_index = max(1, int(round(len(shuffled) * TRAIN_RATIO_IF_NO_EVAL)))
    split_index = min(split_index, len(shuffled) - 1)
    return shuffled[:split_index], shuffled[split_index:]


def write_label_file(path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for image_path, labels in entries:
            label_json = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
            file.write(f"{image_path.as_posix()}\t{label_json}\n")


def get_config_path():
    if MODEL_SIZE == "mobile":
        return REPO_ROOT / "configs/det/PP-OCRv5/PP-OCRv5_mobile_det.yml"
    if MODEL_SIZE == "server":
        return REPO_ROOT / "configs/det/PP-OCRv5/PP-OCRv5_server_det.yml"
    raise ValueError('MODEL_SIZE must be "mobile" or "server"')


def write_prepared_config(base_config_path, output_config_path):
    with base_config_path.open("rb") as file:
        config = yaml.load(file, Loader=yaml.Loader)

    if TRAIN_IMAGE_SIZE is not None:
        size = int(TRAIN_IMAGE_SIZE)
        config["Global"]["d2s_train_image_shape"] = [3, size, size]
        for transform in config["Train"]["dataset"]["transforms"]:
            if "EastRandomCropData" in transform:
                transform["EastRandomCropData"]["size"] = [size, size]
                break
        else:
            raise RuntimeError("EastRandomCropData was not found in Train transforms.")

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with output_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, allow_unicode=True, sort_keys=False)


def build_training_command(config_path, train_list_path, eval_list_path):
    if GPUS.strip():
        command = [
            sys.executable,
            "-m",
            "paddle.distributed.launch",
            f"--gpus={GPUS}",
            "tools/train.py",
        ]
    else:
        command = [sys.executable, "tools/train.py"]

    overrides = [
        "Train.dataset.data_dir=.",
        f"Train.dataset.label_file_list=[{to_override_path(train_list_path)}]",
        "Train.dataset.ratio_list=[1.0]",
        "Eval.dataset.data_dir=.",
        f"Eval.dataset.label_file_list=[{to_override_path(eval_list_path)}]",
        f"Global.save_model_dir={to_override_path(resolve_path(SAVE_MODEL_DIR))}",
    ]

    optional_overrides = {
        "Global.epoch_num": EPOCH_NUM,
        "Train.loader.batch_size_per_card": BATCH_SIZE_PER_CARD,
        "Train.loader.num_workers": TRAIN_NUM_WORKERS,
        "Eval.loader.num_workers": EVAL_NUM_WORKERS,
        "Optimizer.lr.learning_rate": LEARNING_RATE,
        "Global.eval_batch_step": [0, EVAL_BATCH_STEP] if EVAL_BATCH_STEP else None,
        "Global.pretrained_model": PRETRAINED_MODEL,
        "Global.checkpoints": CHECKPOINTS,
    }
    for key, value in optional_overrides.items():
        if value is not None:
            overrides.append(f"{key}={value}")

    return command + ["-c", to_override_path(config_path), "-o"] + overrides


def print_dataset_table(rows):
    if not rows:
        return
    columns = ["dataset", "split", "images", "annotations", "kept_images", "kept_boxes"]
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in rows))
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))


def main():
    random.seed(SEED)
    prepared_data_dir = resolve_path(PREPARED_DATA_DIR)
    base_config_path = get_config_path()
    config_path = prepared_data_dir / f"PP-OCRv5_{MODEL_SIZE}_det_custom.yml"
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Config not found: {base_config_path}")
    write_prepared_config(base_config_path, config_path)

    train_list_path = prepared_data_dir / "train_det_label.txt"
    eval_list_path = prepared_data_dir / "eval_det_label.txt"
    summary_path = prepared_data_dir / "dataset_summary.json"

    train_entries = []
    eval_entries = []
    summary_rows = []
    summary_json = {"datasets": []}

    for dataset_cfg in DATASETS:
        split = dataset_cfg.get("split", "train").lower()
        if split not in {"train", "eval", "val", "test"}:
            raise ValueError(f'Bad split for {dataset_cfg["name"]}: {split}')

        entries, stats = load_coco_entries(dataset_cfg)
        if split == "train":
            train_entries.extend(entries)
        else:
            eval_entries.extend(entries)

        row = {
            "dataset": dataset_cfg["name"],
            "split": split,
            **stats,
        }
        summary_rows.append(row)
        summary_json["datasets"].append(
            {
                **row,
                "annotation": resolve_path(dataset_cfg["annotation"]).as_posix(),
                "image_dir": resolve_path(dataset_cfg["image_dir"]).as_posix(),
            }
        )

    if not train_entries:
        raise RuntimeError('No train samples. Add at least one dataset with split="train".')

    if not eval_entries:
        if len(train_entries) < 2:
            raise RuntimeError(
                "No eval samples and not enough train samples to auto-split."
            )
        train_entries, eval_entries = split_train_eval(train_entries)
        summary_json["auto_split_eval_from_train"] = True
        summary_json["train_ratio_if_no_eval"] = TRAIN_RATIO_IF_NO_EVAL

    write_label_file(train_list_path, train_entries)
    write_label_file(eval_list_path, eval_entries)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    command = build_training_command(config_path, train_list_path, eval_list_path)

    print(f"Base config: {base_config_path}")
    print(f"Prepared config: {config_path}")
    print(f"Train label: {train_list_path}")
    print(f"Eval label: {eval_list_path}")
    print(f"Summary: {summary_path}")
    print()
    print("Datasets:")
    print_dataset_table(summary_rows)
    print()
    print(f"Total train images: {len(train_entries)}")
    print(f"Total eval images: {len(eval_entries)}")
    print()
    print("Training command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    if RUN_TRAIN:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
