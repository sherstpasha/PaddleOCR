import csv
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


# =========================
# Edit only this section
# =========================

DATASETS = [
    {
        "name": "CyrillicHandwritingDataset",
        "root": "CyrillicHandwritingDataset",
        "annotation": "orig_cyrillic_gt.csv",
    },
    # Example for adding more datasets:
    # {
    #     "name": "MySecondDataset",
    #     "root": "MySecondDataset",
    #     "annotation": "labels.csv",
    # },
]

CONFIG_PATH = "configs/rec/PP-OCRv5/multi_language/cyrillic_PP-OCRv5_mobile_rec.yaml"
PRETRAINED_MODEL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
    "cyrillic_PP-OCRv5_mobile_rec_pretrained.pdparams"
)

PREPARED_DATA_DIR = "train_data/custom_ppocrv5_rec"
SAVE_MODEL_DIR = "output/custom_ppocrv5_rec"

MAX_TEXT_LENGTH = 40
TRAIN_RATIO = 0.9
SEED = 42

# Leave empty for normal single-process training.
# Set "0" or "0,1" to run through paddle.distributed.launch.
GPUS = ""

EPOCH_NUM = None
BATCH_SIZE = None
TRAIN_NUM_WORKERS = None
EVAL_NUM_WORKERS = None

# If True, the script prepares data and immediately starts training.
# If False, it only prepares files and prints the command.
RUN_TRAIN = True

# Strict alphabet source:
# 1. If ALPHABET_TEXT is not empty, it is used.
# 2. Otherwise ALPHABET_FILE is used.
# Supported formats:
# - one character per line
# - or one plain text line containing all characters
ALPHABET_TEXT = ""
ALPHABET_FILE = "train_data/custom_ppocrv5_rec/alphabet.txt"


# =========================
# Internal code
# =========================

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ANNOTATION_CANDIDATES = [
    "orig_cyrillic_gt.csv",
    "gt.csv",
    "labels.csv",
    "label.csv",
    "train.csv",
    "rec_gt.csv",
    "test.csv",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def detect_header(row):
    if len(row) < 2:
        return False
    left = row[0].strip().lower()
    right = row[1].strip().lower()
    return left in {
        "image",
        "img",
        "filename",
        "file_name",
        "path",
        "image_path",
    } and right in {"prediction", "label", "text", "gt", "transcription"}


def sanitize_label(label):
    return label.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def unique_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def load_alphabet():
    if ALPHABET_TEXT.strip():
        raw_text = ALPHABET_TEXT
    else:
        alphabet_path = resolve_path(ALPHABET_FILE)
        if not alphabet_path.is_file():
            raise FileNotFoundError(
                "Alphabet is not configured.\n"
                f"Either fill ALPHABET_TEXT in this script or create: {alphabet_path}"
            )
        raw_text = alphabet_path.read_text(encoding="utf-8-sig")

    lines = raw_text.splitlines()
    non_empty_lines = [line for line in lines if line != ""]

    if non_empty_lines and all(len(line) == 1 for line in non_empty_lines):
        alphabet_chars = non_empty_lines
    else:
        alphabet_chars = [char for char in raw_text if char not in {"\r", "\n"}]

    if not alphabet_chars:
        raise RuntimeError("Alphabet is empty.")

    duplicates = []
    seen = set()
    for char in alphabet_chars:
        if char in seen and char not in duplicates:
            duplicates.append(char)
        seen.add(char)
    if duplicates:
        raise RuntimeError(
            "Alphabet contains duplicate characters: "
            + ", ".join(repr(char) for char in duplicates[:20])
        )

    use_space_char = " " in alphabet_chars
    dict_chars = [char for char in alphabet_chars if char != " "]
    allowed_chars = set(dict_chars)
    if use_space_char:
        allowed_chars.add(" ")

    return alphabet_chars, dict_chars, allowed_chars, use_space_char


def write_dict_file(dict_path, dict_chars):
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    with dict_path.open("w", encoding="utf-8", newline="") as handle:
        for char in dict_chars:
            handle.write(char + "\n")


def find_annotation_file(dataset_root, explicit_name):
    if explicit_name:
        annotation_path = (dataset_root / explicit_name).resolve()
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        return annotation_path

    matches = []
    for name in ANNOTATION_CANDIDATES:
        direct_path = dataset_root / name
        if direct_path.is_file():
            return direct_path.resolve()
        matches.extend(dataset_root.rglob(name))

    unique_matches = sorted({match.resolve() for match in matches})
    if not unique_matches:
        raise FileNotFoundError(
            f"No annotation CSV found under {dataset_root}. "
            f"Tried: {', '.join(ANNOTATION_CANDIDATES)}"
        )
    if len(unique_matches) > 1:
        raise RuntimeError(
            f"More than one annotation CSV found under {dataset_root}: "
            + ", ".join(str(path) for path in unique_matches)
        )
    return unique_matches[0]


def build_image_index(search_root):
    index = defaultdict(list)
    for path in search_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.name].append(path.resolve())
    return index


def resolve_image_path(dataset_root, annotation_dir, image_ref, image_index):
    ref_path = Path(image_ref)
    if ref_path.is_absolute() and ref_path.is_file():
        return ref_path.resolve()

    direct_from_annotation = (annotation_dir / ref_path).resolve()
    if direct_from_annotation.is_file():
        return direct_from_annotation

    direct_from_root = (dataset_root / ref_path).resolve()
    if direct_from_root.is_file():
        return direct_from_root

    matches = image_index.get(ref_path.name, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous image reference '{image_ref}' under {dataset_root}. "
            f"Found {len(matches)} files with the same name."
        )

    raise FileNotFoundError(
        f"Image '{image_ref}' referenced by {annotation_dir} was not found."
    )


def read_dataset_entries(dataset_name, dataset_root, annotation_path):
    annotation_dir = annotation_path.parent.resolve()
    image_index = build_image_index(dataset_root)
    entries = []

    with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row is None:
            return entries
        rows = reader if detect_header(first_row) else [first_row] + list(reader)

    for row in rows:
        if len(row) < 2:
            continue
        image_ref = row[0].strip()
        label = sanitize_label(row[1])
        if not image_ref or not label:
            continue
        image_path = resolve_image_path(dataset_root, annotation_dir, image_ref, image_index)
        entries.append(
            {
                "dataset": dataset_name,
                "image_path": image_path,
                "label": label,
            }
        )

    return entries


def filter_entries(entries, allowed_chars):
    stats = Counter()
    invalid_chars = Counter()
    filtered = []

    for entry in entries:
        label = entry["label"]
        stats["raw_total"] += 1

        if len(label) > MAX_TEXT_LENGTH:
            stats["skipped_too_long"] += 1
            continue

        bad_chars = sorted({char for char in label if char not in allowed_chars})
        if bad_chars:
            stats["skipped_bad_chars"] += 1
            invalid_chars.update(bad_chars)
            continue

        filtered.append(entry)
        stats["kept"] += 1

    return filtered, stats, invalid_chars


def split_entries(entries, seed_offset):
    if not entries:
        return [], []

    shuffled = list(entries)
    random.Random(SEED + seed_offset).shuffle(shuffled)

    if len(shuffled) == 1:
        return shuffled, []

    train_count = int(round(len(shuffled) * TRAIN_RATIO))
    train_count = max(1, min(len(shuffled) - 1, train_count))
    return shuffled[:train_count], shuffled[train_count:]


def write_label_file(file_path, entries):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        for entry in entries:
            handle.write(f"{entry['image_path'].as_posix()}\t{entry['label']}\n")


def build_training_command(config_path, train_list_path, val_list_path, dict_path, use_space_char):
    command = [sys.executable]
    if GPUS:
        command.extend(
            [
                "-m",
                "paddle.distributed.launch",
                "--gpus",
                GPUS,
                "tools/train.py",
            ]
        )
    else:
        command.append("tools/train.py")

    command.extend(["-c", to_override_path(config_path), "-o"])
    command.extend(
        [
            f"Global.pretrained_model={PRETRAINED_MODEL}",
            f"Global.character_dict_path={to_override_path(dict_path)}",
            f"Global.save_model_dir={to_override_path(resolve_path(SAVE_MODEL_DIR))}",
            f"Global.max_text_length={MAX_TEXT_LENGTH}",
            f"Global.use_space_char={str(use_space_char)}",
            "Train.dataset.data_dir=.",
            f"Train.dataset.label_file_list=[{to_override_path(train_list_path)}]",
            "Eval.dataset.data_dir=.",
            f"Eval.dataset.label_file_list=[{to_override_path(val_list_path)}]",
        ]
    )

    if EPOCH_NUM is not None:
        command.append(f"Global.epoch_num={EPOCH_NUM}")
    if BATCH_SIZE is not None:
        command.append(f"Train.loader.batch_size_per_card={BATCH_SIZE}")
        command.append(f"Eval.loader.batch_size_per_card={BATCH_SIZE}")
    if TRAIN_NUM_WORKERS is not None:
        command.append(f"Train.loader.num_workers={TRAIN_NUM_WORKERS}")
    if EVAL_NUM_WORKERS is not None:
        command.append(f"Eval.loader.num_workers={EVAL_NUM_WORKERS}")

    return command


def print_dataset_table(rows):
    headers = [
        "dataset",
        "raw",
        "kept",
        "skip_len",
        "skip_chars",
        "train",
        "test",
    ]

    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    header_line = " | ".join(header.ljust(widths[header]) for header in headers)
    separator = "-+-".join("-" * widths[header] for header in headers)
    print(header_line)
    print(separator)
    for row in rows:
        print(
            " | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers)
        )


def main():
    config_path = resolve_path(CONFIG_PATH)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    prepared_data_dir = resolve_path(PREPARED_DATA_DIR)
    train_list_path = prepared_data_dir / "train_list.txt"
    val_list_path = prepared_data_dir / "val_list.txt"
    dict_path = prepared_data_dir / "custom_dict.txt"
    summary_path = prepared_data_dir / "dataset_summary.json"

    _, dict_chars, allowed_chars, use_space_char = load_alphabet()
    write_dict_file(dict_path, dict_chars)

    all_train_entries = []
    all_val_entries = []
    summary_rows = []
    summary_json = {
        "config": config_path.as_posix(),
        "pretrained_model": PRETRAINED_MODEL,
        "prepared_data_dir": prepared_data_dir.as_posix(),
        "save_model_dir": resolve_path(SAVE_MODEL_DIR).as_posix(),
        "max_text_length": MAX_TEXT_LENGTH,
        "train_ratio": TRAIN_RATIO,
        "use_space_char": use_space_char,
        "dictionary_path": dict_path.as_posix(),
        "dictionary_size": len(dict_chars),
        "datasets": [],
    }

    for dataset_index, dataset_cfg in enumerate(DATASETS):
        dataset_name = dataset_cfg["name"]
        dataset_root = resolve_path(dataset_cfg["root"])
        annotation_path = find_annotation_file(
            dataset_root=dataset_root,
            explicit_name=dataset_cfg.get("annotation", ""),
        )

        raw_entries = read_dataset_entries(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            annotation_path=annotation_path,
        )
        filtered_entries, filter_stats, invalid_chars = filter_entries(raw_entries, allowed_chars)
        train_entries, val_entries = split_entries(filtered_entries, dataset_index)

        all_train_entries.extend(train_entries)
        all_val_entries.extend(val_entries)

        invalid_preview = "".join(char for char, _ in invalid_chars.most_common(20))
        summary_rows.append(
            {
                "dataset": dataset_name,
                "raw": filter_stats["raw_total"],
                "kept": filter_stats["kept"],
                "skip_len": filter_stats["skipped_too_long"],
                "skip_chars": filter_stats["skipped_bad_chars"],
                "train": len(train_entries),
                "test": len(val_entries),
            }
        )
        summary_json["datasets"].append(
            {
                "name": dataset_name,
                "root": dataset_root.as_posix(),
                "annotation": annotation_path.as_posix(),
                "raw_total": filter_stats["raw_total"],
                "kept": filter_stats["kept"],
                "skipped_too_long": filter_stats["skipped_too_long"],
                "skipped_bad_chars": filter_stats["skipped_bad_chars"],
                "train_samples": len(train_entries),
                "test_samples": len(val_entries),
                "top_invalid_chars": invalid_preview,
            }
        )

    if not all_train_entries:
        raise RuntimeError("No training samples remain after filtering.")
    if not all_val_entries:
        raise RuntimeError(
            "No test/validation samples remain after filtering. "
            "Lower TRAIN_RATIO or add more data."
        )

    write_label_file(train_list_path, all_train_entries)
    write_label_file(val_list_path, all_val_entries)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    command = build_training_command(
        config_path=config_path,
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        dict_path=dict_path,
        use_space_char=use_space_char,
    )

    print(f"Dictionary file: {dict_path}")
    print(f"Train list: {train_list_path}")
    print(f"Test list: {val_list_path}")
    print(f"Summary: {summary_path}")
    print()
    print("Samples per dataset:")
    print_dataset_table(summary_rows)
    print()
    print(f"Total train samples: {len(all_train_entries)}")
    print(f"Total test samples: {len(all_val_entries)}")
    print()
    print("Training command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    if not RUN_TRAIN:
        return

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
