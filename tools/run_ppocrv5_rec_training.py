import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CONFIG = "configs/rec/PP-OCRv5/multi_language/cyrillic_PP-OCRv5_mobile_rec.yaml"
DEFAULT_PRETRAINED_MODEL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
    "cyrillic_PP-OCRv5_mobile_rec_pretrained.pdparams"
)
DEFAULT_DICT = "ppocr/utils/dict/ppocrv5_cyrillic_dict.txt"
DEFAULT_ANNOTATION_NAMES = [
    "orig_cyrillic_gt.csv",
    "gt.csv",
    "labels.csv",
    "label.csv",
    "train.csv",
    "rec_gt.csv",
    "test.csv",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a PP-OCRv5 recognition dataset from one or more CSV-based folders "
            "and launch training."
        )
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        help=(
            "Dataset root directory. Can be repeated. If omitted and "
            "./CyrillicHandwritingDataset exists, that folder is used."
        ),
    )
    parser.add_argument(
        "--annotation-file",
        action="append",
        default=[],
        help=(
            "Explicit annotation CSV file. Can be repeated. If omitted, the script scans "
            "dataset roots and picks one ground-truth CSV per folder."
        ),
    )
    parser.add_argument(
        "--annotation-name",
        action="append",
        default=[],
        help=(
            "Preferred annotation filename when scanning dataset roots. Can be repeated. "
            "If omitted, built-in defaults are used."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Recognition config to train with.",
    )
    parser.add_argument(
        "--pretrained-model",
        default=DEFAULT_PRETRAINED_MODEL,
        help="Pretrained model path or URL.",
    )
    parser.add_argument(
        "--dict-path",
        default="",
        help=(
            "Existing PaddleOCR dictionary file to use directly. One character per line. "
            "If set, it has priority over --alphabet-file and --auto-dict."
        ),
    )
    parser.add_argument(
        "--alphabet-file",
        default="",
        help=(
            "Custom alphabet file. Either one character per line or a plain text file "
            "containing all characters."
        ),
    )
    parser.add_argument(
        "--auto-dict",
        action="store_true",
        help="Build the dictionary automatically from all labels in the dataset.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train split ratio. The rest is used for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val split.",
    )
    parser.add_argument(
        "--prepared-data-dir",
        default="train_data/custom_ppocrv5_rec",
        help="Directory where train_list.txt, val_list.txt and generated dict are written.",
    )
    parser.add_argument(
        "--save-model-dir",
        default="output/custom_ppocrv5_rec",
        help="Output directory for PaddleOCR checkpoints.",
    )
    parser.add_argument(
        "--epoch-num",
        type=int,
        default=None,
        help="Optional override for Global.epoch_num.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for Train/Eval batch_size_per_card.",
    )
    parser.add_argument(
        "--train-num-workers",
        type=int,
        default=None,
        help="Optional override for Train.loader.num_workers.",
    )
    parser.add_argument(
        "--eval-num-workers",
        type=int,
        default=None,
        help="Optional override for Eval.loader.num_workers.",
    )
    parser.add_argument(
        "--max-text-length",
        type=int,
        default=None,
        help=(
            "Optional override for Global.max_text_length. If omitted, the script uses "
            "max(25, max label length in the dataset)."
        ),
    )
    parser.add_argument(
        "--gpus",
        default="",
        help="GPU ids, for example 0 or 0,1. Empty means the standard single-process launch.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Only prepare files and print the training command.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Alias of --skip-train.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostics while scanning the dataset.",
    )
    space_group = parser.add_mutually_exclusive_group()
    space_group.add_argument(
        "--use-space-char",
        dest="use_space_char",
        action="store_true",
        default=None,
        help="Force Global.use_space_char=True.",
    )
    space_group.add_argument(
        "--no-space-char",
        dest="use_space_char",
        action="store_false",
        help="Force Global.use_space_char=False.",
    )
    return parser.parse_args()


def resolve_existing_path(path_str, repo_root):
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    current_candidate = (Path.cwd() / path).resolve()
    if current_candidate.exists():
        return current_candidate
    repo_candidate = (repo_root / path).resolve()
    if repo_candidate.exists():
        return repo_candidate
    return current_candidate


def resolve_output_path(path_str, repo_root):
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def to_override_path(path_obj, repo_root):
    try:
        return path_obj.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path_obj.resolve().as_posix()


def find_annotation_files(dataset_roots, explicit_files, annotation_names, verbose=False):
    if explicit_files:
        files = []
        for item in explicit_files:
            path = item.resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Annotation file not found: {path}")
            files.append(path)
        return sorted(dict.fromkeys(files))

    selected = []
    seen = set()
    names_lower = [name.lower() for name in annotation_names]
    for root in dataset_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root}")

        files_by_parent = defaultdict(list)
        for csv_path in root.rglob("*.csv"):
            files_by_parent[csv_path.parent.resolve()].append(csv_path.resolve())

        for parent_dir in sorted(files_by_parent):
            name_map = {path.name.lower(): path for path in files_by_parent[parent_dir]}
            chosen = None
            for name in names_lower:
                if name in name_map:
                    chosen = name_map[name]
                    break
            if chosen is None:
                continue
            if chosen in seen:
                continue
            selected.append(chosen)
            seen.add(chosen)
            if verbose:
                print(f"[scan] using annotation: {chosen}")

    if not selected:
        searched = ", ".join(annotation_names)
        roots = ", ".join(str(root) for root in dataset_roots)
        raise FileNotFoundError(
            f"No annotation files found. Searched for [{searched}] under: {roots}"
        )
    return selected


def detect_header(row):
    if len(row) < 2:
        return False
    left = row[0].strip().lower()
    right = row[1].strip().lower()
    left_ok = left in {"image", "img", "filename", "file_name", "path", "image_path"}
    right_ok = right in {"prediction", "label", "text", "gt", "transcription"}
    return left_ok and right_ok


def build_image_index(search_root):
    index = defaultdict(list)
    for path in search_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.name].append(path.resolve())
    return index


def resolve_image_path(annotation_dir, image_ref, image_index):
    ref_path = Path(image_ref)
    if ref_path.is_absolute() and ref_path.is_file():
        return ref_path.resolve()

    direct_candidate = (annotation_dir / ref_path).resolve()
    if direct_candidate.is_file():
        return direct_candidate

    matches = image_index.get(ref_path.name, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        tail = ref_path.as_posix()
        tail_matches = [item for item in matches if item.as_posix().endswith(tail)]
        if len(tail_matches) == 1:
            return tail_matches[0]
        raise RuntimeError(
            f"Ambiguous image reference '{image_ref}' under {annotation_dir}. "
            f"Found {len(matches)} matching files."
        )

    raise FileNotFoundError(
        f"Image '{image_ref}' referenced by {annotation_dir} was not found."
    )


def sanitize_label(label):
    return label.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def read_entries_from_annotation(csv_path, verbose=False):
    annotation_dir = csv_path.parent.resolve()
    image_index = build_image_index(annotation_dir)
    entries = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row is None:
            return entries
        rows_iter = reader if detect_header(first_row) else [first_row] + list(reader)

    for row_idx, row in enumerate(rows_iter, start=1):
        if len(row) < 2:
            if verbose:
                print(f"[warn] skip short row {row_idx} in {csv_path}")
            continue
        image_ref = row[0].strip()
        label = sanitize_label(row[1])
        if not image_ref or not label:
            if verbose:
                print(f"[warn] skip empty row {row_idx} in {csv_path}")
            continue
        image_path = resolve_image_path(annotation_dir, image_ref, image_index)
        entries.append((image_path, label))
    return entries


def deduplicate_entries(entries):
    labels_by_image = {}
    for image_path, label in entries:
        key = image_path.resolve().as_posix()
        if key in labels_by_image:
            if labels_by_image[key] != label:
                raise RuntimeError(
                    f"Conflicting labels for the same image: {image_path}\n"
                    f"First: {labels_by_image[key]}\n"
                    f"Second: {label}"
                )
            continue
        labels_by_image[key] = label
    return [(Path(path_str), label) for path_str, label in sorted(labels_by_image.items())]


def split_entries(entries, train_ratio, seed):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1.")
    if len(entries) < 2:
        raise RuntimeError("Need at least 2 labeled samples to build train and validation splits.")

    shuffled = list(entries)
    random.Random(seed).shuffle(shuffled)

    train_count = int(round(len(shuffled) * train_ratio))
    train_count = max(1, min(len(shuffled) - 1, train_count))
    train_entries = shuffled[:train_count]
    val_entries = shuffled[train_count:]
    return train_entries, val_entries


def write_label_file(file_path, entries):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        for image_path, label in entries:
            handle.write(f"{image_path.as_posix()}\t{label}\n")


def read_dict_chars(dict_path):
    chars = []
    with dict_path.open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line == "":
                continue
            chars.append(line)
    return chars


def unique_preserve_order(items):
    seen = set()
    result = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def build_dict_from_alphabet_file(alphabet_path, use_space_char):
    text = alphabet_path.read_text(encoding="utf-8-sig")
    lines = text.splitlines()
    non_empty_lines = [line for line in lines if line != ""]

    if non_empty_lines and all(len(line) == 1 for line in non_empty_lines):
        chars = non_empty_lines
    else:
        chars = [char for char in text if char not in {"\r", "\n"}]

    chars = unique_preserve_order(chars)
    if use_space_char:
        chars = [char for char in chars if char != " "]
    return chars


def build_dict_from_labels(labels, use_space_char):
    chars = []
    seen = set()
    for label in labels:
        for char in label:
            if use_space_char and char == " ":
                continue
            if char in seen:
                continue
            seen.add(char)
            chars.append(char)
    return chars


def write_dict_file(dict_path, chars):
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    with dict_path.open("w", encoding="utf-8", newline="") as handle:
        for char in chars:
            handle.write(char + "\n")


def collect_missing_chars(labels, dict_chars, use_space_char):
    available = set(dict_chars)
    missing = set()
    for label in labels:
        for char in label:
            if use_space_char and char == " ":
                continue
            if char not in available:
                missing.add(char)
    return sorted(missing)


def build_training_command(args, repo_root, config_path, train_list_path, val_list_path, dict_path, use_space_char, max_text_length):
    command = [sys.executable]
    if args.gpus:
        command.extend(
            [
                "-m",
                "paddle.distributed.launch",
                "--gpus",
                args.gpus,
                "tools/train.py",
            ]
        )
    else:
        command.append("tools/train.py")

    command.extend(["-c", to_override_path(config_path, repo_root), "-o"])

    overrides = [
        f"Global.pretrained_model={args.pretrained_model}",
        f"Global.character_dict_path={to_override_path(dict_path, repo_root)}",
        f"Global.save_model_dir={to_override_path(resolve_output_path(args.save_model_dir, repo_root), repo_root)}",
        f"Global.max_text_length={max_text_length}",
        f"Global.use_space_char={str(use_space_char)}",
        "Train.dataset.data_dir=.",
        f"Train.dataset.label_file_list=[{to_override_path(train_list_path, repo_root)}]",
        "Eval.dataset.data_dir=.",
        f"Eval.dataset.label_file_list=[{to_override_path(val_list_path, repo_root)}]",
    ]

    if args.epoch_num is not None:
        overrides.append(f"Global.epoch_num={args.epoch_num}")
    if args.batch_size is not None:
        overrides.append(f"Train.loader.batch_size_per_card={args.batch_size}")
        overrides.append(f"Eval.loader.batch_size_per_card={args.batch_size}")
    if args.train_num_workers is not None:
        overrides.append(f"Train.loader.num_workers={args.train_num_workers}")
    if args.eval_num_workers is not None:
        overrides.append(f"Eval.loader.num_workers={args.eval_num_workers}")

    command.extend(overrides)
    return command


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    if not args.dataset_root and not args.annotation_file:
        default_root = repo_root / "CyrillicHandwritingDataset"
        if default_root.is_dir():
            args.dataset_root = [str(default_root)]
        else:
            raise RuntimeError("No dataset specified. Use --dataset-root or --annotation-file.")

    annotation_names = args.annotation_name or list(DEFAULT_ANNOTATION_NAMES)
    dataset_roots = [resolve_existing_path(item, repo_root) for item in args.dataset_root]
    explicit_annotation_files = [
        resolve_existing_path(item, repo_root) for item in args.annotation_file
    ]

    config_path = resolve_existing_path(args.config, repo_root)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    prepared_data_dir = resolve_output_path(args.prepared_data_dir, repo_root)
    train_list_path = prepared_data_dir / "train_list.txt"
    val_list_path = prepared_data_dir / "val_list.txt"
    summary_path = prepared_data_dir / "dataset_summary.json"

    annotation_files = find_annotation_files(
        dataset_roots=dataset_roots,
        explicit_files=explicit_annotation_files,
        annotation_names=annotation_names,
        verbose=args.verbose,
    )

    all_entries = []
    for csv_path in annotation_files:
        rows = read_entries_from_annotation(csv_path, verbose=args.verbose)
        if args.verbose:
            print(f"[scan] loaded {len(rows)} rows from {csv_path}")
        all_entries.extend(rows)

    entries = deduplicate_entries(all_entries)
    if not entries:
        raise RuntimeError("No usable samples were found in the provided annotations.")

    labels = [label for _, label in entries]
    contains_spaces = any(" " in label for label in labels)
    use_space_char = args.use_space_char if args.use_space_char is not None else contains_spaces

    if args.max_text_length is None:
        max_text_length = max(25, max(len(label) for label in labels))
    else:
        observed_max = max(len(label) for label in labels)
        if args.max_text_length < observed_max:
            raise RuntimeError(
                f"--max-text-length={args.max_text_length} is too small. "
                f"Observed label length is {observed_max}."
            )
        max_text_length = args.max_text_length

    dict_path = None
    generated_dict = False
    if args.dict_path:
        dict_path = resolve_existing_path(args.dict_path, repo_root)
        if not dict_path.is_file():
            raise FileNotFoundError(f"Dictionary file not found: {dict_path}")
    elif args.alphabet_file:
        alphabet_path = resolve_existing_path(args.alphabet_file, repo_root)
        if not alphabet_path.is_file():
            raise FileNotFoundError(f"Alphabet file not found: {alphabet_path}")
        dict_chars = build_dict_from_alphabet_file(alphabet_path, use_space_char)
        if not dict_chars:
            raise RuntimeError("The alphabet file produced an empty dictionary.")
        dict_path = prepared_data_dir / "custom_dict.txt"
        write_dict_file(dict_path, dict_chars)
        generated_dict = True
    elif args.auto_dict:
        dict_chars = build_dict_from_labels(labels, use_space_char)
        if not dict_chars:
            raise RuntimeError("Automatic dictionary generation produced an empty dictionary.")
        dict_path = prepared_data_dir / "custom_dict.txt"
        write_dict_file(dict_path, dict_chars)
        generated_dict = True
    else:
        dict_path = resolve_existing_path(DEFAULT_DICT, repo_root)
        if not dict_path.is_file():
            raise FileNotFoundError(f"Default dictionary file not found: {dict_path}")

    dict_chars = read_dict_chars(dict_path)
    missing_chars = collect_missing_chars(labels, dict_chars, use_space_char)
    if missing_chars:
        missing_repr = ", ".join(repr(char) for char in missing_chars[:20])
        raise RuntimeError(
            "The selected dictionary does not cover all label characters. "
            f"Missing {len(missing_chars)} chars: {missing_repr}\n"
            "Use --auto-dict or provide --alphabet-file / --dict-path."
        )

    train_entries, val_entries = split_entries(entries, args.train_ratio, args.seed)
    write_label_file(train_list_path, train_entries)
    write_label_file(val_list_path, val_entries)

    summary = {
        "annotation_files": [path.as_posix() for path in annotation_files],
        "total_samples": len(entries),
        "train_samples": len(train_entries),
        "val_samples": len(val_entries),
        "use_space_char": use_space_char,
        "max_text_length": max_text_length,
        "dictionary_path": dict_path.as_posix(),
        "dictionary_generated": generated_dict,
        "dictionary_size": len(dict_chars),
        "train_list_path": train_list_path.as_posix(),
        "val_list_path": val_list_path.as_posix(),
        "config": config_path.as_posix(),
        "pretrained_model": args.pretrained_model,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    command = build_training_command(
        args=args,
        repo_root=repo_root,
        config_path=config_path,
        train_list_path=train_list_path,
        val_list_path=val_list_path,
        dict_path=dict_path,
        use_space_char=use_space_char,
        max_text_length=max_text_length,
    )

    print(f"Prepared {len(entries)} samples from {len(annotation_files)} annotation file(s).")
    print(f"Train split: {len(train_entries)}")
    print(f"Val split: {len(val_entries)}")
    print(f"Dictionary: {dict_path}")
    print(f"Summary: {summary_path}")
    print("Training command:")
    print(" ".join(f'"{part}"' if " " in part else part for part in command))

    if args.skip_train or args.dry_run:
        return

    subprocess.run(command, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
