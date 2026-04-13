import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


DATASET_NAMES = [
    "school_notebooks_RU",
    "CyrillicHandwritingDataset",
    "YeniseiGovReports-HWR",
    "YeniseiGovReports-PRT",
    "RussianSchoolEssays",
    "DonkeySmallOCR-Numbers-Printed-15random",
    "letters_9_may_preview",
    "TrOCR",
    "HandwrittenKazakhRussian",
]

TRAIN_CSVS = [
    r"C:/shared/data ocr/school_notebooks_RU/train.csv",
    r"C:/shared/data ocr/CyrillicHandwritingDataset/train.csv",
    r"C:/shared/data ocr/YeniseiGovReports-HWR/train/labels.csv",
    r"C:/shared/data ocr/YeniseiGovReports-PRT/train/labels.csv",
    r"C:\shared\data ocr\RussianSchoolEssays\train\train.csv",
    r"C:\shared\data ocr\DonkeySmallOCR-Numbers-Printed-15random\train\labels.csv",
    r"C:\shared\data ocr\letters_9_may_preview\train.csv",
    r"C:\shared\data ocr\TrOCR\train.csv",
    r"C:\shared\data ocr\HandwrittenKazakhRussian\train\gt.txt",
]

TRAIN_ROOTS = [
    r"C:/shared/data ocr/school_notebooks_RU/train",
    r"C:/shared/data ocr/CyrillicHandwritingDataset",
    r"C:/shared/data ocr/YeniseiGovReports-HWR/train",
    r"C:/shared/data ocr/YeniseiGovReports-PRT/train",
    r"C:\shared\data ocr\RussianSchoolEssays\train\train_images",
    r"C:\shared\data ocr\DonkeySmallOCR-Numbers-Printed-15random\train\img",
    r"C:\shared\data ocr\letters_9_may_preview\train",
    r"C:\shared\data ocr\TrOCR\train",
    r"C:\shared\data ocr\HandwrittenKazakhRussian\train\img",
]

VAL_CSVS = [
    r"C:/shared/data ocr/school_notebooks_RU/val.csv",
    r"C:/shared/data ocr/CyrillicHandwritingDataset/test.csv",
    r"C:/shared/data ocr/YeniseiGovReports-HWR/val/labels.csv",
    r"C:/shared/data ocr/YeniseiGovReports-PRT/val/labels.csv",
    r"C:\shared\data ocr\RussianSchoolEssays\test\test_v2.csv",
    r"C:\shared\data ocr\DonkeySmallOCR-Numbers-Printed-15random\val\labels.csv",
    r"C:/shared/data ocr/letters_9_may_preview\val.csv",
    r"C:\shared\data ocr\TrOCR\val.csv",
    r"C:\shared\data ocr\HandwrittenKazakhRussian\val\gt.txt",
]

VAL_ROOTS = [
    r"C:/shared/data ocr/school_notebooks_RU/val",
    r"C:/shared/data ocr/CyrillicHandwritingDataset/test",
    r"C:/shared/data ocr/YeniseiGovReports-HWR/val",
    r"C:/shared/data ocr/YeniseiGovReports-PRT/val",
    r"C:\shared\data ocr\RussianSchoolEssays\test\test_images_v2",
    r"C:\shared\data ocr\DonkeySmallOCR-Numbers-Printed-15random\val\img",
    r"C:/shared/data ocr/letters_9_may_preview/val",
    r"C:\shared\data ocr\TrOCR\val",
    r"C:\shared\data ocr\HandwrittenKazakhRussian\val\img",
]

CONFIG_PATH = "configs/rec/PP-OCRv5/multi_language/cyrillic_PP-OCRv5_mobile_rec.yaml"
# Start from the best checkpoint of g1 (acc=76.4%, epoch 11).
# PaddleOCR loads .pdparams; the path is without extension.
PRETRAINED_MODEL = "output/custom_ppocrv5_rec_g1/best_accuracy"

# Reuse the already-prepared data from g1 — no need to regenerate.
PREPARED_DATA_DIR = "train_data/custom_ppocrv5_rec_g1"
SAVE_MODEL_DIR = "output/custom_ppocrv5_rec_g2"

MAX_TEXT_LENGTH = 40

# Leave empty for Windows single-GPU training.
GPUS = ""

EPOCH_NUM = 50
BATCH_SIZE = None
TRAIN_NUM_WORKERS = None
EVAL_NUM_WORKERS = None

# Fine-tuning LR — 10x lower than the default 0.0005 used in g1.
# NOTE: must be written as plain decimal (not 5e-05) for PaddleOCR -o parsing.
LEARNING_RATE = 0.00005
WARMUP_EPOCH = 2

# Set False if you only want to prepare train/val files and inspect stats.
RUN_TRAIN = True

# The alphabet is strict:
# - if a label contains any symbol outside this alphabet, the sample is skipped
# - if a label is longer than MAX_TEXT_LENGTH, the sample is skipped
#
# Reserved tokens <PAD>, <SOS>, <EOS> are accepted in this config block but are not
# written into the PaddleOCR dictionary, because PaddleOCR handles special tokens itself.
ALPHABET_TEXT = """
<PAD>
<SOS>
<EOS>
 
a
b
c
d
e
f
g
h
i
j
k
l
m
n
o
p
q
r
s
t
u
v
w
x
y
z
A
B
C
D
E
F
G
H
I
J
K
L
M
N
O
P
Q
R
S
T
U
V
W
X
Y
Z
0
1
2
3
4
5
6
7
8
9
а
б
в
г
д
е
ё
ж
з
и
й
к
л
м
н
о
п
р
с
т
у
ф
х
ц
ч
ш
щ
ъ
ы
ь
э
ю
я
А
Б
В
Г
Д
Е
Ё
Ж
З
И
Й
К
Л
М
Н
О
П
Р
С
Т
У
Ф
Х
Ц
Ч
Ш
Щ
Ъ
Ы
Ь
Э
Ю
Я
ѣ
Ѣ
і
І
ѳ
Ѳ
ѵ
Ѵ
ѫ
Ѫ
ѭ
Ѭ
ѯ
Ѯ
ѱ
Ѱ
ѡ
Ѡ
ѕ
Ѕ
ѧ
Ѧ
ѩ
Ѩ
.
,
:
;
!
?
-
–
—
…
«
»
(
)
[
]
{
}
"
'
`
/
\
|
_
+
=
*
^
%
$
#
@
&
<
>
~
№
"""


RESERVED_TOKENS = {"<PAD>", "<SOS>", "<EOS>"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
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
    lines = ALPHABET_TEXT.splitlines()
    entries = [line for line in lines if line != ""]
    entries = unique_preserve_order(entries)
    if not entries:
        raise RuntimeError("ALPHABET_TEXT is empty.")

    dict_entries = [item for item in entries if item not in RESERVED_TOKENS]
    if not dict_entries:
        raise RuntimeError("Alphabet has no usable symbols after removing reserved tokens.")

    single_char_entries = [item for item in dict_entries if len(item) == 1]
    allowed_chars = set(single_char_entries)
    use_space_char = " " in allowed_chars

    duplicates = []
    seen = set()
    for item in entries:
        if item in seen and item not in duplicates:
            duplicates.append(item)
        seen.add(item)
    if duplicates:
        raise RuntimeError(
            "Alphabet contains duplicate entries: "
            + ", ".join(repr(item) for item in duplicates[:20])
        )

    return entries, dict_entries, allowed_chars, use_space_char


def write_dict_file(dict_path, dict_entries):
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    with dict_path.open("w", encoding="utf-8", newline="") as handle:
        for item in dict_entries:
            handle.write(item + "\n")


def read_text_with_fallbacks(file_path):
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode annotation file: {file_path}")


def parse_annotation_rows(file_path):
    text = read_text_with_fallbacks(file_path)
    rows = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if "\t" in raw_line:
            parts = raw_line.split("\t", 1)
        else:
            parts = next(csv.reader([raw_line]))
            if len(parts) > 2:
                parts = [parts[0], ",".join(parts[1:])]
        if len(parts) < 2:
            continue
        rows.append([parts[0], parts[1]])
    return rows


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


def build_image_index(image_root):
    index = defaultdict(list)
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.name].append(path.resolve())
    return index


def resolve_image_path(image_root, annotation_path, image_ref, image_index):
    ref_path = Path(image_ref)
    if ref_path.is_absolute() and ref_path.is_file():
        return ref_path.resolve()

    candidate = (image_root / ref_path).resolve()
    if candidate.is_file():
        return candidate

    candidate = (annotation_path.parent / ref_path).resolve()
    if candidate.is_file():
        return candidate

    matches = image_index.get(ref_path.name, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Ambiguous image reference '{image_ref}' for {annotation_path}. "
            f"Found {len(matches)} matching files in {image_root}."
        )

    raise FileNotFoundError(
        f"Image '{image_ref}' from {annotation_path} was not found in {image_root}."
    )


def read_entries(dataset_name, annotation_path, image_root):
    annotation_path = resolve_path(annotation_path)
    image_root = resolve_path(image_root)

    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")

    image_index = build_image_index(image_root)
    rows = parse_annotation_rows(annotation_path)
    if rows and detect_header(rows[0]):
        rows = rows[1:]

    entries = []
    for row in rows:
        image_ref = row[0].strip()
        label = sanitize_label(row[1])
        if not image_ref or not label:
            continue
        image_path = resolve_image_path(image_root, annotation_path, image_ref, image_index)
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
    if LEARNING_RATE is not None:
        command.append(f"Optimizer.lr.learning_rate={LEARNING_RATE:.10f}".rstrip('0').rstrip('.'))
    if WARMUP_EPOCH is not None:
        command.append(f"Optimizer.lr.warmup_epoch={WARMUP_EPOCH}")

    return command


def print_table(rows):
    headers = [
        "dataset",
        "train",
        "test",
        "train_raw",
        "train_skip_len",
        "train_skip_chars",
        "test_raw",
        "test_skip_len",
        "test_skip_chars",
    ]

    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    print(" | ".join(header.ljust(widths[header]) for header in headers))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(" | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers))


def validate_parallel_lists():
    lengths = {
        "DATASET_NAMES": len(DATASET_NAMES),
        "TRAIN_CSVS": len(TRAIN_CSVS),
        "TRAIN_ROOTS": len(TRAIN_ROOTS),
        "VAL_CSVS": len(VAL_CSVS),
        "VAL_ROOTS": len(VAL_ROOTS),
    }
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise RuntimeError(f"Parallel config lists have different lengths: {lengths}")


def main():
    validate_parallel_lists()

    config_path = resolve_path(CONFIG_PATH)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    _, dict_entries, allowed_chars, use_space_char = load_alphabet()

    prepared_data_dir = resolve_path(PREPARED_DATA_DIR)
    train_list_path = prepared_data_dir / "train_list.txt"
    val_list_path = prepared_data_dir / "val_list.txt"
    dict_path = prepared_data_dir / "custom_dict.txt"
    summary_path = prepared_data_dir / "dataset_summary.json"

    write_dict_file(dict_path, dict_entries)

    all_train_entries = []
    all_val_entries = []
    summary_rows = []
    summary_json = {
        "config": config_path.as_posix(),
        "pretrained_model": PRETRAINED_MODEL,
        "prepared_data_dir": prepared_data_dir.as_posix(),
        "save_model_dir": resolve_path(SAVE_MODEL_DIR).as_posix(),
        "max_text_length": MAX_TEXT_LENGTH,
        "dictionary_path": dict_path.as_posix(),
        "dictionary_size": len(dict_entries),
        "datasets": [],
    }

    for idx, dataset_name in enumerate(DATASET_NAMES):
        train_entries = read_entries(dataset_name, TRAIN_CSVS[idx], TRAIN_ROOTS[idx])
        val_entries = read_entries(dataset_name, VAL_CSVS[idx], VAL_ROOTS[idx])

        filtered_train, train_stats, train_invalid = filter_entries(train_entries, allowed_chars)
        filtered_val, val_stats, val_invalid = filter_entries(val_entries, allowed_chars)

        all_train_entries.extend(filtered_train)
        all_val_entries.extend(filtered_val)

        summary_rows.append(
            {
                "dataset": dataset_name,
                "train": len(filtered_train),
                "test": len(filtered_val),
                "train_raw": train_stats["raw_total"],
                "train_skip_len": train_stats["skipped_too_long"],
                "train_skip_chars": train_stats["skipped_bad_chars"],
                "test_raw": val_stats["raw_total"],
                "test_skip_len": val_stats["skipped_too_long"],
                "test_skip_chars": val_stats["skipped_bad_chars"],
            }
        )
        summary_json["datasets"].append(
            {
                "name": dataset_name,
                "train_csv": resolve_path(TRAIN_CSVS[idx]).as_posix(),
                "train_root": resolve_path(TRAIN_ROOTS[idx]).as_posix(),
                "val_csv": resolve_path(VAL_CSVS[idx]).as_posix(),
                "val_root": resolve_path(VAL_ROOTS[idx]).as_posix(),
                "train_samples_kept": len(filtered_train),
                "test_samples_kept": len(filtered_val),
                "train_raw": train_stats["raw_total"],
                "train_skipped_too_long": train_stats["skipped_too_long"],
                "train_skipped_bad_chars": train_stats["skipped_bad_chars"],
                "test_raw": val_stats["raw_total"],
                "test_skipped_too_long": val_stats["skipped_too_long"],
                "test_skipped_bad_chars": val_stats["skipped_bad_chars"],
                "train_bad_chars_top20": "".join(char for char, _ in train_invalid.most_common(20)),
                "test_bad_chars_top20": "".join(char for char, _ in val_invalid.most_common(20)),
            }
        )

    if not all_train_entries:
        raise RuntimeError("No training samples remain after filtering.")
    if not all_val_entries:
        raise RuntimeError("No test samples remain after filtering.")

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
    print_table(summary_rows)
    print()
    print(f"Total train samples: {len(all_train_entries)}")
    print(f"Total test samples: {len(all_val_entries)}")
    print()
    print("Training command:")
    print(" ".join(f'\"{part}\"' if " " in part else part for part in command))

    if not RUN_TRAIN:
        return

    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
