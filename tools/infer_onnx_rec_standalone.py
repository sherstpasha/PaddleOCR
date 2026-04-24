import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--image_dir", required=True, help="Image file or directory")
    parser.add_argument("--dict", required=True, help="Character dictionary path")
    parser.add_argument("--use_space_char", action="store_true")
    parser.add_argument("--rec_image_shape", default="3,48,320")
    return parser.parse_args()


def load_dict(dict_path, use_space_char):
    chars = []
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            chars.append(line.rstrip("\r\n"))
    if use_space_char:
        chars.append(" ")
    # CTC blank token is index 0.
    return ["blank"] + chars


def get_image_paths(image_dir):
    path = Path(image_dir)
    if path.is_file():
        return [path]
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts])


def resize_norm_img(img, rec_image_shape, input_width=None):
    img_c, img_h, img_w = rec_image_shape
    if input_width is not None and input_width > 0:
        img_w = input_width
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = img_w if int(np.ceil(img_h * ratio)) > img_w else int(np.ceil(img_h * ratio))
    resized_image = cv2.resize(img, (resized_w, img_h)).astype("float32")
    resized_image = resized_image.transpose((2, 0, 1)) / 255.0
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((img_c, img_h, img_w), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def ctc_decode(preds, character):
    preds_idx = preds.argmax(axis=2)
    preds_prob = preds.max(axis=2)
    results = []
    for batch_idx in range(preds_idx.shape[0]):
        char_list = []
        conf_list = []
        last_idx = None
        for idx, prob in zip(preds_idx[batch_idx], preds_prob[batch_idx]):
            idx = int(idx)
            if idx == 0 or idx == last_idx:
                last_idx = idx
                continue
            char_list.append(character[idx])
            conf_list.append(float(prob))
            last_idx = idx
        text = "".join(char_list)
        score = float(np.mean(conf_list)) if conf_list else 0.0
        results.append((text, score))
    return results


def main():
    args = parse_args()
    character = load_dict(args.dict, args.use_space_char)
    rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    dynamic_width = None
    if len(input_shape) >= 4 and isinstance(input_shape[3], int):
        dynamic_width = input_shape[3]

    image_paths = get_image_paths(args.image_dir)
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"{image_path}\tERROR\t0.0")
            continue
        norm_img = resize_norm_img(img, rec_image_shape, dynamic_width)
        preds = sess.run(None, {input_name: norm_img[np.newaxis, :]})[0]
        text, score = ctc_decode(preds, character)[0]
        print(f"{image_path}\t{text}\t{score:.6f}")


if __name__ == "__main__":
    main()
