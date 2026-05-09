#!/usr/bin/env python3
"""Train YOLOv8 and YOLO11 model suites on the filtered single-class datasets."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


MODEL_WEIGHTS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "YOLO11m": "yolo11m.pt",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def augment_bottle_training_set(dataset_dir: Path, output_dir: Path, variants: int) -> Path:
    """Create an augmented bottle training copy with grayscale, blur, noise, and lighting variation."""
    if variants <= 0:
        return dataset_dir

    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError as exc:
        raise RuntimeError("Pillow is required for offline bottle augmentations: pip install pillow") from exc

    copy_tree(dataset_dir, output_dir)
    train_images = output_dir / "train" / "images"
    train_labels = output_dir / "train" / "labels"

    originals = sorted(p for p in train_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for image_path in originals:
        label_path = train_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for i in range(variants):
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if random.random() < 0.45:
                    image = image.convert("L").convert("RGB")
                image = ImageEnhance.Color(image).enhance(random.uniform(0.25, 1.8))
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.55, 1.55))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.65, 1.65))
                if random.random() < 0.35:
                    image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.6)))
                if random.random() < 0.45:
                    image = add_noise(image, amount=random.uniform(6.0, 18.0))

                out_name = f"{image_path.stem}_shape_aug_{i}{image_path.suffix}"
                image.save(train_images / out_name, quality=95)
                shutil.copy2(label_path, train_labels / f"{Path(out_name).stem}.txt")

    return output_dir


def add_noise(image, amount: float):
    import numpy as np
    from PIL import Image

    arr = np.asarray(image).astype("int16")
    noise = np.random.normal(0, amount, arr.shape).astype("int16")
    arr = np.clip(arr + noise, 0, 255).astype("uint8")
    return Image.fromarray(arr)


def train_args_for_dataset(dataset_name: str) -> dict:
    common = {
        "imgsz": 640,
        "patience": 30,
        "cache": False,
        "plots": True,
        "save_json": True,
        "degrees": 8.0,
        "translate": 0.10,
        "scale": 0.45,
        "shear": 2.0,
        "perspective": 0.0005,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "mixup": 0.10,
        "copy_paste": 0.10,
        "erasing": 0.25,
    }
    if "bottle" in dataset_name:
        common.update(
            {
                "hsv_h": 0.08,
                "hsv_s": 0.85,
                "hsv_v": 0.65,
                "degrees": 12.0,
                "scale": 0.55,
                "mixup": 0.18,
                "erasing": 0.40,
            }
        )
    else:
        common.update({"hsv_h": 0.025, "hsv_s": 0.45, "hsv_v": 0.35})
    return common


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", default="datasets", type=Path)
    parser.add_argument("--runs-root", default="runs/model_suite", type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default=None, help="Ultralytics device value, e.g. 0, cpu, mps")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bottle-offline-augments", default=2, type=int)
    parser.add_argument("--models", nargs="*", default=list(MODEL_WEIGHTS), choices=list(MODEL_WEIGHTS))
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required: pip install ultralytics") from exc

    random.seed(args.seed)
    datasets = [args.datasets_root / "bottle_yolo", args.datasets_root / "hammer_yolo"]
    for dataset_dir in datasets:
        if not (dataset_dir / "data.yaml").exists():
            raise FileNotFoundError(f"Missing {dataset_dir / 'data.yaml'}. Run filter_dataset.py first.")

    for dataset_dir in datasets:
        train_dataset_dir = dataset_dir
        if dataset_dir.name == "bottle_yolo":
            train_dataset_dir = augment_bottle_training_set(
                dataset_dir=dataset_dir,
                output_dir=args.datasets_root / "bottle_yolo_train_augmented",
                variants=args.bottle_offline_augments,
            )

        for model_name in args.models:
            model = YOLO(MODEL_WEIGHTS[model_name])
            run_name = f"{dataset_dir.name}_{model_name}"
            train_kwargs = train_args_for_dataset(dataset_dir.name)
            train_kwargs.update(
                {
                    "data": str((train_dataset_dir / "data.yaml").resolve()),
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "workers": args.workers,
                    "project": str(args.runs_root.resolve()),
                    "name": run_name,
                    "exist_ok": True,
                    "seed": args.seed,
                }
            )
            if args.device is not None:
                train_kwargs["device"] = args.device
            print(f"Training {run_name} with {MODEL_WEIGHTS[model_name]}")
            model.train(**train_kwargs)


if __name__ == "__main__":
    main()
