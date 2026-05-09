#!/usr/bin/env python3
"""Train YOLOv8 and YOLO11 model suites on the filtered single-class datasets."""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Set


MODEL_WEIGHTS = {
    "YOLOv8n": "yolov8n.pt",
    "YOLOv8s": "yolov8s.pt",
    "YOLOv8m": "yolov8m.pt",
    "YOLO11n": "yolo11n.pt",
    "YOLO11s": "yolo11s.pt",
    "YOLO11m": "yolo11m.pt",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LOGGER = logging.getLogger("train_model_suite")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def count_files(path: Path, suffixes: Optional[Set[str]] = None) -> int:
    if not path.exists():
        return 0
    if suffixes is None:
        return sum(1 for p in path.iterdir() if p.is_file())
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)


def log_dataset_summary(dataset_dir: Path, label: str) -> None:
    LOGGER.info("Dataset summary for %s: %s", label, dataset_dir.resolve())
    for split in ("train", "valid", "test"):
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        LOGGER.info(
            "  %s: %s images, %s labels",
            split,
            count_files(images_dir, IMAGE_EXTS),
            count_files(labels_dir, {".txt"}),
        )


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def augment_bottle_training_set(dataset_dir: Path, output_dir: Path, variants: int) -> Path:
    """Create an augmented bottle training copy with grayscale, blur, noise, and lighting variation."""
    if variants <= 0:
        LOGGER.info("Bottle offline augmentation disabled; using %s", dataset_dir.resolve())
        return dataset_dir

    try:
        from PIL import Image, ImageEnhance, ImageFilter
    except ImportError as exc:
        raise RuntimeError("Pillow is required for offline bottle augmentations: pip install pillow") from exc

    started = time.perf_counter()
    LOGGER.info(
        "Creating bottle offline augmentation copy: %s -> %s with %s variants per image",
        dataset_dir.resolve(),
        output_dir.resolve(),
        variants,
    )
    copy_tree(dataset_dir, output_dir)
    train_images = output_dir / "train" / "images"
    train_labels = output_dir / "train" / "labels"

    originals = sorted(p for p in train_images.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    LOGGER.info("Found %s original bottle training images to augment", len(originals))
    created = 0
    progress_every = max(len(originals) // 10, 1) if originals else 1
    for image_index, image_path in enumerate(originals, start=1):
        label_path = train_labels / f"{image_path.stem}.txt"
        if not label_path.exists():
            LOGGER.warning("Skipping augmentation for %s because label file is missing", image_path.name)
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
                created += 1

        if len(originals) >= 10 and image_index % progress_every == 0:
            LOGGER.info("Augmentation progress: processed %s/%s images", image_index, len(originals))

    LOGGER.info(
        "Bottle offline augmentation complete: created %s images in %.1fs",
        created,
        time.perf_counter() - started,
    )
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
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    configure_logging(args.log_level)

    LOGGER.info("Starting training suite")
    LOGGER.info("Datasets root: %s", args.datasets_root.resolve())
    LOGGER.info("Runs root: %s", args.runs_root.resolve())
    LOGGER.info("Models: %s", ", ".join(args.models))
    LOGGER.info(
        "Training options: epochs=%s batch=%s device=%s workers=%s seed=%s bottle_offline_augments=%s",
        args.epochs,
        args.batch,
        args.device if args.device is not None else "auto",
        args.workers,
        args.seed,
        args.bottle_offline_augments,
    )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required: pip install ultralytics") from exc

    suite_started = time.perf_counter()
    random.seed(args.seed)
    datasets = [args.datasets_root / "bottle_yolo", args.datasets_root / "hammer_yolo"]
    for dataset_dir in datasets:
        if not (dataset_dir / "data.yaml").exists():
            raise FileNotFoundError(f"Missing {dataset_dir / 'data.yaml'}. Run filter_dataset.py first.")
        log_dataset_summary(dataset_dir, dataset_dir.name)

    for dataset_dir in datasets:
        train_dataset_dir = dataset_dir
        if dataset_dir.name == "bottle_yolo":
            train_dataset_dir = augment_bottle_training_set(
                dataset_dir=dataset_dir,
                output_dir=args.datasets_root / "bottle_yolo_train_augmented",
                variants=args.bottle_offline_augments,
            )
            log_dataset_summary(train_dataset_dir, f"{dataset_dir.name} training copy")

        for model_name in args.models:
            run_started = time.perf_counter()
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

            LOGGER.info("Starting run %s with weights %s", run_name, MODEL_WEIGHTS[model_name])
            LOGGER.info("Run output directory: %s", (args.runs_root / run_name).resolve())
            LOGGER.debug("Ultralytics train kwargs for %s: %s", run_name, train_kwargs)
            model.train(**train_kwargs)
            LOGGER.info("Finished run %s in %.1f minutes", run_name, (time.perf_counter() - run_started) / 60.0)

    LOGGER.info("Training suite complete in %.1f minutes", (time.perf_counter() - suite_started) / 60.0)


if __name__ == "__main__":
    main()
