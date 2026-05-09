#!/usr/bin/env python3
"""Validate trained models, count FP/FN, and save failure-case images."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_NAMES = ("YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLO11n", "YOLO11s", "YOLO11m")


@dataclass
class Box:
    xyxy: tuple[float, float, float, float]
    conf: float = 1.0


def find_models(runs_root: Path) -> list[tuple[str, str, Path]]:
    found: list[tuple[str, str, Path]] = []
    for weight_path in sorted(runs_root.glob("**/weights/best.pt")):
        run_name = weight_path.parents[1].name
        dataset = "bottle_yolo" if "bottle_yolo" in run_name else "hammer_yolo" if "hammer_yolo" in run_name else ""
        model_name = next((name for name in MODEL_NAMES if name in run_name), "")
        if dataset and model_name:
            found.append((dataset, model_name, weight_path))
    return found


def split_images(dataset_dir: Path, split: str) -> list[Path]:
    return sorted(p for p in (dataset_dir / split / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS)


def yolo_to_xyxy(row: str, width: int, height: int) -> Box:
    parts = row.split()
    _, xc, yc, bw, bh = map(float, parts[:5])
    x1 = (xc - bw / 2) * width
    y1 = (yc - bh / 2) * height
    x2 = (xc + bw / 2) * width
    y2 = (yc + bh / 2) * height
    return Box((x1, y1, x2, y2))


def load_gt_boxes(label_path: Path, width: int, height: int) -> list[Box]:
    if not label_path.exists():
        return []
    return [
        yolo_to_xyxy(line.strip(), width, height)
        for line in label_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def count_failures(gt_boxes: list[Box], pred_boxes: list[Box], iou_threshold: float) -> tuple[int, int]:
    matched_gt: set[int] = set()
    false_positives = 0
    for pred in sorted(pred_boxes, key=lambda box: box.conf, reverse=True):
        best_i = -1
        best_iou = 0.0
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            score = iou(pred, gt)
            if score > best_iou:
                best_i = i
                best_iou = score
        if best_i >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_i)
        else:
            false_positives += 1
    false_negatives = len(gt_boxes) - len(matched_gt)
    return false_positives, false_negatives


def draw_failure(image_path: Path, gt_boxes: list[Box], pred_boxes: list[Box], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box in gt_boxes:
        draw.rectangle(box.xyxy, outline="lime", width=3)
        draw.text((box.xyxy[0] + 2, box.xyxy[1] + 2), "GT", fill="lime", font=font)
    for box in pred_boxes:
        draw.rectangle(box.xyxy, outline="red", width=3)
        draw.text((box.xyxy[0] + 2, max(0, box.xyxy[1] - 12)), f"P {box.conf:.2f}", fill="red", font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def metric_value(metrics, *names: str) -> float:
    value = metrics
    for name in names:
        value = getattr(value, name)
    try:
        return float(value)
    except TypeError:
        return float(value())


def speed_ms(metrics) -> float:
    speed = getattr(metrics, "speed", {}) or {}
    return float(speed.get("inference", 0.0))


def evaluate_one(model, dataset_dir: Path, split: str, failure_dir: Path, max_failures: int, conf: float, iou_threshold: float):
    images = split_images(dataset_dir, split)
    total_fp = 0
    total_fn = 0
    saved = 0
    for image_path in images:
        with Image.open(image_path) as image:
            width, height = image.size
        gt_boxes = load_gt_boxes(dataset_dir / split / "labels" / f"{image_path.stem}.txt", width, height)
        result = model.predict(str(image_path), conf=conf, verbose=False)[0]
        pred_boxes = [
            Box(tuple(map(float, xyxy)), float(score))
            for xyxy, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist())
        ]
        fp, fn = count_failures(gt_boxes, pred_boxes, iou_threshold)
        total_fp += fp
        total_fn += fn
        if (fp or fn) and saved < max_failures:
            draw_failure(image_path, gt_boxes, pred_boxes, failure_dir / f"{image_path.stem}_fp{fp}_fn{fn}.jpg")
            saved += 1
    return total_fp, total_fn, saved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", default="datasets", type=Path)
    parser.add_argument("--runs-root", default="runs/model_suite", type=Path)
    parser.add_argument("--output", default="comparison_table.csv", type=Path)
    parser.add_argument("--failure-root", default="failure_cases", type=Path)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.50, type=float)
    parser.add_argument("--max-failures", default=25, type=int)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required: pip install ultralytics") from exc

    rows = []
    for dataset_name, model_name, weight_path in find_models(args.runs_root):
        dataset_dir = args.datasets_root / dataset_name
        data_yaml = dataset_dir / "data.yaml"
        model = YOLO(str(weight_path))
        val_kwargs = {"data": str(data_yaml.resolve()), "split": args.split, "conf": args.conf, "iou": args.iou, "verbose": False}
        if args.device is not None:
            val_kwargs["device"] = args.device
        metrics = model.val(**val_kwargs)

        failure_dir = args.failure_root / dataset_name.replace("_yolo", "") / model_name
        fp, fn, saved = evaluate_one(model, dataset_dir, args.split, failure_dir, args.max_failures, args.conf, args.iou)

        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "weights": str(weight_path),
                "mAP50": metric_value(metrics.box, "map50"),
                "mAP50-95": metric_value(metrics.box, "map"),
                "precision": metric_value(metrics.box, "mp"),
                "recall": metric_value(metrics.box, "mr"),
                "false_positives": fp,
                "false_negatives": fn,
                "inference_speed_ms_per_image": speed_ms(metrics),
                "model_size_mb": weight_path.stat().st_size / (1024 * 1024),
                "failure_case_images": saved,
                "failure_case_dir": str(failure_dir),
            }
        )
        print(f"Evaluated {dataset_name} {model_name}: FP={fp} FN={fn}")

    fieldnames = [
        "dataset",
        "model",
        "weights",
        "mAP50",
        "mAP50-95",
        "precision",
        "recall",
        "false_positives",
        "false_negatives",
        "inference_speed_ms_per_image",
        "model_size_mb",
        "failure_case_images",
        "failure_case_dir",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
