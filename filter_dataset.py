#!/usr/bin/env python3
"""Split a Roboflow YOLO dataset into single-class YOLO datasets."""

from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def parse_data_yaml(path: Path) -> dict:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ImportError:
        data: dict[str, object] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            if key.strip() == "names":
                data["names"] = ast.literal_eval(value)
            elif key.strip() == "nc":
                data["nc"] = int(value)
            else:
                data[key.strip()] = value
        return data


def normalize_names(names: object) -> list[str]:
    if isinstance(names, list):
        return [str(name) for name in names]
    if isinstance(names, dict):
        return [str(names[i]) for i in sorted(names)]
    raise ValueError("data.yaml must contain a YOLO names list or dict")


def resolve_class_id(names: list[str], requested: str, aliases: list[str]) -> tuple[int, str]:
    candidates = [requested, *aliases]
    lowered = {name.lower(): i for i, name in enumerate(names)}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()], candidate
    raise ValueError(
        f"Could not find class '{requested}' in data.yaml names={names}. "
        f"Tried aliases: {aliases}"
    )


def split_image_dir(dataset_root: Path, yaml_data: dict, split: str) -> Path:
    yaml_key = "val" if split == "valid" else split
    configured = yaml_data.get(yaml_key)
    if configured:
        candidate = (dataset_root / str(configured)).resolve()
        if candidate.exists():
            return candidate
    fallback = dataset_root / split / "images"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Could not find images for split '{split}'")


def image_files(image_dir: Path) -> list[Path]:
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def filtered_label_rows(label_path: Path, class_id: int) -> list[str]:
    rows: list[str] = []
    if not label_path.exists():
        return rows

    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            row_class_id = int(float(parts[0]))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"Invalid YOLO label row in {label_path}:{line_number}: {raw_line}") from exc
        if row_class_id == class_id:
            rows.append(" ".join(["0", *parts[1:]]))
    return rows


def write_data_yaml(output_dir: Path, class_name: str) -> None:
    content = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        f"names: ['{class_name}']\n"
    )
    (output_dir / "data.yaml").write_text(content, encoding="utf-8")


def build_single_class_dataset(
    dataset_root: Path,
    yaml_data: dict,
    output_root: Path,
    output_name: str,
    source_class_id: int,
    output_class_name: str,
    overwrite: bool,
) -> dict[str, int]:
    output_dir = output_root / output_name
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    elif output_dir.exists():
        raise FileExistsError(f"{output_dir} already exists. Use --overwrite to replace it.")

    stats = {"images": 0, "labels": 0, "boxes": 0}
    for split in SPLITS:
        src_images = split_image_dir(dataset_root, yaml_data, split)
        src_labels = src_images.parent / "labels"
        dst_images = output_dir / split / "images"
        dst_labels = output_dir / split / "labels"
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        for image_path in image_files(src_images):
            label_path = src_labels / f"{image_path.stem}.txt"
            rows = filtered_label_rows(label_path, source_class_id)
            if not rows:
                continue
            shutil.copy2(image_path, dst_images / image_path.name)
            (dst_labels / f"{image_path.stem}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            stats["images"] += 1
            stats["labels"] += 1
            stats["boxes"] += len(rows)

    write_data_yaml(output_dir, output_class_name)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Mallet", type=Path, help="Roboflow YOLO export directory")
    parser.add_argument("--output-root", default="datasets", type=Path, help="Directory for filtered datasets")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing filtered datasets")
    parser.add_argument(
        "--hammer-aliases",
        nargs="*",
        default=["mallet"],
        help="Class names to accept when data.yaml does not contain a literal 'hammer' class",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    data_yaml = source / "data.yaml"
    yaml_data = parse_data_yaml(data_yaml)
    names = normalize_names(yaml_data["names"])

    bottle_id, bottle_source = resolve_class_id(names, "bottle", [])
    hammer_id, hammer_source = resolve_class_id(names, "hammer", args.hammer_aliases)

    specs = [
        ("bottle_yolo", "bottle", bottle_id, bottle_source),
        ("hammer_yolo", "hammer", hammer_id, hammer_source),
    ]

    args.output_root.mkdir(parents=True, exist_ok=True)
    for output_name, output_class_name, class_id, source_name in specs:
        stats = build_single_class_dataset(
            dataset_root=source,
            yaml_data=yaml_data,
            output_root=args.output_root,
            output_name=output_name,
            source_class_id=class_id,
            output_class_name=output_class_name,
            overwrite=args.overwrite,
        )
        alias_note = "" if source_name == output_class_name else f" (source class: {source_name})"
        print(
            f"{output_name}: class_id={class_id}{alias_note}, "
            f"images={stats['images']}, labels={stats['labels']}, boxes={stats['boxes']}"
        )


if __name__ == "__main__":
    main()
