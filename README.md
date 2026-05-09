# Single-Class YOLO Dataset Filtering and Model Comparison

This repo contains a local Roboflow YOLO export in `Mallet/`. Its `data.yaml` lists:

```yaml
names: ['bottle', 'mallet']
```

The requested `hammer_yolo` dataset is created from the `mallet` source class and renamed to `hammer` in the filtered output. If you later export a dataset with a literal `hammer` class, `filter_dataset.py` will use it automatically.

## Git Ignore Policy

The repo intentionally ignores bulky or generated local artifacts:

- `Mallet/` raw Roboflow export
- `datasets/` filtered datasets and offline augmented training copies
- `runs/` training outputs
- `failure_cases/` evaluation overlays
- model exports such as `*.pt`, `*.onnx`, `*.tflite`, and related formats
- Python caches, virtual environments, and `.DS_Store`

Keep the scripts, this README, and `comparison_table.csv` tracked. Recreate the ignored artifacts with the workflow below.

## 1. Create Single-Class Datasets

```bash
python3 filter_dataset.py --source Mallet --output-root datasets --overwrite
```

This creates:

- `datasets/bottle_yolo`
- `datasets/hammer_yolo`

Each output preserves `train`, `valid`, and `test`, keeps only rows for the target class, remaps that class to `0`, skips images with empty final labels, and writes a one-class `data.yaml`.

## 2. Install Training Dependencies

```bash
python3 -m pip install ultralytics pillow numpy
```

Use the correct PyTorch build for your machine if Ultralytics does not install one that supports your hardware.

## 3. Train All Models

```bash
python3 train_model_suite.py --epochs 100 --batch 16 --device mps
```

Use `--device 0` for CUDA, `--device cpu` for CPU, or omit `--device` to let Ultralytics choose.

Models trained for each dataset:

- `YOLOv8n`
- `YOLOv8s`
- `YOLOv8m`
- `YOLO11n`
- `YOLO11s`
- `YOLO11m`

Training outputs are written under `runs/model_suite/`.

### Bottle Augmentation Guidance

`train_model_suite.py` applies stronger bottle augmentation so the model does not overfit to bottle color. It combines Ultralytics HSV, brightness, geometric, mosaic, mixup, copy-paste, and erasing augmentations with an offline augmented bottle training copy that includes grayscale, brightness/contrast/exposure-like shifts, blur, and noise.

The intent is to push the bottle models toward shape, neck-body geometry, edges, and silhouette rather than color.

### Hammer Augmentation Guidance

Hammer training uses milder color augmentation so consistent color cues can still help, while geometric augmentation encourages learning handle-head structure, proportions, edges, and object geometry.

## 4. Evaluate and Compare

```bash
python3 evaluate_models.py --split test --device mps
```

This writes:

- `comparison_table.csv`
- `failure_cases/bottle/<model>/`
- `failure_cases/hammer/<model>/`

The comparison table includes:

- `mAP50`
- `mAP50-95`
- `precision`
- `recall`
- `false_positives`
- `false_negatives`
- `inference_speed_ms_per_image`
- `model_size_mb`
- failure-case image counts and folders

Failure-case images draw ground truth boxes in green and predictions in red.

## Useful Options

Train only selected models:

```bash
python3 train_model_suite.py --models YOLOv8n YOLO11n --epochs 50 --device mps
```

Reduce bottle offline augmentation:

```bash
python3 train_model_suite.py --bottle-offline-augments 1
```

Evaluate validation instead of test:

```bash
python3 evaluate_models.py --split valid
```

## Expected Workflow

```bash
python3 filter_dataset.py --overwrite
python3 train_model_suite.py --epochs 100 --batch 16 --device mps
python3 evaluate_models.py --split test --device mps
```
