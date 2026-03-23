# YOLO Carton Detection Project

This repository is a modular YOLO-based object detection project for carton recognition. It was built around a CVAT YOLO export, supports a very simple day-to-day inference workflow, and now includes stronger dataset inspection, reproducible training, config-driven commands, and a Raspberry Pi export path.

The project is designed for two realities at the same time:

- easy local usage with short commands such as `image` and `webcam`
- production-minded growth for training, dataset QA, and edge deployment

## Quick Start

These two commands were tested in this project and are the fastest way to run inference with your trained model:

```bash
.venv/bin/image --model-path runs/train/carton_detector_gpu/weights/best.pt
.venv/bin/webcam --model-path runs/train/carton_detector_gpu/weights/best.pt
```

If your virtual environment is activated correctly, the same commands also work without the full path:

```bash
image --model-path runs/train/carton_detector_gpu/weights/best.pt
webcam --model-path runs/train/carton_detector_gpu/weights/best.pt
```

## Project Goal

The goal is to detect your carton classes, not generic COCO classes. The default pretrained YOLO weights know classes such as `person`, `chair`, and `tv`, but your real dataset classes are:

- `Milch_Karton_shokolade`
- `Miclh_Karton_Vanille`
- `Teeschachtel`
- `Cube_Karton`

That means the correct workflow is:

1. Inspect the CVAT export.
2. Prepare a clean training dataset.
3. Train a custom model.
4. Run inference with the trained `best.pt`.
5. Export edge-friendly artifacts for Raspberry Pi deployment.

## What Changed In The Latest Architecture

The project now includes the three high-impact upgrades we discussed:

1. Custom training is more stable.
The dataset is inspected before training, invalid or missing labels are reported, split metadata is saved, and training writes a reproducible summary.

2. Config + subcommand architecture is now available.
The simple launchers still work, but the internal CLI now also supports `predict`, `inspect-dataset`, `prepare-dataset`, `train`, and `export`.

3. Raspberry Pi export support was added.
The project can export trained weights to deployment-oriented formats and generate an export manifest plus labels file for edge packaging.

## Current Folder Structure

```text
.
├── README.md
├── requirements.txt
├── main.py
├── train.py
├── export.py
├── auto_git_manager.py
├── configs/
│   └── defaults.yaml
├── tools/
│   ├── auto_git_manager.py
│   └── install_shortcuts.py
├── tests/
│   ├── test_cli.py
│   └── test_dataset_manager.py
├── data/
│   ├── cvat_exports/
│   │   └── caton_hause/
│   ├── images/
│   ├── videos/
│   └── processed/
├── models/
├── runs/
├── logs/
└── yolo_edge/
    ├── __init__.py
    ├── cli.py
    ├── config.py
    ├── training.py
    ├── edge_export.py
    ├── core/
    │   ├── detector.py
    │   └── video_streamer.py
    ├── data/
    │   └── dataset_manager.py
    └── utils/
        └── logging_utils.py
```

## Installation

Create and populate the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
.venv/bin/python3 tools/install_shortcuts.py
```

That shortcut installer creates these launcher commands in `.venv/bin`:

- `image`
- `video`
- `webcam`
- `external-camera`
- `external_camera`
- `inspect-dataset`
- `prepare-dataset`
- `train-carton`
- `export-edge`

If the virtual environment activation on your machine is broken, use the interpreter directly:

```bash
.venv/bin/python3
```

## Configuration

Default values now live in:

```text
configs/defaults.yaml
```

This file controls:

- default model path
- image and video directories
- logging defaults
- dataset paths
- training defaults
- export defaults

You can keep using the simple commands and only change `configs/defaults.yaml` when you want project-wide defaults.

## Dataset Status

Dataset root:

```text
data/cvat_exports/caton_hause/
```

The CVAT export was inspected carefully. Current findings:

- `train.txt` references `74` images
- only `65` label files exist
- `9` images are missing labels
- training currently uses only the labeled subset

Known missing label stems:

- `emrah_carton_hause_19`
- `emrah_carton_hause_40`
- `emrah_carton_hause_46`
- `emrah_carton_hause_51`
- `emrah_carton_hause_66`
- `emrah_carton_hause_67`
- `emrah_carton_hause_68`
- `emrah_carton_hause_69`
- `emrah_carton_hause_71`

The dataset manager now also validates label file format and normalized coordinate ranges before training starts.

## Simple Inference Commands

The daily workflow stays simple.

### Image

```bash
.venv/bin/image --model-path runs/train/carton_detector_gpu/weights/best.pt
```

Behavior:

- reads images from `data/images/` by default
- plays them one after another automatically
- keeps each image visible for `2` seconds by default
- `q` or `Esc` stops playback

### Video

```bash
.venv/bin/video --model-path runs/train/carton_detector_gpu/weights/best.pt
```

If there are multiple videos, choose one explicitly:

```bash
.venv/bin/video \
    --path data/videos/example.mp4 \
    --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### Webcam

```bash
.venv/bin/webcam --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### External Camera

```bash
.venv/bin/external-camera --model-path runs/train/carton_detector_gpu/weights/best.pt
```

This uses camera index `1` by default.

### RTSP Stream

```bash
.venv/bin/webcam \
    --camera-index rtsp://username:password@camera-ip:554/stream \
    --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### Display Scaling

Large images are scaled down for display automatically. Override if needed:

```bash
.venv/bin/image \
    --model-path runs/train/carton_detector_gpu/weights/best.pt \
    --display-max-width 900 \
    --display-max-height 700
```

## New Subcommand Architecture

The same project now also supports a more professional command structure.

### Predict

```bash
.venv/bin/python3 main.py predict image --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### Inspect Dataset

```bash
.venv/bin/python3 main.py inspect-dataset
```

### Prepare Dataset

```bash
.venv/bin/python3 main.py prepare-dataset --overwrite
```

### Train

```bash
.venv/bin/python3 main.py train --overwrite
```

### Export

```bash
.venv/bin/python3 main.py export --source-model runs/train/carton_detector_gpu/weights/best.pt
```

These subcommands are the better long-term interface for automation and MLOps-style workflows, while the short launchers remain the easiest day-to-day commands.

## Training The Custom Carton Model

Start training with the standalone training entry point:

```bash
.venv/bin/python3 train.py \
    --dataset-root data/cvat_exports/caton_hause \
    --image-source-dir data/images \
    --prepared-dataset-dir data/processed/caton_hause \
    --base-model models/yolov8n.pt \
    --epochs 50 \
    --image-size 640 \
    --batch-size 8 \
    --overwrite
```

You can also use the launcher:

```bash
.venv/bin/train-carton --overwrite
```

Training improvements now included:

- GPU is selected automatically when CUDA is available
- dataset inspection happens before training
- training/validation split metadata is saved
- dataset report is written to `data/processed/.../dataset_report.yaml`
- training summary is written to `runs/train/<run-name>/training_summary.yaml`
- deterministic settings are applied for reproducibility

Expected trained model output:

```text
runs/train/carton_detector_gpu/weights/best.pt
```

## Export For Raspberry Pi

After training, export the model for edge deployment:

```bash
.venv/bin/export-edge \
    --source-model runs/train/carton_detector_gpu/weights/best.pt
```

By default, the export workflow is configured to create these formats:

- `onnx`
- `openvino`
- `tflite`

Export outputs are stored under:

```text
runs/export/
```

The export workflow also writes:

- `labels.txt`
- `edge_export_manifest.yaml`

If a benchmark image is configured, the native `.pt` model is also benchmarked and the latency summary is logged.

## Logging

Application logs are written to:

```text
logs/application.log
```

Current logging includes:

- model loading
- dataset inspection summaries
- training configuration summaries
- per-frame detection summaries
- class counts and total detections
- edge export actions
- benchmark summary for export workflows

Example inference log:

```text
2026-03-23 21:52:33,297 | INFO | yolo_edge.cli | source=emrah_carton_hause_1.jpeg frame=0 detected=yes total=6 classes={'chair': 2, 'person': 2, 'potted plant': 1, 'tv': 1} average_confidence=0.5268
```

After you use your trained carton model, this summary should reflect your custom classes instead of COCO classes.

## Raspberry Pi 5 Deployment

This project is already prepared for a Raspberry Pi focused next step, especially for a lightweight AI camera pipeline.

Recommended path:

1. Train on the development machine with GPU.
2. Export edge artifacts with `export-edge`.
3. Copy the chosen export artifact, `labels.txt`, and `edge_export_manifest.yaml` to the Raspberry Pi.
4. Validate latency and thermals on the Pi with the real camera stream.
5. Keep image size conservative for stable frame rate and memory usage.

Practical recommendations for Raspberry Pi 5:

- prefer the smallest model that still meets accuracy needs
- start with `640` or lower image size
- avoid unnecessary UI overhead
- save logs and outputs only when needed
- benchmark with the real AI camera stream, not only static images

For Raspberry Pi AI camera integration specifically:

- keep the trained labels file with the model artifact
- validate whether your camera stack expects ONNX, TFLite, OpenVINO, or another conversion step
- treat `runs/export/edge_export_manifest.yaml` as the deployment handoff document

## Tests

Run the lightweight regression tests with:

```bash
.venv/bin/python3 -m unittest discover -s tests -v
```

## Requirements

Main dependencies:

- `numpy`
- `ultralytics`
- `opencv-python`
- `PyYAML`

## Git Automation

If you want automatic commits and pushes in the background:

```bash
.venv/bin/python3 auto_git_manager.py
```

## Recommended Next Technical Steps

The current architecture is now ready for the next layer of improvement:

1. add richer evaluation outputs such as confusion matrices and failure-case samples
2. add JSONL inference logging for later analytics
3. add Raspberry Pi specific runtime benchmarks against the exported models
