# YOLO Carton Detection Project

This repository is a modular YOLO object detection project for carton recognition. It started from a CVAT YOLO export and was extended into a cleaner training, inference, dataset inspection, and edge deployment workflow.

The project is intentionally designed for two use cases at once:

- simple local usage with short commands such as `image` and `webcam`
- structured long-term growth with reproducible training, configuration, export, and Raspberry Pi deployment

## Current Status

The project is no longer at the "raw dataset only" stage. It has already reached a working custom-model stage.

Current stage:

1. CVAT export was inspected and validated.
2. A cleaned training dataset is generated automatically from labeled samples.
3. A custom YOLO model was trained successfully.
4. Inference works with the trained model on images and webcam.
5. Edge export workflow exists for Raspberry Pi deployment preparation.

Most important current output:

```text
runs/train/carton_detector_gpu/weights/best.pt
```

This is the trained custom model that should be used instead of a generic COCO model such as `yolov8n.pt`.

## What We Did Today

Today the project was reviewed, corrected, extended, and stabilized step by step.

Completed work:

- reorganized the project into a cleaner modular structure
- fixed dependency and virtual environment related issues
- simplified the user workflow with launcher commands such as `image`, `video`, `webcam`, and `external-camera`
- improved display scaling so images no longer open too large
- validated the CVAT dataset and detected missing labels
- added stronger error handling and structured logging
- trained a custom carton model on the available labeled data
- added configuration support through `configs/defaults.yaml`
- added subcommands for `predict`, `inspect-dataset`, `prepare-dataset`, `train`, and `export`
- added dataset report generation and training summary generation
- added Raspberry Pi oriented export support
- fixed export behavior so missing ONNX-related dependencies fail early with a clear message
- updated the README to match the real working commands

## Quick Start

These commands were tested and are the fastest way to run inference with the trained custom model:

```bash
.venv/bin/image --model-path runs/train/carton_detector_gpu/weights/best.pt
.venv/bin/webcam --model-path runs/train/carton_detector_gpu/weights/best.pt
```

If the virtual environment is activated correctly:

```bash
image --model-path runs/train/carton_detector_gpu/weights/best.pt
webcam --model-path runs/train/carton_detector_gpu/weights/best.pt
```

## Project Goal

The real goal is to detect your carton classes, not generic COCO classes like `person`, `chair`, or `tv`.

The current custom classes from the inspected CVAT export are:

- `Milch_Karton_shokolade`
- `Miclh_Karton_Vanille`
- `Teeschachtel`
- `Cube_Karton`

That means the intended workflow is:

1. inspect CVAT export
2. prepare clean dataset
3. train custom model
4. run inference
5. export edge artifacts
6. deploy to Raspberry Pi 5 with AI camera

## Architecture Overview

The project now includes the three high-impact upgrades that were planned:

1. Custom model training was stabilized.
The dataset is inspected before training, missing and malformed labels are reported, a cleaned split is created, and metadata is written for reproducibility.

2. Config + subcommand architecture was added.
The simple launcher commands still work, but the internal CLI now also supports scalable automation with subcommands.

3. Raspberry Pi export support was added.
The project can export a trained model, generate a labels file, and write an export manifest for deployment packaging.

## Current Folder Structure

```text
.
├── README.md
├── RECOMMENDATIONS.md
├── requirements.txt
├── requirements-export.txt
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

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
.venv/bin/python3 tools/install_shortcuts.py
```

Shortcut commands created in `.venv/bin`:

- `image`
- `video`
- `webcam`
- `external-camera`
- `external_camera`
- `inspect-dataset`
- `prepare-dataset`
- `train-carton`
- `export-edge`

## Configuration

Default project values are stored in:

```text
configs/defaults.yaml
```

This file controls:

- default model path
- dataset paths
- image and video directories
- training defaults
- export defaults
- logging defaults
- camera defaults

## Dataset Status

Dataset root:

```text
data/cvat_exports/caton_hause/
```

Findings from the current inspected CVAT export:

- `train.txt` references `74` images
- only `65` label files exist
- `9` images are missing labels
- training uses the labeled subset automatically
- label file structure and coordinate ranges are validated before training

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

## Simple Daily Commands

### Image Playback Inference

```bash
.venv/bin/image --model-path runs/train/carton_detector_gpu/weights/best.pt
```

Behavior:

- reads images from `data/images/` by default
- advances automatically through images
- keeps each image on screen for `2` seconds by default
- stops on `q` or `Esc`

### Video Inference

```bash
.venv/bin/video --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### Webcam Inference

```bash
.venv/bin/webcam --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### External Camera

```bash
.venv/bin/external-camera --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### RTSP Stream

```bash
.venv/bin/webcam \
    --camera-index rtsp://username:password@camera-ip:554/stream \
    --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### Display Scaling

```bash
.venv/bin/image \
    --model-path runs/train/carton_detector_gpu/weights/best.pt \
    --display-max-width 900 \
    --display-max-height 700
```

## Subcommand Workflow

The same project now also supports a more structured CLI.

### Predict

```bash
.venv/bin/python3 main.py predict image --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### Inspect Dataset

```bash
.venv/bin/inspect-dataset
```

### Prepare Dataset

```bash
.venv/bin/prepare-dataset --overwrite
```

### Train

```bash
.venv/bin/train-carton --overwrite
```

### Export

```bash
.venv/bin/export-edge --source-model runs/train/carton_detector_gpu/weights/best.pt
```

## Training

Manual training command:

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

Training improvements currently included:

- automatic GPU usage when CUDA is available
- dataset inspection before training
- cleaned train/validation split creation
- reproducible split metadata
- dataset report generation
- training summary generation
- deterministic settings for more reproducible runs

Important outputs:

- `data/processed/caton_hause/data.yaml`
- `data/processed/caton_hause/dataset_report.yaml`
- `runs/train/carton_detector_gpu/weights/best.pt`
- `runs/train/carton_detector_gpu/training_summary.yaml`

## Export For Raspberry Pi

Install export dependencies first:

```bash
.venv/bin/pip install -r requirements-export.txt
```

Then export:

```bash
.venv/bin/export-edge \
    --source-model runs/train/carton_detector_gpu/weights/best.pt
```

Default export targets currently configured:

- `onnx`
- `openvino`
- `tflite`

Outputs are written under:

```text
runs/export/
```

Generated deployment files include:

- exported model artifacts
- `labels.txt`
- `edge_export_manifest.yaml`

## Logging

Application log file:

```text
logs/application.log
```

Logging currently includes:

- model loading
- dataset inspection summaries
- training summaries
- per-frame detection summaries
- class counts
- confidence averages
- export actions
- dependency problems

Example detection log:

```text
2026-03-23 22:50:10,487 | INFO | yolo_edge.cli | source=emrah_carton_hause_1.jpeg frame=0 detected=yes total=1 classes={'Miclh_Karton_Vanille': 1} average_confidence=0.9787
```

## Raspberry Pi 5 Direction

This project is already prepared for a Raspberry Pi 5 deployment path.

Recommended deployment flow:

1. train on the stronger development machine
2. export edge artifacts
3. copy model, labels, and export manifest to Raspberry Pi
4. test with the real AI camera stream
5. tune image size, latency, and thermal behavior on-device

For the next deployment stage, read:

[RECOMMENDATIONS.md](/home/emrahtek/Schreibtisch/CodeLab/YOLO_Nachhaltigkeit/RECOMMENDATIONS.md)

## Tests

```bash
.venv/bin/python3 -m unittest discover -s tests -v
```
