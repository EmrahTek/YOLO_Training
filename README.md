# YOLO Carton Detection Project

This repository is a practical computer vision project for custom carton detection with YOLO. It is designed for a clean local workflow today and an edge-device deployment path later, including Raspberry Pi style usage.

The project covers four main jobs:

1. Inspect and validate the CVAT YOLO export.
2. Prepare a clean train/validation dataset from the labeled images.
3. Train a custom YOLO model for your carton classes.
4. Run inference in a simple way on images, videos, webcams, or an external camera.

## Project Goal

The goal is not to use the generic COCO YOLO model forever. The generic model only knows common classes like `person`, `chair`, or `tv`. Your real target classes come from the CVAT export:

- `Milch_Karton_shokolade`
- `Miclh_Karton_Vanille`
- `Teeschachtel`
- `Cube_Karton`

Because of that, the correct workflow is:

1. Validate the CVAT export.
2. Build a clean training dataset.
3. Train a custom model.
4. Run inference with the trained `best.pt`.

## Current Folder Structure

```text
.
├── README.md
├── requirements.txt
├── main.py
├── train.py
├── auto_git_manager.py
├── yolo_edge/
│   ├── cli.py
│   ├── training.py
│   ├── core/
│   │   ├── detector.py
│   │   └── video_streamer.py
│   ├── data/
│   │   └── dataset_manager.py
│   └── utils/
│       └── logging_utils.py
├── tools/
│   └── auto_git_manager.py
├── tests/
│   ├── test_cli.py
│   └── test_dataset_manager.py
├── data/
│   ├── cvat_exports/
│   │   └── caton_hause/
│   ├── images/
│   └── processed/
├── models/
│   └── yolov8n.pt
├── runs/
│   └── train/
└── logs/
```

## Dataset Status

The CVAT export was inspected carefully.

Dataset root:

```text
data/cvat_exports/caton_hause/
```

Important findings:

- `train.txt` references `74` images.
- Only `65` label files exist.
- `9` images are missing labels.
- Training should use only the labeled subset unless you manually complete the annotations.

Missing label stems:

- `emrah_carton_hause_19`
- `emrah_carton_hause_40`
- `emrah_carton_hause_46`
- `emrah_carton_hause_51`
- `emrah_carton_hause_66`
- `emrah_carton_hause_67`
- `emrah_carton_hause_68`
- `emrah_carton_hause_69`
- `emrah_carton_hause_71`

The training preparation code automatically builds a clean dataset from the `65` labeled images.

## Installation

Create a fresh virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
.venv/bin/python3 tools/install_shortcuts.py
```

This installs the shortcut commands into `.venv/bin`.

Tested and confirmed working on this project:

```bash
.venv/bin/image --model-path runs/train/carton_detector_gpu/weights/best.pt
.venv/bin/webcam --model-path runs/train/carton_detector_gpu/weights/best.pt
```

If your current `.venv` activation is broken, use the interpreter directly:

```bash
.venv/bin/python3
```

If activation works correctly, these commands become available directly:

```bash
image
video
webcam
external-camera
```

You can also run them without activating the environment:

```bash
.venv/bin/image
.venv/bin/video
.venv/bin/webcam
.venv/bin/external-camera
```

## Dependencies

Main runtime dependencies:

- `numpy`
- `ultralytics`
- `opencv-python`
- `PyYAML`

## Training The Custom Carton Model

This is the most important step. Without it, you will not get carton class detections.

Start training:

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

Important notes:

- GPU is now selected automatically when CUDA is available.
- On your machine, the NVIDIA RTX 2000 Ada GPU is available and should be used.
- The prepared training dataset is created automatically before training starts.

Expected trained model output:

```text
runs/train/carton_detector/weights/best.pt
```

If you create a new run name, the folder name changes accordingly.

If you trained with the GPU run name used in this repository, your actual model may be:

```text
runs/train/carton_detector_gpu/weights/best.pt
```

## Simplified Inference Usage

The CLI is now intentionally simple.

You can start modes with only one word:

- `image`
- `video`
- `webcam`
- `external-camera`

Because shell commands cannot contain spaces, use `external-camera` or `external_camera`.

### Image Mode

```bash
.venv/bin/image --model-path runs/train/carton_detector_gpu/weights/best.pt
```

Behavior:

- It reads images from `data/images/` by default.
- It shows them one after another automatically.
- Each image stays on screen for `2` seconds by default.
- Press `q` or `Esc` to stop playback.

If you want a different delay:

```bash
.venv/bin/image \
    --model-path runs/train/carton_detector_gpu/weights/best.pt \
    --image-delay-ms 2000
```

If you want to open one specific image only:

```bash
.venv/bin/image \
    --path data/images/emrah_carton_hause_1.jpeg \
    --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### Video Mode

If `data/videos/` contains exactly one video, this is enough:

```bash
.venv/bin/video --model-path runs/train/carton_detector_gpu/weights/best.pt
```

Or specify the file:

```bash
.venv/bin/video \
    --path data/videos/example.mp4 \
    --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### Default Webcam

```bash
.venv/bin/webcam --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### External Camera

```bash
.venv/bin/external-camera --model-path runs/train/carton_detector_gpu/weights/best.pt
```

This uses camera index `1` by default.

If your external camera is on another index:

```bash
.venv/bin/external-camera \
    --camera-index 2 \
    --model-path runs/train/carton_detector_gpu/weights/best.pt
```

### RTSP Camera

Use webcam mode with a URL:

```bash
.venv/bin/webcam \
    --camera-index rtsp://username:password@camera-ip:554/stream \
    --model-path runs/train/carton_detector_gpu/weights/best.pt
```

## Display Scaling

Large images are automatically scaled for display.

Default display limits:

- max width: `1280`
- max height: `720`

You can change them:

```bash
.venv/bin/image \
    --model-path runs/train/carton_detector_gpu/weights/best.pt \
    --display-max-width 900 \
    --display-max-height 700
```

If you do not want a window:

```bash
.venv/bin/image \
    --model-path runs/train/carton_detector_gpu/weights/best.pt \
    --no-show
```

## Logging

Logs are written to:

```text
logs/application.log
```

Each processed frame logs:

- source name
- frame index
- whether an object was detected
- total detection count
- class-wise counts
- average confidence

Example:

```text
2026-03-23 21:47:40,689 | INFO | yolo_edge.cli | source=emrah_carton_hause_1.jpeg frame=0 detected=yes total=6 classes={'chair': 2, 'person': 2, 'potted plant': 1, 'tv': 1} average_confidence=0.5268
```

If you see COCO classes like `person`, `chair`, or `tv`, that means you are still using a generic model such as `models/yolov8n.pt`, not your trained carton model.

## Why The Generic Model Looked Wrong

This point is critical:

- `models/yolov8n.pt` is a generic pretrained YOLO model.
- It is not trained on your carton classes.
- Therefore it will not reliably output `Milch_Karton_shokolade`, `Cube_Karton`, and the other custom classes.

To get carton predictions, use the trained file:

```text
runs/train/carton_detector_gpu/weights/best.pt
```

or the latest GPU-based run you created.

## Dataset Utilities

The dataset utilities can:

- inspect the CVAT export
- validate image/label consistency
- create a clean training dataset
- prepare YOLO-style train/val folders

Programmatic example:

```python
from pathlib import Path

from yolo_edge.data.dataset_manager import DatasetManager

manager = DatasetManager()
description = manager.inspect_dataset_directory(Path("data/cvat_exports/caton_hause"))
print(description.image_count)
print(description.label_count)
print(description.missing_labels)
```

Create a clean train/val dataset:

```python
from pathlib import Path

from yolo_edge.data.dataset_manager import DatasetManager

manager = DatasetManager()
manager.create_training_dataset(
    dataset_root=Path("data/cvat_exports/caton_hause"),
    image_source_directory=Path("data/images"),
    output_directory=Path("data/processed/caton_hause"),
    validation_ratio=0.2,
    overwrite=True,
)
```

## Testing

Run the unit tests:

```bash
python3 -m unittest discover -s tests -v
```

## Git Automation

Automatic Git synchronization is available through:

```bash
python3 auto_git_manager.py --watch-dir .
```

It performs:

```text
git add
git commit
git push
```

using a simple polling-based watcher.

## Recommended Next Step

If your GPU training finishes successfully, the next correct validation command is:

```bash
.venv/bin/image \
    --model-path runs/train/carton_detector_gpu/weights/best.pt
```

If that run name differs, use the actual `best.pt` produced by training.

## Raspberry Pi 5 Notes

For Raspberry Pi deployment later:

- prefer a smaller model
- reduce image size if necessary
- use `--no-show` in headless mode
- save outputs only when needed
- benchmark inference speed on the final hardware

The current project structure is already prepared for that deployment path.
