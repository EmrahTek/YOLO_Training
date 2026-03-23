# YOLO Edge Detection Pipeline

This repository contains a modular, production-oriented Python object detection pipeline built around Ultralytics YOLO. The project is designed to stay lightweight, readable, and easy to deploy on constrained hardware such as a Raspberry Pi 5.

The codebase focuses on three practical goals:

1. Clean, maintainable architecture with small single-responsibility modules.
2. Hardware-agnostic inference for images, videos, USB cameras, and RTSP streams.
3. Safe dataset preparation utilities for CVAT YOLO exports packed as zip files.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── auto_git_manager.py
├── dataset_manager.py
├── detector.py
├── main.py
└── video_streamer.py
```

## Prerequisites

- Python 3.10 or newer
- `git`
- A valid Git remote configured for push operations
- A YOLO model checkpoint such as `yolov8n.pt`
- Optional: webcam, USB camera, or RTSP camera stream

## Virtual Environment Setup

Create and activate a dedicated virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip` before installing dependencies:

```bash
python -m pip install --upgrade pip
```

## Install Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Dataset Extraction From CVAT ZIP

The repository includes a dedicated `dataset_manager.py` module that can:

- Extract a CVAT YOLO zip archive.
- Validate that the exported structure contains the expected image and label folders.
- Discover the dataset YAML file if one exists.

Typical workflow:

1. Place the CVAT export zip file in a local data directory, for example `Data/Caton_Hause.zip`.
2. Use the dataset manager from Python or integrate it into future training scripts.
3. Point the detector or training workflow to the validated dataset root.

Example usage from Python:

```python
from pathlib import Path

from dataset_manager import DatasetManager

manager = DatasetManager()
extracted_path = manager.extract_cvat_zip(
    zip_path=Path("Data/Caton_Hause.zip"),
    output_directory=Path("data/dataset")
)

dataset_info = manager.validate_yolo_dataset(extracted_path)
print(dataset_info)
```

## Running The Application

The CLI entry point is `main.py`. The `--source` argument accepts exactly three choices:

- `image`
- `video`
- `webcam`

### Image Inference

```bash
python main.py \
    --source image \
    --path Data/images/example.jpg \
    --model-path yolov8n.pt
```

### Video Inference

```bash
python main.py \
    --source video \
    --path Data/videos/example.mp4 \
    --model-path yolov8n.pt
```

### Default Webcam Inference

```bash
python main.py \
    --source webcam \
    --camera-index 0 \
    --model-path yolov8n.pt
```

### External USB Webcam Inference

```bash
python main.py \
    --source webcam \
    --camera-index 1 \
    --model-path yolov8n.pt
```

### RTSP Stream Inference

```bash
python main.py \
    --source webcam \
    --camera-index rtsp://username:password@camera-ip-address:554/stream \
    --model-path yolov8n.pt
```

## CLI Arguments Overview

Core arguments:

- `--source`: Required. One of `image`, `video`, or `webcam`.
- `--path`: Required for `image` and `video`.
- `--camera-index`: Required for `webcam` logic, defaults to `0`. Can be an integer camera index or an RTSP URL.
- `--model-path`: Path to the YOLO model checkpoint.
- `--confidence`: Confidence threshold for detections.
- `--device`: Optional inference device override such as `cpu`.
- `--save-output`: Save annotated frames or videos to disk.
- `--output-dir`: Directory used for saved outputs.
- `--show`: Display annotated output in an OpenCV window.

## Git Automation

This repository includes `auto_git_manager.py` to monitor file changes and automatically run:

```bash
git add <file>
git commit -m "feat: add/update <file>"
git push
```

Start it in the background after the initial setup:

```bash
python auto_git_manager.py --watch-dir .
```

The watcher is useful because it keeps your strict version control workflow active while you continue iterating on the project.

## Raspberry Pi 5 Deployment

The codebase is intentionally structured for edge deployment. For Raspberry Pi 5, follow this sequence:

1. Install system packages required by OpenCV and Python virtual environments.
2. Create a fresh virtual environment on the Pi.
3. Install the Python dependencies from `requirements.txt`.
4. Use a lightweight model checkpoint such as `yolov8n.pt`.
5. Prefer CPU inference first, then benchmark smaller image sizes before increasing resolution.

Recommended setup flow:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libatlas-base-dev libopenjp2-7 libtiff6
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Practical Raspberry Pi 5 recommendations:

- Use the smallest model that still satisfies accuracy needs.
- Lower the frame size for real-time streams.
- Disable on-screen display in headless deployments.
- Save output only when necessary to reduce I/O overhead.
- Use a dedicated power supply and active cooling during long inference sessions.
- Prefer USB cameras with stable Linux support when possible.

For headless execution on the Pi:

```bash
python main.py \
    --source webcam \
    --camera-index 0 \
    --model-path yolov8n.pt \
    --device cpu
```

## Clean Code Principles Applied

The modules in this project are designed to keep responsibilities separate:

- `main.py` handles CLI parsing and orchestration.
- `detector.py` encapsulates YOLO model loading and inference.
- `video_streamer.py` handles media input and streaming concerns.
- `dataset_manager.py` handles dataset extraction and validation.
- `auto_git_manager.py` handles automatic version control actions.

This separation keeps the code easier to test, extend, and deploy.

## Future Extensions

The current architecture is ready for the next stage of evolution, including:

- batch inference jobs
- training entry points
- model export to ONNX or TensorRT
- structured logging
- unit and integration tests
- configuration files for deployment profiles
