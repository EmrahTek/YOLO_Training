# YOLO Edge Detection Pipeline

This project is a modular and production-oriented YOLO object detection codebase designed for lightweight execution, clean maintenance, and future Raspberry Pi 5 deployment. The architecture now keeps the main implementation directly in the root Python files, adds structured logging, and includes automated tests.

## Dataset Analysis Of `Caton_Hause.zip`

The provided CVAT export was inspected directly before the refactor. It is not a complete self-contained YOLO training dataset in the most common folder-based form.

Observed structure inside the zip:

- `data.yaml`
- `train.txt`
- `labels/train/*.txt`

Important findings:

- `train.txt` references 74 images.
- The zip contains 65 label files.
- 9 images are missing label files.
- The image paths in `train.txt` point to `data/images/train/...`.
- The local images in this repository are currently stored in `Data/images/...`.

Missing label stems detected in the export:

- `emrah_carton_hause_19`
- `emrah_carton_hause_40`
- `emrah_carton_hause_46`
- `emrah_carton_hause_51`
- `emrah_carton_hause_66`
- `emrah_carton_hause_67`
- `emrah_carton_hause_68`
- `emrah_carton_hause_69`
- `emrah_carton_hause_71`

This is exactly why the dataset manager was rewritten. A production pipeline should fail loudly when annotations are incomplete instead of silently proceeding with corrupted supervision.

## Project Structure

```text
.
├── README.md
├── requirements.txt
├── main.py
├── detector.py
├── video_streamer.py
├── dataset_manager.py
├── auto_git_manager.py
├── logging_utils.py
└── tests/
    ├── test_cli.py
    └── test_dataset_manager.py
```

All root-level Python files are now the actual implementation files, which keeps the project easier to inspect in an IDE and avoids wrapper-related confusion.

## Prerequisites

- Python 3.10 or newer
- `git`
- A configured Git remote
- A YOLO checkpoint such as `yolov8n.pt`
- Optional image, video, webcam, or RTSP source

## Virtual Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Inference Commands

### Image Inference

```bash
python3 main.py \
    --source image \
    --path Data/images/emrah_carton_hause_1.jpeg \
    --model-path yolov8n.pt \
    --show
```

### Video Inference

```bash
python3 main.py \
    --source video \
    --path path/to/video.mp4 \
    --model-path yolov8n.pt \
    --save-output
```

### Default Webcam

```bash
python3 main.py \
    --source webcam \
    --camera-index 0 \
    --model-path yolov8n.pt
```

### External USB Camera

```bash
python3 main.py \
    --source webcam \
    --camera-index 1 \
    --model-path yolov8n.pt
```

### RTSP Stream

```bash
python3 main.py \
    --source webcam \
    --camera-index rtsp://username:password@camera-ip:554/stream \
    --model-path yolov8n.pt
```

Useful optional flags:

- `--confidence 0.25`
- `--device cpu`
- `--image-size 640`
- `--show`
- `--save-output`
- `--output-dir outputs`
- `--log-level INFO`
- `--log-dir logs`

## Dataset Handling

The rewritten dataset manager supports two real-world cases:

1. CVAT YOLO exports that contain `train.txt`, `data.yaml`, and labels.
2. Canonical YOLO folder datasets that contain image and label directories.

Example dataset inspection:

```python
from pathlib import Path

from dataset_manager import DatasetManager

manager = DatasetManager()
description = manager.inspect_cvat_zip(Path("Data/Caton_Hause.zip"))

print(description.image_count)
print(description.label_count)
print(description.missing_labels)
```

Example normalization attempt using local images:

```python
from pathlib import Path

from dataset_manager import DatasetManager

manager = DatasetManager()
manager.prepare_cvat_export(
    zip_path=Path("Data/Caton_Hause.zip"),
    extraction_directory=Path("data/interim/caton_hause"),
    normalized_dataset_directory=Path("data/processed/caton_hause"),
    image_source_directory=Path("Data/images"),
    overwrite=True,
)
```

By default the normalization step fails when labels are missing. This is intentional because silent dataset corruption is more expensive than a loud validation error.

## Testing

Run the tests with:

```bash
python3 -m unittest discover -s tests -v
```

The tests cover:

- CLI validation rules
- CVAT zip inspection behavior
- failure on incomplete annotations during normalization

## Logging

The project includes centralized logging with console output and rotating log files.

Runtime logs can be written to:

```text
logs/application.log
```

This is especially helpful for edge deployments where you may need post-run diagnostics instead of live terminal monitoring.

## Git Automation

Run the Git watcher in the background if you want continuous automatic commits and pushes:

```bash
python3 auto_git_manager.py --watch-dir .
```

It performs:

```bash
git add <file>
git commit -m "feat: add/update <file>"
git push
```

## Raspberry Pi 5 Deployment

For Raspberry Pi 5, keep the runtime conservative and CPU-focused first:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip libatlas-base-dev libopenjp2-7 libtiff6
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

Recommended deployment practices:

- Start with `yolov8n.pt` or another nano-scale model.
- Keep `--image-size` modest.
- Use `--device cpu` unless you have a validated accelerator path.
- Avoid `--show` for headless deployments.
- Save outputs only when needed to reduce storage writes.
- Use active cooling for sustained inference workloads.

Example headless command on Raspberry Pi 5:

```bash
python3 main.py \
    --source webcam \
    --camera-index 0 \
    --model-path yolov8n.pt \
    --device cpu \
    --image-size 416
```

## Design Notes

Key architectural decisions in this final version:

- Root-level implementation files to keep the project simple and IDE-friendly.
- Logging separated into a shared utility module.
- Dataset validation aligned with the actual CVAT export you provided.
- Tests added to prevent regressions in CLI and dataset logic.
