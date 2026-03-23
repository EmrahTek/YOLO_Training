# YOLO Edge Detection Pipeline

This project is a modular and production-oriented YOLO object detection codebase designed for lightweight execution, clean maintenance, and future Raspberry Pi 5 deployment. The runtime is organized around a standard Python package, clearer local asset folders, structured logging, and basic automated tests.

## Current Project Structure

```text
.
├── README.md
├── requirements.txt
├── main.py
├── auto_git_manager.py
├── yolo_edge/
│   ├── __init__.py
│   ├── cli.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── detector.py
│   │   └── video_streamer.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_manager.py
│   └── utils/
│       ├── __init__.py
│       └── logging_utils.py
├── tools/
│   └── auto_git_manager.py
├── tests/
│   ├── test_cli.py
│   └── test_dataset_manager.py
├── data/
│   ├── cvat_exports/
│   │   └── caton_hause/
│   └── images/
└── models/
    └── yolov8n.pt
```

`main.py` is the only root application entry point. The detection, streaming, dataset, and logging code now lives under `yolo_edge/`, which makes the codebase easier to maintain and test.

## Dependency Note

`numpy` is now declared explicitly in `requirements.txt`. Even though Ultralytics and OpenCV typically pull it in transitively, it should be pinned directly because the detector and streamer import it at runtime.

Install dependencies with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Review

The original `Caton_Hause.zip` archive was removed because it is no longer needed in the working tree. The extracted dataset is now kept under:

```text
data/cvat_exports/caton_hause/
```

Important findings from the dataset inspection:

- The export contains `data.yaml`, `train.txt`, and `labels/train/*.txt`.
- `train.txt` references 74 images.
- Only 65 label files exist.
- 9 images are missing label files.

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

The dataset manager intentionally fails normalization by default when labels are missing, because silent data corruption is dangerous for training pipelines.

## Inference Commands

### Image

```bash
python3 main.py \
    --source image \
    --path data/images/emrah_carton_hause_1.jpeg \
    --model-path models/yolov8n.pt \
    --show
```

### Video

```bash
python3 main.py \
    --source video \
    --path path/to/video.mp4 \
    --model-path models/yolov8n.pt \
    --save-output
```

### Default Webcam

```bash
python3 main.py \
    --source webcam \
    --camera-index 0 \
    --model-path models/yolov8n.pt
```

### External USB Camera

```bash
python3 main.py \
    --source webcam \
    --camera-index 1 \
    --model-path models/yolov8n.pt
```

### RTSP Stream

```bash
python3 main.py \
    --source webcam \
    --camera-index rtsp://username:password@camera-ip:554/stream \
    --model-path models/yolov8n.pt
```

## Logging Behavior

The application now logs per-frame detection summaries. Each processed frame reports:

- whether any object was detected
- total detection count
- per-class counts
- average confidence

Example log output:

```text
2026-03-23 12:00:00,000 | INFO | yolo_edge.core.detector | Loaded YOLO model from /path/to/models/yolov8n.pt
2026-03-23 12:00:01,200 | INFO | yolo_edge.cli | source=emrah_carton_hause_1.jpeg frame=0 detected=yes total=2 classes={'Cube_Karton': 1, 'Teeschachtel': 1} average_confidence=0.8421
```

If no object is detected, the log shows:

```text
source=emrah_carton_hause_1.jpeg frame=0 detected=no total=0 classes={} average_confidence=0.0000
```

Logs are written to the console and optionally to:

```text
logs/application.log
```

## Dataset Manager Usage

Example inspection:

```python
from pathlib import Path

from yolo_edge.data.dataset_manager import DatasetManager

manager = DatasetManager()
description = manager.inspect_dataset_directory(Path("data/cvat_exports/caton_hause"))

print(description.image_count)
print(description.label_count)
print(description.missing_labels)
```

Example normalization:

```python
from pathlib import Path

from yolo_edge.data.dataset_manager import DatasetManager

manager = DatasetManager()
manager.prepare_cvat_export(
    dataset_root=Path("data/cvat_exports/caton_hause"),
    normalized_dataset_directory=Path("data/processed/caton_hause"),
    image_source_directory=Path("data/images"),
    overwrite=True,
)
```

## Testing

Run the tests with:

```bash
python3 -m unittest discover -s tests -v
```

## Git Automation

The Git watcher no longer depends on `watchdog`. It now uses simple polling from the standard library, which is easier to run in minimal environments.

Start it with:

```bash
python3 auto_git_manager.py --watch-dir .
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

- Start with `models/yolov8n.pt` or another nano-scale model.
- Keep `--image-size` modest.
- Use `--device cpu` unless you have a validated accelerator path.
- Avoid `--show` for headless deployments.
- Save outputs only when needed to reduce storage writes.
- Use active cooling for sustained inference workloads.
