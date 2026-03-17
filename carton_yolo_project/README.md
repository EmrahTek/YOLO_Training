# Carton YOLO Project
**Single-class YOLO pipeline for `carton_pack` detection**

This repository contains a clean, modular, and extensible YOLO-based computer vision pipeline prepared specifically for the **`carton_pack`** task.

The project is designed for:

- a **single-class baseline** (`carton_pack`)
- a **CVAT-ready** annotation workflow
- **clean code** and modular Python structure
- **logging** and **error handling**
- easy usage from the **Ubuntu terminal**
- future portability to **Raspberry Pi 5** and edge/AI camera workflows
- separate command entry points for **dataset preparation**, **training**, **validation**, **image inference**, **video inference**, **webcam inference**, and **model export**

---

## Table of Contents

1. [Project Goal](#1-project-goal)
2. [Why This Repository Exists](#2-why-this-repository-exists)
3. [Target Use Case](#3-target-use-case)
4. [Hardware and Environment](#4-hardware-and-environment)
5. [Project Structure](#5-project-structure)
6. [Core Workflow](#6-core-workflow)
7. [Installation](#7-installation)
8. [Command Setup](#8-command-setup)
9. [Dataset Input from CVAT](#9-dataset-input-from-cvat)
10. [Preparing the Dataset](#10-preparing-the-dataset)
11. [Training the Model](#11-training-the-model)
12. [Validating the Model](#12-validating-the-model)
13. [Running Inference on Images](#13-running-inference-on-images)
14. [Running Inference on Videos](#14-running-inference-on-videos)
15. [Running Inference on a Webcam](#15-running-inference-on-a-webcam)
16. [Exporting the Model](#16-exporting-the-model)
17. [Configuration Files](#17-configuration-files)
18. [Logging](#18-logging)
19. [Error Handling](#19-error-handling)
20. [Git Strategy](#20-git-strategy)
21. [Recommended Training Strategy](#21-recommended-training-strategy)
22. [Edge Deployment Notes](#22-edge-deployment-notes)
23. [Troubleshooting](#23-troubleshooting)
24. [Example End-to-End Session](#24-example-end-to-end-session)
25. [Recommended Next Steps](#25-recommended-next-steps)

---

## 1. Project Goal

The main goal of this repository is to build a reliable **first baseline** for detecting **carton packages** with a workflow that is:

- easy to understand
- easy to maintain
- easy to extend later into a multi-class team project
- suitable for later edge deployment experiments

At this stage, the repository is intentionally focused on **one class only**:

```text
carton_pack
```

Starting with a single-class detector is a practical engineering choice. It reduces complexity, simplifies dataset debugging, and helps establish a stable training and inference pipeline before moving to a larger shared model.

---

## 2. Why This Repository Exists

This project exists to solve a very specific practical need:

- annotated data already comes from **CVAT**
- training should be easy to repeat
- dataset preparation should be reproducible
- errors should be visible in logs
- commands should be simple enough to run from the terminal
- the code should remain readable for future maintenance

This is not just a quick notebook experiment. It is structured as a small but serious software project.

---

## 3. Target Use Case

The repository is built for the following scenario:

1. collect carton images
2. annotate them in CVAT
3. export YOLO-style labels
4. prepare train/validation folders
5. train a small YOLO baseline
6. validate the trained model
7. test on unseen images
8. test on video
9. test on webcam
10. later export the model for edge deployment

This repository is especially useful when the goal is not only “getting something to run once”, but creating a **repeatable project structure**.

---

## 4. Hardware and Environment

This project was designed with the following environment in mind:

- **Operating System:** Ubuntu
- **Python:** 3.12.x
- **GPU:** NVIDIA RTX 2000 Ada Generation Laptop GPU
- **CUDA:** available
- **PyTorch:** CUDA-enabled
- **Ultralytics:** YOLO v8 / YOLO11 compatible environment

A lightweight model is used by default because this improves flexibility for later deployment on limited hardware.

Default starter model:

```text
yolo11n.pt
```

This is a good choice because it is:

- lightweight
- fast to train
- suitable for first experiments
- easier to export
- more realistic for later Raspberry Pi / edge workflows than a large model

---

## 5. Project Structure

```text
carton_yolo_project/
├── README.md
├── requirements.txt
├── .gitignore
├── bin/
│   ├── prepare_dataset
│   ├── train
│   ├── validate
│   ├── image
│   ├── video
│   ├── camera
│   └── export_model
├── configs/
│   ├── default.yaml
│   └── logging.yaml
├── data/
│   ├── external/
│   │   └── cvat_export/
│   │       ├── README.md
│   │       ├── images/
│   │       └── labels/
│   ├── dataset/
│   │   ├── README.md
│   │   ├── data_carton.yaml
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   ├── demo/
│   │   ├── README.md
│   │   ├── unseen_images/
│   │   └── unseen_labels/
│   └── raw/
│       └── README.md
├── docs/
│   └── RESOURCES.md
├── logs/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
├── runs/
│   └── .gitkeep
└── src/
    └── carton_yolo/
        ├── __init__.py
        ├── main.py
        ├── config.py
        ├── constants.py
        ├── dataset.py
        ├── exceptions.py
        ├── model_export.py
        ├── predictor.py
        ├── trainer.py
        ├── validator.py
        └── utils/
            ├── __init__.py
            ├── io_utils.py
            ├── logging_utils.py
            └── paths.py
```

### Folder meaning

#### `bin/`
Shell entry points for daily usage.

#### `configs/`
Configuration files for runtime and logging behavior.

#### `data/external/cvat_export/`
Raw annotation export coming from CVAT. This is the starting point for the training dataset.

#### `data/dataset/`
Prepared YOLO training structure generated from the CVAT export.

#### `data/demo/`
Unseen test data used for qualitative checks after training.

#### `runs/`
Training outputs, YOLO experiment folders, weights, metrics, and plots.

#### `logs/`
Application logs for debugging and traceability.

#### `src/carton_yolo/`
Python source code for the project.

---

## 6. Core Workflow

The intended workflow is:

```text
CVAT export
   ↓
prepare_dataset
   ↓
train
   ↓
validate
   ↓
image / video / camera
   ↓
export_model
```

This sequence keeps the project organized and reproducible.

---

## 7. Installation

### 7.1 Activate the virtual environment

```bash
source .venv/bin/activate
```

Check that Python is coming from the virtual environment:

```bash
which python
python --version
```

### 7.2 Install dependencies

```bash
pip install -r requirements.txt
```

Check installed packages:

```bash
pip list | grep -E "ultralytics|torch|opencv|PyYAML"
```

Optional CUDA check:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"
```

---

## 8. Command Setup

The repository uses executable helper scripts inside `bin/`.

### 8.1 Enable them for the current terminal session

From inside the project directory:

```bash
chmod +x bin/*
export PATH="$PWD/bin:$PATH"
```

Now you can call commands directly:

```bash
prepare_dataset
train
validate
image
video
camera
export_model
```

### 8.2 Make the commands permanent

Add the project `bin/` directory to your shell startup file:

```bash
echo 'export PATH="/full/project/path/carton_yolo_project/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Check that the commands are available:

```bash
which train
which image
which camera
```

---

## 9. Dataset Input from CVAT

This project assumes that annotation has already been done in **CVAT** and that you have a YOLO-compatible export.

Place the exported files here:

```text
data/external/cvat_export/images/
data/external/cvat_export/labels/
```

### Expected content

#### Images
All source images should go into:

```text
data/external/cvat_export/images/
```

#### Labels
Matching YOLO `.txt` label files should go into:

```text
data/external/cvat_export/labels/
```

### Expected naming rule

Each image must have a matching label file with the same base name.

Example:

```text
images/carton_001.jpg
labels/carton_001.txt

images/carton_002.png
labels/carton_002.txt
```

If an image exists without a label, or a label exists without an image, dataset preparation should fail clearly.

---

## 10. Preparing the Dataset

Once the CVAT export is in place, prepare the dataset:

```bash
prepare_dataset
```

### What this command does

It should:

- scan the external CVAT export folder
- verify image-label matching
- validate input structure
- split the dataset into training and validation sets
- generate the YOLO dataset YAML file
- log what happened
- stop with a clear error if something is wrong

### Output structure after preparation

The command populates:

```text
data/dataset/images/train/
data/dataset/images/val/
data/dataset/labels/train/
data/dataset/labels/val/
data/dataset/data_carton.yaml
```

### Basic verification after preparation

Check that the folders are populated:

```bash
find data/dataset/images/train -type f | wc -l
find data/dataset/images/val -type f | wc -l
find data/dataset/labels/train -type f | wc -l
find data/dataset/labels/val -type f | wc -l
```

Open the YAML file:

```bash
cat data/dataset/data_carton.yaml
```

The YAML should point to the correct train/val paths and list the class name.

---

## 11. Training the Model

Run training with:

```bash
train
```

This uses the default configuration from the project.

### What training does

Training should:

- load the dataset YAML
- load the selected base model
- train on the train split
- validate during training
- save metrics and plots
- save trained weights
- store outputs under `runs/`

### Typical outputs

You will usually find results in a folder like:

```text
runs/detect/train/
```

or a similar YOLO run directory.

### Check the trained weights

After training, inspect the generated files:

```bash
find runs -type f | grep -E "best.pt|last.pt|results.csv"
```

### Practical note

For a first baseline, focus on:

- whether training runs without crashing
- whether labels are read correctly
- whether loss decreases
- whether predictions make visual sense

Do not optimize too early. First confirm that the pipeline is correct.

---

## 12. Validating the Model

Run validation with:

```bash
validate
```

### Purpose of validation

Validation checks how the trained model performs on the validation split. It helps answer:

- is the model learning something useful?
- is the dataset quality acceptable?
- are the labels likely correct?
- are there obvious failure cases?

### What to inspect

Pay attention to:

- precision
- recall
- mAP
- confusion behavior
- visual predictions on validation images

### Good engineering habit

Always validate after training. Do not rely only on training loss.

---

## 13. Running Inference on Images

Run image-folder inference with:

```bash
image
```

By default, this command reads from:

```text
data/demo/unseen_images/
```

This folder is intended for **unseen qualitative test images**.

### Use a custom image folder

```bash
image --source /home/emrah/test_images
```

### Recommended test folder content

```text
/home/emrah/test_images/
├── img_001.jpg
├── img_002.jpg
├── img_003.jpg
```

### Why image inference matters

Image inference is the fastest way to check:

- whether the model loads correctly
- whether predictions look reasonable
- whether bounding boxes align with the carton object
- whether false positives are obvious

---

## 14. Running Inference on Videos

Run video inference with:

```bash
video --source /home/emrah/test_videos/carton.mp4
```

### Good use cases for video testing

Use video testing to understand:

- temporal stability
- repeated detection consistency
- motion blur robustness
- lighting change sensitivity
- practical demo behavior

### Example

```bash
mkdir -p /home/emrah/test_videos
video --source /home/emrah/test_videos/carton_demo.mp4
```

Video inference is often more informative than still images because it reveals instability that does not appear in a single frame.

---

## 15. Running Inference on a Webcam

Run webcam inference with:

```bash
camera
```

This is the most realistic test for a live demo scenario.

### What webcam testing helps you evaluate

- real-time usability
- camera focus sensitivity
- lighting behavior
- detection stability
- frame-rate limitations
- practical distance and angle constraints

### Before testing

Make sure:

- your webcam is connected
- no other application is locking the camera
- lighting is acceptable
- the carton object appears at realistic distances

### Linux camera check

Check whether a camera is visible:

```bash
ls /dev/video*
```

---

## 16. Exporting the Model

Export the trained model with:

```bash
export_model
```

### Purpose of export

Exporting allows later deployment in environments where a `.pt` training checkpoint is not the desired runtime format.

Common deployment-oriented formats include:

- ONNX
- TorchScript
- engine-specific optimized formats

### Check exported artifacts

```bash
find outputs -type f
find runs -type f | grep -E "onnx|engine|torchscript|tflite"
```

The exact output location depends on the implementation and configuration.

---

## 17. Configuration Files

### `configs/default.yaml`

This file contains the main runtime configuration. It typically includes values related to:

- dataset paths
- model path
- training hyperparameters
- output directories
- inference defaults
- export settings

Inspect it with:

```bash
cat configs/default.yaml
```

### `configs/logging.yaml`

This file controls logging behavior.

Inspect it with:

```bash
cat configs/logging.yaml
```

Typical logging configuration includes:

- log level
- output file
- console formatting
- timestamp formatting
- logger names

---

## 18. Logging

Logging is included to make the pipeline easier to debug and maintain.

### Why logging matters

Logs help answer:

- what command ran?
- which folder was read?
- which file caused an error?
- what model was used?
- where were outputs written?

### Typical log use

After a command runs, inspect logs:

```bash
ls logs
tail -n 50 logs/*.log
```

Use logs whenever:

- dataset preparation fails
- training stops unexpectedly
- a path is wrong
- outputs are not where expected
- a model file cannot be found

---

## 19. Error Handling

The codebase includes explicit error handling so that failures are easier to understand.

Examples of expected checked conditions:

- missing image folder
- missing label folder
- mismatched image/label names
- missing model file
- invalid config path
- empty dataset
- invalid export target
- missing webcam device

This is important because silent failures waste a lot of time in ML workflows.

---

## 20. Git Strategy

This repository intentionally avoids storing large generated artifacts in Git.

Usually ignored:

- datasets
- CVAT exports
- training outputs
- large model weights
- exported deployment artifacts
- videos
- archives

### Why this matters

YOLO projects often become difficult to maintain when Git is polluted with:

- large `.pt` files
- `.zip` archives
- training runs
- raw videos
- generated output images

Always check Git status before committing:

```bash
git status
```

Check for very large files:

```bash
find . -type f -size +50M
```

Useful Git commands:

```bash
git add .
git commit -m "feat: update carton YOLO pipeline"
git push
```

---

## 21. Recommended Training Strategy

Do not begin with aggressive optimization. A stable baseline is more valuable than a complicated unstable setup.

### Recommended order

#### Step 1
Use the prepared dataset and train the default small model.

```bash
train
```

#### Step 2
Validate and inspect qualitative results.

```bash
validate
image
camera
```

#### Step 3
Only after confirming the pipeline works, start tuning:

- image count
- annotation quality
- camera angles
- lighting diversity
- augmentation choices
- hyperparameters
- model size

### Engineering advice

Many YOLO problems are actually **dataset problems**, not model problems.

Before changing the model, verify:

- is the box annotation accurate?
- is the object visible enough?
- is the class definition consistent?
- are train and validation distributions reasonable?
- are difficult lighting conditions represented?

---

## 22. Edge Deployment Notes

This repository is designed with future edge deployment in mind.

### Why that matters now

Even if training is done on a strong laptop GPU, later deployment may happen on much more limited hardware such as:

- Raspberry Pi 5
- AI camera pipeline
- edge inference runtime

That is why this project starts with a small model and keeps the software structure modular.

### Recommended future path

1. train and validate the baseline on the laptop
2. export the trained model to ONNX
3. benchmark inference speed on the target hardware
4. reduce latency if needed
5. optimize the inference pipeline separately from the training pipeline

---

## 23. Troubleshooting

### Problem: `prepare_dataset` fails

Check:

```bash
ls data/external/cvat_export/images
ls data/external/cvat_export/labels
```

Make sure image/label names match.

### Problem: `train` cannot find the dataset

Check:

```bash
cat data/dataset/data_carton.yaml
```

Verify that the paths are correct.

### Problem: CUDA is not used

Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Problem: webcam does not open

Check:

```bash
ls /dev/video*
```

Also ensure no other application is using the camera.

### Problem: GitHub push fails because of large files

Check:

```bash
find . -type f -size +50M
git status
```

Make sure datasets, runs, videos, archives, and model weights are ignored by `.gitignore`.

---

## 24. Example End-to-End Session

Below is a typical full workflow from a fresh terminal.

### Step 1: go to the project directory

```bash
cd /home/emrah/Schreibtisch/CodeLab/YOLO/carton_yolo_project
```

### Step 2: activate the virtual environment

```bash
source .venv/bin/activate
```

### Step 3: enable command shortcuts

```bash
chmod +x bin/*
export PATH="$PWD/bin:$PATH"
```

### Step 4: place CVAT export files

```text
data/external/cvat_export/images/
data/external/cvat_export/labels/
```

### Step 5: prepare the dataset

```bash
prepare_dataset
```

### Step 6: train the model

```bash
train
```

### Step 7: validate the model

```bash
validate
```

### Step 8: test on unseen images

```bash
image
```

### Step 9: test on webcam

```bash
camera
```

### Step 10: export the model

```bash
export_model
```

---

## 25. Recommended Next Steps

After the first baseline is working, the most reasonable next steps are:

1. improve dataset quality
2. add more realistic carton images
3. include difficult backgrounds and lighting
4. test with webcam footage frequently
5. compare multiple training runs
6. export and benchmark the best model
7. prepare for future multi-class team integration

Possible future additions:

- `data_team.yaml`
- multi-class training
- shared team inference pipeline
- lightweight deployment benchmark scripts
- Raspberry Pi 5 specific inference tests

---

## Final Note

This repository is intentionally focused on a **clean and stable single-class baseline**.

That is the right engineering starting point.

A reliable baseline is more valuable than a complicated project structure that is hard to debug. Once the carton pipeline is stable, the project can be extended step by step with much lower risk.