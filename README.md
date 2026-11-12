# YOLO Person Detection and Relay Activation

**Project:** Ashok Leyland person-detection and Jetson relay prototype

This project uses a YOLO model to detect people in recorded video or a 360° RTSP camera feed. The Jetson Nano relay script processes the live camera feed and switches a GPIO relay when a qualifying person detection is present.

```text
Video file  → video.py → YOLO → annotated MP4
360° RTSP   → live.py  → YOLO → live view
360° RTSP   → relay.py → YOLO → Jetson Nano GPIO pin 18 relay
```

## Project files

- **`video.py`** — runs person detection on a recorded video, displays annotated frames, and saves an annotated MP4 by default.
- **`live.py`** — runs person detection on an RTSP camera feed and displays the annotated live view.
- **`relay.py`** — runs live RTSP person detection in the terminal and controls a Jetson GPIO relay; it uses BCM pin 18.
- **`data.yaml`** — describes the one-class person dataset layout used for training.

The inference scripts use person class ID `0`. The recorded-video and live-view scripts use image size 640, confidence 0.30, and IoU 0.40 by default. Relay mode uses the separate settings below.

## Recorded training configuration

The project training record describes a YOLOv8n run with the following setup:

| Setting | Recorded value |
| --- | --- |
| Training images | 4,108 |
| Validation images | 388 |
| Classes | 1 (`person`) |
| Epochs | 100 |
| Training image size | 640 × 640 |
| Batch size | 8 |
| Training device | NVIDIA RTX 3050 laptop GPU |

Example command for the recorded training settings:

```powershell
yolo detect train `
  data=data.yaml `
  model=yolov8n.pt `
  epochs=100 `
  imgsz=640 `
  batch=8 `
  device=0
```

No final accuracy score or validation on unseen footage is documented, so no such result is claimed here.

## Relay settings

The final relay configuration in `relay.py`:

- Reads an RTSP feed and processes every ninth frame.
- Resizes each processed frame to 350 × 350 before inference.
- Uses YOLO image size 224 and confidence threshold 0.45.
- Considers class ID `0` (`person`) and applies box filters: area at least 2,800 pixels and no more than 80% of the frame; height at least half the width; and width and height at least 25 pixels.
- Sets BCM pin 18 LOW at startup, HIGH when a qualifying person is detected, and LOW when no qualifying detection is present.
- Prints detection and relay status changes to the terminal.

The relay output level and wiring must match the relay module and Jetson hardware.

## Setup

Use a Python and PyTorch installation compatible with the computer or Jetson software image. The general dependencies listed in `requirements.txt` are OpenCV, PyTorch, and Ultralytics. On Jetson Nano, use the PyTorch/OpenCV packages compatible with the installed JetPack image; `requirements-jetson.txt` notes that `Jetson.GPIO` must be installed separately using a compatible Jetson package or build.

For a Windows development environment, create a virtual environment and install the general dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put your trained YOLO weights at `weights/best.pt` for the default `video.py` and `live.py` setup. Set `MODEL_PATH` or pass `--model` to use a different weights file. For Jetson relay operation, set `MODEL_PATH` to the model file on the Jetson. The dataset, trained weights, camera recordings, and generated output videos are not included in this repository.

## Run the project

### Test a recorded video

```powershell
python video.py --video "path\to\input.mp4"
```

By default, `video.py` displays the annotated video and writes `output.mp4`. Press **Esc** to close the display. Use `--model` or `MODEL_PATH` to select another model file, and `--output` to change the output video name.

### View a live RTSP camera feed

Set `CAMERA_URL` and, when needed, `MODEL_PATH` in PowerShell, keeping the camera address and credentials private:

```powershell
$env:CAMERA_URL = "rtsp://<user>:<password>@<camera-host>:554/<stream-path>"
$env:MODEL_PATH = "weights\best.pt"
python live.py
```

Alternatively, pass the stream address with `--source`. `live.py` reads `MODEL_PATH` or defaults to `weights/best.pt`; you can also pass `--model`.

### Run relay control on Jetson Nano

Install the Jetson-compatible GPIO package and set both environment variables to your camera stream and model file:

```bash
export CAMERA_URL='rtsp://<user>:<password>@<camera-host>:554/<stream-path>'
export MODEL_PATH='/path/to/best.pt'
python3 relay.py
```

`relay.py` requires `CAMERA_URL`, reads `MODEL_PATH` when set, and uses the BCM numbering scheme for GPIO pin 18.

## Dataset and attribution

The original `data.yaml` identifies [Roboflow Universe Object Detection Project, version 4](https://universe.roboflow.com/ml-project-oihij/object-detection-project-ki1qs/dataset/4) and records the dataset license as **CC BY 4.0**. Dataset files are not included. Retain the required attribution and follow the license terms if redistributing dataset images or derived materials.

## Limitations and safety

This relay prototype is **not safety-certified** and must not be used as the sole protection for people or equipment. This project description does not claim thermal sensing, completed PLC integration, measured accuracy, or verified performance on unseen footage.

Keep camera URLs, credentials, private recordings, dataset exports, and model weights out of public commits.
