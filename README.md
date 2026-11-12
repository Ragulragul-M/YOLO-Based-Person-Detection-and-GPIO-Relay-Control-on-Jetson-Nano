# Ashok Leyland Person Detection Demo

YOLO-based person detection for recorded video and an RTSP camera feed, with an optional Jetson Nano relay-control script.

## What is included

- `video.py` — detects people in a video file and can save an annotated copy.
- `live.py` — runs detection on a live RTSP camera stream and displays the annotated feed.
- `relay.py` — terminal-only live detection that switches a relay when a person is detected. This requires Jetson GPIO hardware.
- `data.yaml` — dataset layout for the one-class `person` detector used in the training setup.
- `CHANGELOG.md` — project features and the repository-preparation updates.

The dataset, Python environment, camera recordings, and model weights are not included. Put your trained model at `weights/best.pt`; the repository ignores model files and generated recordings by default.

## Setup

Create and activate a Python virtual environment, then install the packages:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install a PyTorch build that matches your computer or Jetson software image. Jetson installations often need the device-specific PyTorch package. For relay control, install the GPIO package supported by your Jetson image as described below.

Copy your trained YOLO model to `weights/best.pt`. This file is intentionally excluded from Git. `video.py` and `live.py` use that path by default; `relay.py` defaults to `/home/foundry/best.pt` to match the Jetson setup and can be changed with `MODEL_PATH`.

## Run detection on a video

```powershell
python video.py --video "path\to\input.mp4"
```

By default, the script displays detections and writes `output.mp4`. To choose a different model or output path:

```powershell
python video.py --video "path\to\input.mp4" --model "weights\best.pt" --output "output.mp4"
```

## Run with an RTSP camera

Set the camera stream address in your terminal. Keep credentials out of source files and commits.

```powershell
$env:CAMERA_URL = "rtsp://<user>:<password>@<camera-host>:554/<stream-path>"
python live.py
```

For Jetson relay control, set `CAMERA_URL` and, if needed, `MODEL_PATH`, then run:

```bash
export CAMERA_URL='rtsp://<user>:<password>@<camera-host>:554/<stream-path>'
export MODEL_PATH='/home/foundry/best.pt'
python3 relay.py
```

On the Jetson, install `Jetson.GPIO` using the package or instructions compatible with its JetPack image. Confirm the relay board's active-high/active-low behavior and wiring before connecting it to equipment. The script defaults to BCM pin 18; set `RELAY_PIN` if your setup uses another pin.

## Train the detector

Download or place the appropriately licensed dataset in `dataset/` using the layout described by `data.yaml`, then run from the repository root:

```powershell
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=8 device=0 project=runs
```

The previous training log showed a one-class person dataset and this training configuration. No final accuracy or validation score is claimed here. Add the resulting model to `weights/best.pt` locally to run inference.

## Data and attribution

The saved dataset configuration identifies the [Roboflow Universe object detection dataset, version 4](https://universe.roboflow.com/ml-project-oihij/object-detection-project-ki1qs/dataset/4) under the CC BY 4.0 license. The image files are not included here. If you redistribute dataset images or derived material, follow the dataset license and retain the required attribution. Review the dataset page and its current terms before publishing it.

## Security and privacy

Camera addresses and credentials are read from `CAMERA_URL` rather than stored in these scripts. Do not commit `.env` files, passwords, private camera footage, dataset exports, or other credentials. If a real camera password was used in an earlier version, change it before sharing the project.

## License

No license has been added yet. Add one only after confirming that you have the rights to license the code and any material you distribute with it.
