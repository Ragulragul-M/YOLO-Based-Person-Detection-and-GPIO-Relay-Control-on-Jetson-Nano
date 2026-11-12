# Changelog

## 2026-11-12 — GitHub preparation

- Organized the person-detection scripts for recorded video, RTSP camera input, and optional Jetson relay control.
- Replaced machine-specific model and video paths with configurable paths.
- Removed the embedded RTSP camera login from the shareable source; live scripts now read `CAMERA_URL` from the environment.
- Updated `relay.py` to follow the user's final code structure and relay logic: resize frames to 350 × 350, run YOLO every ninth frame with confidence 0.45, filter small/oversized or unusually shaped boxes, and switch the relay on detection.
- Kept the camera URL and model path configurable so credentials and machine-specific settings are not embedded in the public source.
- Added setup, training, hardware, dataset-attribution, and security notes.
- Kept datasets, model weights, the virtual environment, and generated recordings out of the repository by default.

## Existing project features recorded in the source

- YOLO person detection using class ID 0.
- Video and RTSP live-camera inference with annotated output.
- A terminal-only relay-control version for Jetson GPIO hardware.
- Relay activation uses BCM pin 18 by default, turns off at startup and shutdown, and changes state only when detection status changes.
- The saved training setup uses a one-class person dataset and a YOLOv8n training run configured for 100 epochs, 640-pixel images, and batch size 8.
