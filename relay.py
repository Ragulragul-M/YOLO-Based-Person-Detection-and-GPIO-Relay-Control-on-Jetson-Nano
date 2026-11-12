import os

import cv2
import torch
from ultralytics import YOLO
import Jetson.GPIO as GPIO

# ---------------- GPIO SETUP ----------------
RELAY_PIN = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY_PIN, GPIO.OUT)

# Relay OFF initially
GPIO.output(RELAY_PIN, GPIO.LOW)

# ---------------- DEVICE ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device, flush=True)

# ---------------- LOAD MODEL ----------------
model_path = os.environ.get("MODEL_PATH", "/home/foundry/best.pt")
model = YOLO(model_path)

# ---------------- RTSP CAMERA INPUT ----------------
# Set CAMERA_URL in your environment; do not put camera credentials in this file.
rtsp_url = os.environ["CAMERA_URL"]

cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

# Reduce RTSP buffering
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error opening RTSP stream")
    exit()

print("RTSP Camera Started", flush=True)

frame_count = 0
prev_status = ""
relay_state = 0

# ---------------- MAIN LOOP ----------------
while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera Frame Error")
        continue

    # Resize for speed
    frame = cv2.resize(frame, (350, 350))

    frame_count += 1

    # Skip frames for speed
    if frame_count % 9 != 0:
        continue

    # ---------------- YOLO INFERENCE ----------------

    results = model(
        frame,
        imgsz=224,
        conf=0.45,
        device=device,
        verbose=False
    )

    person_detected = False

    frame_area = frame.shape[0] * frame.shape[1]

    # ---------------- DETECTIONS ----------------

    for r in results:

        if r.boxes is None:
            continue

        for box in r.boxes:

            cls = int(box.cls[0])

            # PERSON CLASS
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            w = x2 - x1
            h = y2 - y1

            area = w * h

            # ---------------- FILTERS ----------------

            if area < 2800:
                continue

            if area > 0.80 * frame_area:
                continue

            if h < w * 0.5:
                continue

            if h < 25 or w < 25:
                continue

            # PERSON DETECTED
            person_detected = True

    # ---------------- STATUS ----------------

    status = "PERSON DETECTED" if person_detected else "NO PERSON"

    if status != prev_status:

        print(status, flush=True)

        prev_status = status

    # ---------------- RELAY CONTROL ----------------

    if person_detected:

        if relay_state == 0:

            GPIO.output(RELAY_PIN, GPIO.HIGH)

            print("RELAY ON", flush=True)

            relay_state = 1

    else:

        if relay_state == 1:

            GPIO.output(RELAY_PIN, GPIO.LOW)

            print("RELAY OFF", flush=True)

            relay_state = 0

# ---------------- CLEANUP ----------------

cap.release()

GPIO.cleanup()

print("Program Ended")
