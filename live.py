"""Run YOLO person detection on an RTSP camera stream."""

import argparse
import os
import sys
import threading
import time

import cv2
import torch
from ultralytics import YOLO


class ThreadedVideoCapture:
    """Keep the latest RTSP frame in a background reader thread."""

    def __init__(self, source):
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera stream. Check CAMERA_URL and network access.")
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                self.frame = frame
            else:
                time.sleep(0.02)

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)
        self.cap.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=os.environ.get("CAMERA_URL"), help="RTSP URL; defaults to CAMERA_URL")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "weights/best.pt"),
        help="YOLO model weights (default: MODEL_PATH or weights/best.pt)",
    )
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.4)
    args = parser.parse_args()

    if not args.source:
        parser.error("Set CAMERA_URL or pass --source. Do not put camera credentials in source code.")
    if not os.path.isfile(args.model):
        print(f"Model weights not found: {args.model}", file=sys.stderr)
        print("Place your trained weights at weights/best.pt or pass --model.", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = YOLO(args.model)
    print("Classes:", model.names)
    person_class_id = 0

    try:
        cap = ThreadedVideoCapture(args.source)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 1

    last_detect_time = 0.0
    hold_time = 1.0
    try:
        while True:
            frame = cap.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (800, 600))
            results = model(
                frame,
                imgsz=640,
                conf=args.confidence,
                iou=args.iou,
                device=device,
                classes=[person_class_id],
                verbose=False,
            )

            person_count = 0
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    if int(box.cls[0]) != person_class_id:
                        continue
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    width, height = x2 - x1, y2 - y1
                    if height < 60 or width < 30:
                        continue
                    person_count += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{confidence:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

            now = time.time()
            if person_count:
                last_detect_time = now
            detected = now - last_detect_time < hold_time
            status = f"PERSON DETECTED ({person_count})" if detected else "NO PERSON"
            color = (0, 0, 255) if detected else (0, 255, 0)
            print(status, flush=True)
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
