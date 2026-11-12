"""Run YOLO person detection on a recorded video."""

import argparse
import os
import sys
import time

import cv2
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument(
        "--model",
        default=os.environ.get("MODEL_PATH", "weights/best.pt"),
        help="YOLO model weights (default: MODEL_PATH or weights/best.pt)",
    )
    parser.add_argument("--output", default="output.mp4", help="Annotated output video")
    parser.add_argument("--no-save", action="store_true", help="Do not write an output video")
    parser.add_argument("--confidence", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.4)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.video):
        print(f"Input video not found: {args.video}", file=sys.stderr)
        return 1
    if not os.path.isfile(args.model):
        print(f"Model weights not found: {args.model}", file=sys.stderr)
        print("Place your trained weights at weights/best.pt or pass --model.", file=sys.stderr)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    model = YOLO(args.model)
    print("Classes:", model.names)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Could not open video: {args.video}", file=sys.stderr)
        return 1

    writer = None
    last_detect_time = 0.0
    hold_time = 1.0
    person_class_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
                        f"Person {confidence:.2f}",
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
            print(status)
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if not args.no_save and writer is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not fps or fps <= 0:
                    fps = 20.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.output, fourcc, fps, (800, 600))
                if not writer.isOpened():
                    print(f"Could not create output video: {args.output}", file=sys.stderr)
                    return 1

            if writer is not None:
                writer.write(frame)

            cv2.imshow("Video Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
