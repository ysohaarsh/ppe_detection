"""Real-time PPE (personal protective equipment) detection with YOLO.

Reads frames from a webcam or video file, runs the YOLO model and draws a
colour-coded box per detection: green = PPE present, red = PPE missing,
blue = anything else.
"""
import argparse
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

CLASS_NAMES = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
               'Safety Vest', 'machinery', 'vehicle']

# BGR colours (OpenCV order)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

MISSING_PPE = {'NO-Hardhat', 'NO-Safety Vest', 'NO-Mask'}
PRESENT_PPE = {'Hardhat', 'Safety Vest', 'Mask'}

DEFAULT_MODEL = Path(__file__).with_name("ppe.pt")


def color_for_class(class_name: str) -> Tuple[int, int, int]:
    """Return the BGR box colour for a detected class name."""
    if class_name in MISSING_PPE:
        return RED
    if class_name in PRESENT_PPE:
        return GREEN
    return BLUE


def round_up_confidence(conf: float) -> float:
    """Round a confidence score up to two decimals (0.731 -> 0.74)."""
    return math.ceil(conf * 100) / 100


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time PPE detection with YOLO.")
    parser.add_argument("--source", default="0",
                        help="webcam index (default: 0) or path to a video file")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="path to the YOLO weights (default: ppe.pt next to this script)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="minimum confidence to draw a detection (default: 0.5)")
    parser.add_argument("--width", type=int, default=1280, help="webcam capture width")
    parser.add_argument("--height", type=int, default=720, help="webcam capture height")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    # Heavy imports live here so the helpers above can be imported (and tested) without them.
    import cv2
    import cvzone
    from ultralytics import YOLO

    is_webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if is_webcam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")
    if is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    model = YOLO(args.model)

    try:
        while True:
            success, img = cap.read()
            if not success:  # end of video or camera lost
                break
            for r in model(img, stream=True):
                for box in r.boxes:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                    conf = round_up_confidence(float(box.conf[0]))
                    if conf <= args.conf:
                        continue
                    class_name = CLASS_NAMES[int(box.cls[0])]
                    color = color_for_class(class_name)
                    cvzone.putTextRect(img, f'{class_name} {conf}',
                                       (max(0, x1), max(35, y1)), scale=1, thickness=1,
                                       colorB=color, colorT=(255, 255, 255), colorR=color,
                                       offset=5)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
