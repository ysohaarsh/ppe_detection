# PPE Detection using YOLO

![PPE Detection](ppe.jpg)

Real-time detection of personal protective equipment (hard hats, masks, safety vests)
on a webcam or video stream, built on [Ultralytics YOLO](https://docs.ultralytics.com/).
Inspired by Murtaza's Workshop object detection tutorial.

Detected classes: `Hardhat`, `Mask`, `NO-Hardhat`, `NO-Mask`, `NO-Safety Vest`, `Person`,
`Safety Cone`, `Safety Vest`, `machinery`, `vehicle`.
Boxes are green when PPE is present, red when it is missing and blue for everything else.

## Setup

```bash
git clone https://github.com/ysohaarsh/ppe_detection.git
cd ppe_detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.8+ is required. The weights (`PPEDetection/ppe.pt`) are included in the repo.

## Usage

```bash
# webcam 0 (default)
python PPEDetection/PPEDetection.py

# a video file, with a stricter confidence threshold
python PPEDetection/PPEDetection.py --source path/to/video.mp4 --conf 0.6
```

| Option | Default | Description |
|---|---|---|
| `--source` | `0` | Webcam index or video file path |
| `--model` | `PPEDetection/ppe.pt` | YOLO weights |
| `--conf` | `0.5` | Minimum confidence to draw a detection |
| `--width` / `--height` | `1280` / `720` | Webcam capture size |

Press `q` in the preview window to quit.

## Tests

```bash
pip install pytest
pytest
```

The tests cover the pure helper functions only; they need no GPU, camera or weights.

## Other scripts

`pokerHandDetector/` contains a separate playing-card / poker-hand detector using the same approach.

## Credits

Based on Murtaza's Workshop object detection tutorial.
