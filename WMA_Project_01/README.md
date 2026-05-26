# Lab 1 – Red Object Detection & Horizontal Tracking

A real-time computer vision script that detects a red object in a video stream using HSV color segmentation and image moments, then tracks its horizontal position relative to the frame center.

---

## Demo

The script displays a side-by-side window: the original frame with overlays on the left, and the binary color mask on the right.

```
┌─────────────────────────┬─────────────────────────┐
│   Original + overlays   │      Red mask (B&W)      │
└─────────────────────────┴─────────────────────────┘
```

---

## Features

- **HSV color segmentation** – two hue ranges cover both ends of the red spectrum (0–8° and 172–180°)
- **Morphological cleaning** – OPEN removes noise, CLOSE fills gaps in the detected region
- **Moment-based detection** – centroid and approximate radius computed from image moments (no contour fitting)
- **Deviation overlay** – horizontal distance from the frame center shown as text and a color bar (blue = left, green = right)

---

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## Usage

```bash
python lab1_object_detection_1.py --video <path_to_video>
```

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to the input video file |
| `--min-area` | `120.0` | Minimum pixel area to accept a detection |

### Example

```bash
python lab1_object_detection_1.py --video test_clip.mp4 --min-area 200
```

Press **Q** or **Esc** to quit.

---

## How It Works

1. Each frame is converted from BGR to HSV.
2. Two `inRange` masks are combined to capture the full red hue range.
3. Morphological OPEN + CLOSE operations clean up the mask.
4. Image moments (`cv2.moments`) compute the centroid `(cx, cy)` and an area-based radius.
5. Detections outside the `[min_area, 20000 px]` range are discarded.
6. The horizontal deviation `cx − frame_center_x` is visualised with a bar and text overlay.

---

## Project Structure

```
.
├── lab1_object_detection_1.py   # Main script
└── README.md
```

---

## License

For educational use.
