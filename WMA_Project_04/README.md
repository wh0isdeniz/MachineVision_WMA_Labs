# WMA Project 04 – Optical Flow Feature Tracking

A real-time video processing pipeline that detects feature points and tracks them across frames using the Lucas-Kanade optical flow algorithm. Accumulates motion trajectories as colored trails on the output video.

Evaluation: Full score (10/10) ✅

---

## Features

- **Shi-Tomasi corner detection** — finds up to 100 high-quality feature points on the first frame (and whenever all points are lost)
- **Lucas-Kanade optical flow** — pyramidal LK with a 15×15 window and 2 pyramid levels tracks points between consecutive frames
- **Status filtering** — only points with `status == 1` are kept; failed tracks are discarded each frame
- **Trajectory accumulation** — red lines are drawn on a persistent mask that overlays the live frame, building up motion trails over time
- **Auto re-detection** — when all tracked points are lost, features are re-detected and the trail mask is reset

---

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- NumPy

```bash
pip install opencv-python numpy
```

---

## Usage

```bash
python lab4_tracking.py --video input_movie.mov
```

Press **Q** or **Esc** to quit.

---

## How It Works

```
First frame → grayscale → goodFeaturesToTrack → initial points
                                   ↓
              ┌────────────── main loop ──────────────────┐
              │  next frame → grayscale                   │
              │       ↓                                   │
              │  calcOpticalFlowPyrLK (LK pyramidal)      │
              │       ↓                                   │
              │  filter by status == 1                    │
              │       ↓                                   │
              │  draw red trails on persistent mask       │
              │  draw green dots on current frame         │
              │  cv2.add(frame, mask) → display           │
              │       ↓                                   │
              │  no points left? → re-detect + reset mask │
              └───────────────────────────────────────────┘
```

### Key parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `maxCorners` | 100 | Max feature points detected |
| `qualityLevel` | 0.3 | Minimum corner quality (0–1) |
| `minDistance` | 7 px | Minimum distance between corners |
| `winSize` | 15×15 | LK search window per pyramid level |
| `maxLevel` | 2 | Pyramid depth for LK |
| `waitKey` delay | 30 ms | Playback speed (~33 fps cap) |

---

## Project Structure

```
WMA_Project_04/
├── lab4_tracking.py      # Main script
├── input_movie.mov       # Input video
└── README.md
```

---

## License

For educational use.
