# Lab 2 – Coin Detection & Tray Localization

A computer vision pipeline that detects Polish coins (5 PLN and 5 gr) in static images, localizes an orange tray using HSV segmentation + Hough line detection, and computes the total monetary value of coins both on and off the tray.

Evaluation: Full score (10/10) ✅

---

## Results

| tray4.jpg | tray5.jpg |
|:---------:|:---------:|
| <img width="1502" height="2054" alt="SCR-20260322-pbbg" src="https://github.com/user-attachments/assets/4138687e-754e-4991-b0d3-e2e9a4775d25" />| <img width="1500" height="2058" alt="SCR-20260322-pbdd" src="https://github.com/user-attachments/assets/3d2ff96f-d171-44ec-995a-27041c0e773c" /> |

| tray8.jpg | tray2.jpg |
|:---------:|:---------:|
| <img width="1496" height="2052" alt="SCR-20260322-pbgh" src="https://github.com/user-attachments/assets/b46a96a8-1a1a-494f-8368-4b466c9c05a6" /> | <img width="1498" height="2046" alt="SCR-20260322-pbiw" src="https://github.com/user-attachments/assets/ca7abd18-5b73-4db5-a1f0-5f12ef1d9f3c" /> |

> **Green circle** = 5 PLN &nbsp;|&nbsp; **Blue circle** = 5 gr &nbsp;|&nbsp; **Yellow box** = detected tray boundary

---

## Features

- **Tray detection** – HSV mask for orange colour → morphological cleaning → HoughLinesP for edge support → `minAreaRect` for a tight rotated bounding box
- **Coin detection** – Gaussian blur → `HoughCircles` (Gradient method) → edge-density ring filter to reject false positives
- **Coin classification** – primary rule: radius threshold (≥ 34 px → 5 PLN); borderline radii (32–33 px) refined by HSV saturation and brightness features
- **On/Off tray split** – centroid-in-bbox test separates coins placed on the tray from those outside
- **Batch processing** – processes all `tray1.jpg … tray8.jpg` in a directory and prints a grand summary

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

### Single image (programmatic)

```python
from lab2_coin_detection import process_image

result = process_image("tray4.jpg", display=True, save_dir="results")
print(result["total_on"], result["total_off"])
```

### Batch mode

```bash
python lab2_coin_detection.py <image_directory>
```

Add `--no-display` to skip the interactive window (useful for headless environments):

```bash
python lab2_coin_detection.py ./images --no-display
```

Expected input filenames: `tray1.jpg` … `tray8.jpg`

---

## Console output example

```
═══════════════════════════════════════════════════════
  Image: tray4.jpg
═══════════════════════════════════════════════════════
  Tray bounding box : (278,255) size=(337x867)
  Valid circles found : 12

  ── Coins ON the tray (6) ─────────────
     5PLN  r=36px  centre=(428,390)  ...  5.00 PLN
     5gr   r=30px  centre=(428,462)  ...  0.05 PLN
     ...
     Subtotal ON  : 5.25 PLN

  ── Coins OFF the tray (6) ───────────
     ...
     Subtotal OFF : 5.25 PLN

  ── Grand total ───────────────────────────────
     ON  tray :  6 coins = 5.25 PLN
     OFF tray :  6 coins = 5.25 PLN
     TOTAL    : 12 coins = 10.50 PLN
```

---

## How It Works

### 1 · Tray localization

```
BGR → HSV  →  inRange([5,120,120]–[25,255,255])
           →  CLOSE + OPEN (7×7 kernel)
           →  Canny edges
           →  HoughLinesP → horizontal/vertical line buckets
           →  Largest contour → minAreaRect
```

Horizontal lines vote on top/bottom edges; vertical lines vote on left/right edges. The final bounding box is the union of the contour rect and the Hough votes.

### 2 · Coin detection

```
BGR → Gray  →  GaussianBlur(9×9)
            →  HoughCircles (dp=1.2, param2=33, r∈[20,120])
            →  Ring edge-density filter (≥ 10 % edge pixels in outer ring)
```

The ring filter removes circles that land on uniform surfaces (tray walls, background texture).

### 3 · Classification

| Radius | Class |
|--------|-------|
| ≥ 34 px | **5 PLN** |
| 32–33 px | **5 PLN** if low saturation + high brightness, else **5 gr** |
| ≤ 31 px | **5 gr** |

### 4 · Visualisation

| Colour | Meaning |
|--------|---------|
| 🟢 Bright green | 5 PLN on tray |
| 🟢 Dark green | 5 PLN off tray |
| 🔵 Cyan | 5 gr on tray |
| 🔵 Dark blue | 5 gr off tray |
| 🟡 Yellow | Tray bounding box |

---

## Key Parameters

| Constant | Value | Purpose |
|----------|-------|---------|
| `RADIUS_THRESHOLD_5PLN` | 34 px | Hard radius cut-off for 5 PLN |
| `COIN_MAX_VALID_RADIUS` | 55 px | Discard oversized false positives |
| `COIN_MIN_RING_DENSITY` | 0.10 | Minimum edge fraction to accept a circle |
| `RING_WIDTH` | 6 px | Annular ring width for edge check |
| `HOUGH_CIRCLE_PARAMS.param2` | 33 | Accumulator threshold (lower → more circles) |

---

## Project Structure

```
.
├── lab2_coin_detection.py   # Main script
├── input_images/            # Input images (tray1.jpg … tray8.jpg)
├── results/                 # Output annotated images (auto-created)
└── README.md
```

---

## License

For educational use.
