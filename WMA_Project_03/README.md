# WMA Project 03 – Feature-Based Object Detection & Tracking

A computer vision pipeline that detects and tracks an object in both static images and video using local feature descriptors (ORB, BRISK, SIFT), descriptor matching with Lowe's ratio test, and homography-based localization.

---

## Results

### Static image — object found

<img width="1546" height="1050" alt="SCR-20260330-syqh" src="https://github.com/user-attachments/assets/68dc2a63-02fd-478b-b571-dccd10a45b3f" />

### Keypoint match visualization (reference ↔ test image)

<img width="2756" height="884" alt="SCR-20260330-syrm" src="https://github.com/user-attachments/assets/ca01f7ca-305d-4a7f-8d31-5e8b14c627cf" />

### Video tracking — detected 

<img width="1608" height="962" alt="SCR-20260527-disc" src="https://github.com/user-attachments/assets/55b33ba2-fcdc-426f-a110-bf0ec48da674" />

### Video tracking — not detected

<img width="1608" height="960" alt="SCR-20260527-diyl" src="https://github.com/user-attachments/assets/9987f121-5724-47d5-8577-50b3a5698d96" />

---

## Features

- **Three descriptor methods** — ORB, BRISK, SIFT selectable at runtime via `--method`
- **Lowe's ratio test** — filters ambiguous matches (default ratio: 0.80)
- **Homography + RANSAC** — estimates the 2D transform between reference and scene; sanity-checked by determinant range, inlier count, and projected area
- **Green polygon overlay** — draws the detected object boundary on both images and video frames
- **Status overlay** — displays match count and DETECTED / NOT DETECTED in real time
- **Image and video modes** — same pipeline, single script

---

## Requirements

- Python 3.8+
- OpenCV (`cv2`) with contrib (required for SIFT)
- NumPy

```bash
pip install opencv-contrib-python numpy
```

---

## Usage

### Static image

```bash
python lab3_object.py --reference input/saw1.jpg --image input/saw2.jpg --method ORB
```

Opens three windows: **Reference image**, **Detected object**, **Matched keypoints**.

### Video

```bash
python lab3_object.py --reference input/saw1.jpg --video input/sawmovie.mp4 --method ORB
```

Press **Q** or **Esc** to quit.

### Method comparison

| Flag | Descriptor | Matcher norm |
|------|-----------|-------------|
| `--method ORB` | ORB (binary, 5000 features) | NORM_HAMMING |
| `--method BRISK` | BRISK (binary) | NORM_HAMMING |
| `--method SIFT` | SIFT (float) | NORM_L2 |

---

## How It Works

```
Reference image → grayscale → detectAndCompute → keypoints + descriptors
Test image/frame → grayscale → detectAndCompute → keypoints + descriptors
                                        ↓
                          knnMatch (k=2) + Lowe's ratio test
                                        ↓
                    findHomography (RANSAC, threshold=5px)
                                        ↓
              sanity checks: det(H) ∈ [0.01, 100], inliers ≥ 4 & ≥ 20%, area ≥ 100px
                                        ↓
                   perspectiveTransform → green polygon on scene
```

### Localization sanity checks

False homographies are rejected by three independent guards:

| Check | Threshold | Purpose |
|-------|-----------|---------|
| `det(H)` | 0.01 – 100 | Rejects degenerate or extreme scale changes |
| Inlier count | ≥ 4 | Minimum geometric support |
| Inlier ratio | ≥ 20 % | Filters low-quality homographies |
| Projected area | ≥ 100 px | Discards collapsed quads |

---

## Project Structure

```
WMA_Project_03/
├── lab3_object.py        # Main script
├── input/
│   ├── saw1.jpg          # Reference image
│   ├── saw2.jpg          # Test image
│   ├── saw3.jpg          # Test image
│   ├── saw4.jpg          # Test image
│   └── sawmovie.mp4      # Test video
└── README.md
```

---

## License

For educational use.
