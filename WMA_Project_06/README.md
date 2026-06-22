# WMA Project 06 — YOLO Object Detection (Avocado Fine-Tuning)

YOLOv8 object detection with custom fine-tuning. The base **YOLOv8n** model
is fine-tuned on a custom **avocado** class (not part of COCO), and the
two models are compared on the same image to show what fine-tuning buys
you.

Built with [Ultralytics YOLO](https://docs.ultralytics.com/) + OpenCV.

---

## Task

Implement a full YOLO pipeline end-to-end:
1. Load the YOLO base model
2. Run inference on images, video files, or a camera stream
3. Parse boxes / class IDs / confidences and draw them
4. Prepare a YOLO-format dataset (images + labels + `data.yaml`)
5. Validate dataset structure before training
6. Fine-tune the base model on the custom class
7. Load the fine-tuned weights and compare against the base model on the
   same image

---

## Dataset

Custom avocado dataset from
[Roboflow Universe](https://universe.roboflow.com/), exported in
**YOLOv8 format** and reorganized to match the layout the script expects.

| Split | Images |
|-------|--------|
| Train | 105    |
| Val   | 30     |
| Test  | 16     |
| **Total** | **151** |

Single class: `avocado` (whole, halved, sliced). Chosen specifically
because **avocado is not in COCO**, so the base YOLOv8n cannot detect it
at all — the contrast with the fine-tuned model is unambiguous.

The dataset is **not included in this repository**. Download a small
avocado dataset from Roboflow Universe (or annotate your own) and arrange
it as below.

### Expected dataset layout

```
avocado_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/         # one .txt per image, YOLO format
│   └── val/
└── data.yaml
```

### `data.yaml`

```yaml
path: avocado_dataset
train: images/train
val: images/val
names:
  0: avocado
```

YOLO label format (one line per object, all values normalized to `[0, 1]`):

```
<class_id> <x_center> <y_center> <width> <height>
```

---

## Pipeline

### Inference (base or fine-tuned)
1. `load_model()` — load weights via `ultralytics.YOLO`
2. `run_detection()` — forward pass on an image / frame
3. `parse_results()` — extract `xyxy`, `conf`, `cls` from the boxes object
4. `filter_detections()` — drop detections below `--confidence`
5. `draw_detections()` — bounding boxes + `label: conf` text overlay
6. `add_diagnostics()` — detection count + confidence threshold corner HUD

### Training
1. `check_dataset_structure()` — verifies all 4 image/label folders + `data.yaml` exist before doing anything else
2. `fine_tune_model()` — `model.train(data=…, epochs=…, imgsz=…)`
3. Returns the path to `runs/detect/train*/weights/best.pt`

### Hyperparameters used

| Param      | Value      |
|------------|------------|
| Base model | `yolov8n.pt` |
| Epochs     | 20         |
| Image size | 640        |
| Optimizer  | Ultralytics default (SGD, auto-tuned LR) |
| Hardware   | MacBook Air M2, CPU only |
| Train time | ~10 minutes |

---

## How to run

### Install

```bash
pip install ultralytics opencv-python numpy
```

The first run will auto-download `yolov8n.pt` (~6 MB) to the working
directory.

### Inference on an image

```bash
python LAB6.py --model yolov8n.pt --image path/to/img.jpg
```

### Inference on a video / camera

```bash
python LAB6.py --model yolov8n.pt --video path/to/clip.mp4
python LAB6.py --model yolov8n.pt --camera
```

Press `q` or `Esc` to quit the video window.

### Fine-tune on the avocado dataset

```bash
python LAB6.py \
    --model yolov8n.pt \
    --train \
    --train-data ./avocado_dataset \
    --epochs 20 \
    --imgsz 640
```

Weights are saved under `runs/detect/train*/weights/best.pt`.

### Compare base vs fine-tuned on the same image

```bash
python LAB6.py \
    --model yolov8n.pt \
    --trained-model runs/detect/train/weights/best.pt \
    --image avocado_test.jpg \
    --compare
```

Two OpenCV windows open side by side — *Base model* and *Fine-tuned model*.

### Annotation help

```bash
python LAB6.py --model yolov8n.pt --show-annotation-help
```

Prints the expected dataset layout, label format, and recommended
annotation tools (LabelImg, CVAT, Makesense.ai, Roboflow).

---

## Results

Final metrics on the validation set after 20 epochs:

| Metric         | Value |
|----------------|-------|
| Precision      | 0.949 |
| Recall         | 0.913 |
| mAP@0.5        | 0.962 |
| mAP@0.5:0.95   | 0.837 |

Solid numbers given the small dataset (105 train images) and short
training schedule.

### Base vs Fine-tuned (same image)

On a photo of three avocado halves on a wooden tray:

| Model       | Detections | What it found |
|-------------|------------|---------------|
| YOLOv8n (base)     | 2 | `dining table` (the tray) + `cake` (one avocado) — both wrong, no avocado class to begin with |
| YOLOv8n (fine-tuned) | 4 | 4× `avocado` at confidence **0.89 / 0.81 / 0.90 / 0.80** |

<img width="1419" height="546" alt="base_vs_finetuned" src="https://github.com/user-attachments/assets/47ad44ab-f623-4170-b281-32d076b3fef6" />

### Training artifacts

YOLOv8 auto-generates these under `runs/detect/train*/` after training.

**Training curves** — loss, precision, recall, and mAP across the 20 epochs:

<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/8ed24596-bbfc-4e1e-b680-5263ea6a1b1d" />

**Confusion matrix** — single-class case, but useful to see false positives /
background confusion:

<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/b92b8dae-5148-4d3f-b776-fca34898e683" />

**Validation predictions** — sample images from the val set with predicted
boxes:

<img width="1920" height="1920" alt="val_batch_pred" src="https://github.com/user-attachments/assets/f59561e2-65fe-48f1-8afb-df023a0092ca" />

**Label statistics** — class count and bounding-box position/size
distributions across the training set:

<img width="1600" height="1600" alt="labels" src="https://github.com/user-attachments/assets/a0b19dc1-fa55-4e26-a4fa-10a93fa725d0" />

**Sample training batch** — images with ground-truth boxes as the model
sees them during training:

<img width="1920" height="1920" alt="train_batch" src="https://github.com/user-attachments/assets/8c42de6e-ef13-4d6a-800b-398e5d81747a" />

---

## Project structure

```
WMA_Project_06/
├── LAB6.py                 # main script (inference + training + comparison)
├── README.md               # this file
└── results/                # figures embedded in the README
    ├── base_vs_finetuned.jpeg
    ├── results.png
    ├── confusion_matrix.png
    ├── confusion_matrix_normalized.png
    ├── labels.jpg
    ├── train_batch.jpg
    └── val_batch_pred.jpg
```

The dataset, base weights (`yolov8n.pt`), and Ultralytics training outputs
(`runs/`) are generated locally and excluded from the repo via `.gitignore`.

---

## Notes / things I tried

- **Why avocado?** Picking a class the base model already knows (e.g. apple,
  orange) would make the "before vs after" comparison ambiguous —
  fine-tuning would just nudge confidences. Avocado is completely outside
  COCO, so the base model literally cannot output the right label. The
  contrast is then clean.
- **Dataset size.** 105 training images sounds tiny but it was enough for
  P/R > 0.9 because avocados are visually distinctive (round, dark green,
  often halved with a pit) and the dataset has limited background variety.
  More data would mostly help with weird angles and partial occlusion.
- **Annotation quality > quantity.** I checked a couple of other
  Roboflow avocado datasets first — some had boxes that didn't actually
  enclose the fruit, or labeled the cutting board as avocado. Training on
  those would have produced garbage metrics regardless of size.
- **20 epochs on CPU.** Felt like a sweet spot. With more epochs val mAP
  plateaued. Doing it on a GPU would have cut training to ~1 minute but
  CPU is fine for a dataset this small.
- **Ultralytics ergonomics.** The same `YOLO` class handles loading,
  inference, and `.train()` — no separate training script, no boilerplate.
  Most of the code in `LAB6.py` is actually OpenCV plumbing
  (drawing boxes, diagnostic HUD, video loop), not YOLO itself.
