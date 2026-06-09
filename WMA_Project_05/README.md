# WMA Project 05 — Fruit Classification with CNN

Convolutional neural network that classifies images into three fruit classes:
**banana**, **orange**, **lemon**.

Evaluation: Full score (10/10) ✅

Built with TensorFlow / Keras. Trained on a curated subset of the
[Fruits 360](https://www.kaggle.com/datasets/moltean/fruits) dataset.

---

## Task

Implement and train a CNN end-to-end:
1. Load and preprocess images from a folder-per-class dataset
2. Define a data augmentation pipeline
3. Build a CNN architecture from scratch
4. Compile and train with validation split + early stopping
5. Evaluate on a single test image
6. Plot accuracy / loss curves

---

## Dataset

Studio-style fruit images on a white background, organized as:

```
dataset/
├── banana/   (1210 images)
├── orange/   (1195 images)
└── lemon/    ( 492 images)
```

- **Total:** ~2897 images
- **Source:** Fruits 360 (filtered to the three target classes)
- **Note on imbalance:** the `lemon` class has roughly 40% of the samples of
  the other two. A stratified train/val split is used to keep this proportion
  consistent between splits.

> The provided archive unzips to `wma05_Dataset/`. Rename it to `dataset/`
> (or update `dataset_path` in `main()`) before running.

---

## Pipeline

### 1. Loading & preprocessing
- Images read with OpenCV, converted BGR → RGB
- Resized to **128 × 128**
- Normalized to `[0, 1]`
- Labels one-hot encoded (3 classes)
- Shuffled with a fixed seed (`np.random.seed(42)`) for reproducibility

### 2. Augmentation
A mild augmentation pipeline is defined (rotation, flip, small shifts, brightness),
but **not applied during training**. Fruits 360 has clean studio backgrounds, and
aggressive transforms made validation loss noisy in practice. The pipeline is left
in the code for reference and easy re-enabling.

### 3. Model architecture

```
Input (128, 128, 3)
│
├── Conv2D(32, 3×3, ReLU, same) ─ Conv2D(32, 3×3, ReLU, same) ─ MaxPool(2×2)
├── Conv2D(64, 3×3, ReLU, same) ─ Conv2D(64, 3×3, ReLU, same) ─ MaxPool(2×2)
├── Conv2D(128, 3×3, ReLU, same) ───────────────────────────── MaxPool(2×2)
│
├── Flatten
├── Dense(128, ReLU)
├── Dropout(0.3)
└── Dense(3, Softmax)
```

Filters increase 32 → 64 → 128 to learn progressively higher-level features
(edges → textures → fruit shape). Dropout(0.3) on the classifier head reduces
overfitting on the smaller `lemon` class.

### 4. Training
- **Loss:** `categorical_crossentropy`
- **Optimizer:** Adam, learning rate `1e-3`
- **Metric:** accuracy
- **Epochs:** 30 (with EarlyStopping, `patience=5`, restore best weights)
- **Batch size:** 32
- **Split:** 80 / 20 train/val, stratified on class

Outputs:
- `model.keras` — trained model
- `training_curves.png` — accuracy & loss plots

---

## How to run

### Requirements
```bash
pip install tensorflow opencv-python scikit-learn matplotlib numpy
```

### Setup

The dataset is **not included in this repository**. Download a subset of
[Fruits 360](https://www.kaggle.com/datasets/moltean/fruits) containing the
three target classes and arrange it as shown in the [Dataset](#dataset)
section above:

```bash
# Expected layout next to wma_05.py
dataset/
├── banana/
├── orange/
└── lemon/
```

Optional — place a test image next to the script. If `test.jpg` is missing,
the script automatically picks one image from the dataset:

```bash
cp some_fruit.jpg test.jpg
```

### Run
```bash
python wma_05.py
```

Steps performed by `main()`:
1. Load and shuffle the dataset
2. Build & compile the CNN, print `model.summary()`
3. Train with EarlyStopping
4. Save `model.keras`
5. Save `training_curves.png`
6. Classify `test.jpg` and print class probabilities

---

## Example output

```
=== Classification Result ===
Image           : test.jpg
Predicted class : orange
Probabilities   :
  banana  : 0.12%
  orange  : 99.74%
  lemon   : 0.14%
```

---

## Project structure

```
WMA_Project_05/
├── wma_05.py             # main script (load, build, train, evaluate)
├── README.md             # this file
├── dataset/              # extracted Fruits 360 subset (not committed)
│   ├── banana/
│   ├── orange/
│   └── lemon/
├── model.keras           # generated after training
├── training_curves.png   # generated after training
└── test.jpg              # optional test image
```

---

## Notes / things I tried

- **Heavier augmentation (rotation 40°, zoom 0.2, channel shift)** → val
  accuracy dropped ~3-5 points. Clean backgrounds mean the model never sees
  the kind of variation augmentation simulates in deployment, so it was just
  adding noise to gradients.
- **Adding a 4th conv block (256 filters)** → marginal improvement, ~3× more
  parameters. Not worth it for 3 classes on this dataset.
- **`class_weight` to handle the lemon imbalance** → stratified split was
  enough; weighting did not change final val accuracy meaningfully.
