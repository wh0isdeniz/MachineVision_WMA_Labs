# WMA Project 07 — Vision Transformer vs CNN

A from-scratch **Vision Transformer (ViT)** trained side-by-side with the
CNN from [WMA Project 05](../WMA_Project_05/), plus a pretrained ImageNet
backbone fine-tuned for comparison. Three experiments quantify the gap
between the two architectures under different conditions: small data,
full data + augmentation, and transfer learning.

Built with TensorFlow / Keras.

---

## Task

1. Load the LAB5 fruit dataset
2. Rebuild the LAB5 CNN baseline
3. Implement a Vision Transformer from scratch — patch embedding,
   learnable positional encoding, multi-head self-attention blocks,
   classification head
4. Train both models under two regimes:
   - **Experiment A** — 40 % of the training set (small-data)
   - **Experiment B** — full training set with data augmentation
5. **Experiment C** — fine-tune a pretrained ImageNet backbone end-to-end
6. Compare accuracy, loss, training time, and parameter count

---

## Dataset

Same 3-class fruit dataset as WMA Project 05.

| Class  | Images |
|--------|--------|
| banana | 1 210  |
| lemon  | 492    |
| orange | 1 195  |
| **Total** | **2 897** |

Split **70 / 15 / 15** into train / val / test (~2 027 / 433 / 437
images), resized to 224×224. The script auto-creates the split on the
first run.

The dataset is **not committed**. Drop it as a flat folder of class
subdirectories and point `data_path` in `main()` to it:

```
dataset/
├── banana/
├── lemon/
└── orange/
```

---

## Architectures

### CNN baseline (from LAB5)

Three Conv blocks + dense head: `32→32 → 64→64 → 128` channels, ReLU,
MaxPool between blocks, Dropout(0.3) before the final `Dense(3)` softmax.
**~12.98 M parameters.**

### Vision Transformer (from scratch)

| Component            | Choice                                                |
|----------------------|-------------------------------------------------------|
| Patch size           | 16 × 16                                               |
| Patches per image    | 196 (14 × 14, since 224 / 16 = 14)                    |
| Projection dim       | 64                                                    |
| Positional encoding  | Learnable (`Embedding` layer)                         |
| Transformer blocks   | 4                                                     |
| Attention heads      | 4                                                     |
| MLP inside block     | `Dense(128) → GELU → Dense(64)`, with dropout         |
| Classification head  | `GlobalAvgPool → Dense(3)` softmax                    |
| **Parameters**       | **~403 K (~32× fewer than the CNN)**                  |

---

## Experiments

|  | Training data | Augmentation | Models trained |
|---|---|---|---|
| **A — Reduced data**     | 40 % of train | none | CNN, ViT (from scratch) |
| **B — Full + augment**   | 100 % of train | rotation ±10 %, horizontal flip, zoom ±10 % | CNN, ViT (from scratch) |
| **C — Pretrained**       | 100 % of train | none | EfficientNetB0 (ImageNet) with a fresh `Dense(3)` head, fine-tuned end-to-end at lr = 1e-4 |

**Note on Experiment C.** The task asked for a pretrained **ViT-B/16**
via TensorFlow Hub. When the Hub model failed to load in the
environment, the script fell back to a pretrained **EfficientNetB0** —
same idea (ImageNet pretraining + transfer), different backbone.

---

## How to run

### Install

```bash
pip install tensorflow matplotlib numpy
```

### Point the script at your dataset

Open `LAB7.py` and update the path in `main()`:

```python
train_ds, val_ds, test_ds = load_datasets("/path/to/dataset")
```

### Run all three experiments

```bash
python LAB7.py
```

The script writes into `outputs/`:
- `history_A_CNN.png`, `history_A_ViT.png` — Experiment A curves
- `history_B_CNN.png`, `history_B_ViT.png` — Experiment B curves
- `history_C_Pretrained.png` — Experiment C curves
- `sample_predictions.png` — 9 test predictions from the best model
- `*.weights.h5` — saved weights for each trained model
- A printed comparison table

---

## Results

Final metrics on the held-out test set:

| Model      | Experiment        | Accuracy | Loss   | Time (s) | Parameters  |
|------------|-------------------|----------|--------|----------|-------------|
| CNN        | A — 40 % data     | 0.9954   | 0.0397 | 50.8     | 12 984 995  |
| ViT        | A — 40 % data     | 0.9931   | 0.0155 | 44.2     | 403 395     |
| CNN        | B — full + aug    | 1.0000   | 0.0000 | 92.4     | 12 984 995  |
| ViT        | B — full + aug    | 1.0000   | 0.0016 | 51.7     | 403 395     |
| Pretrained | C — EfficientNetB0| 1.0000   | 0.0003 | 197.1    | 4 053 414   |

### Experiment A — small data (40 % of train)

CNN accuracy / loss:

<img width="1320" height="480" alt="history_A_CNN" src="https://github.com/user-attachments/assets/f4835452-ceff-43ff-93bd-58f42ba450d9" />

ViT accuracy / loss:

<img width="1320" height="480" alt="history_A_ViT" src="https://github.com/user-attachments/assets/0c90967c-34b0-49e4-845a-49781cae756e" />

### Experiment B — full data + augmentation

<img width="1320" height="480" alt="history_B_CNN" src="https://github.com/user-attachments/assets/480e8434-b11c-42cd-b161-cd4aef378aac" />

<img width="1320" height="480" alt="history_B_ViT" src="https://github.com/user-attachments/assets/78fd9bc8-02c7-45b8-849e-ab703e0b0631" />

### Experiment C — pretrained backbone

<img width="1320" height="480" alt="history_C_Pretrained" src="https://github.com/user-attachments/assets/f56b31ee-45ad-494f-8e99-6510132cb101" />

### Sample predictions (best model, test set)

<img width="1080" height="1080" alt="sample_predictions" src="https://github.com/user-attachments/assets/2448fd98-4600-425a-848d-fc971cadc27c" />

---

## Project structure

```
WMA_Project_07/
├── LAB7.py                 # main script (3 experiments)
├── README.md               
└── results/
    ├── history_A_CNN.png
    ├── history_A_ViT.png
    ├── history_B_CNN.png
    ├── history_B_ViT.png
    ├── history_C_Pretrained.png
    └── sample_predictions.png
```

The dataset, `outputs/` folder (saved weights and generated plots), and
TensorFlow caches are produced locally and excluded via `.gitignore`.

---

## Notes / things I tried

- **The dataset is too easy.** Both architectures saturate at ~100 % on
  the full set, so the interesting signal is in *how* they converge, not
  the peak number. The CNN crosses 99 % val accuracy after epoch 1; the
  ViT starts lower and climbs steadily but matches the CNN by epoch 10
  — with **~32× fewer parameters**.
- **Why CNNs dominate the small-data regime.** Conv layers come with
  locality and translation invariance baked in, so the model doesn't
  have to learn those from data. The ViT has to learn all spatial
  structure from scratch through attention, which is exactly why it's
  data-hungry and why pretraining helps so much.
- **Augmentation barely moved the needle.** On a dataset that's already
  almost solved without it, flip + rotation + zoom mostly smoothed the
  ViT's curve. On a harder dataset (or with the ViT only), the effect
  would be much larger.
- **The "pretrained" run is technically EfficientNetB0, not a ViT.** The
  Hub ViT-B/16 weights didn't load in the test environment, so the
  fallback path fired. The takeaway holds either way: ImageNet
  pretraining lets you hit perfect test accuracy in ~5 epochs even with
  end-to-end fine-tuning, at fewer parameters than the CNN baseline.
- **Single-script setup, no argparse.** The whole run is `python
  LAB7.py`. Dataset path is hard-coded in `main()`; edit it
  before running. Everything else — splitting, training, plotting,
  weight saving — is automatic.
