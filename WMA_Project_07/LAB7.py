#!/usr/bin/env python3
"""
LAB7: Vision Transformers

Goals:
1. Load data
2. Build CNN (LAB5)
3. Build Vision Transformer
4. Train models
5. Perform experiments
6. Compare results
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import shutil
import random

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 3

# --- extra hyper-parameters for the Vision Transformer ---
PATCH_SIZE = 16                                   # 224 / 16 = 14 -> 196 patches
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2       # 196
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_LAYERS = 4
MLP_UNITS = [PROJECTION_DIM * 2, PROJECTION_DIM]

OUTPUT_DIR = "outputs"
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# TODO 1: Load datasets
# ============================================================

def load_datasets(data_path):
    """
    TODO:
    - Load train/val/test datasets
    - Resize to 224x224
    - Normalize data
    """
    # If the dataset is a flat folder of classes, split it once into
    # train/val/test (70/15/15) inside "<data_path>_split".
    split_dir = data_path.rstrip("/") + "_split"

    if not os.path.isdir(split_dir):
        classes = sorted(
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d))
        )
        for subset in ("train", "val", "test"):
            for c in classes:
                os.makedirs(os.path.join(split_dir, subset, c), exist_ok=True)

        for c in classes:
            files = [
                f for f in os.listdir(os.path.join(data_path, c))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            random.shuffle(files)
            n = len(files)
            n_train, n_val = int(n * 0.70), int(n * 0.15)
            buckets = {
                "train": files[:n_train],
                "val": files[n_train:n_train + n_val],
                "test": files[n_train + n_val:],
            }
            for subset, names in buckets.items():
                for name in names:
                    shutil.copy(
                        os.path.join(data_path, c, name),
                        os.path.join(split_dir, subset, c, name),
                    )

    def _load(subset, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            os.path.join(split_dir, subset),
            labels="inferred",
            label_mode="int",
            image_size=(IMG_SIZE, IMG_SIZE),   # resize to 224x224
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            seed=SEED,
        )

    train_ds = _load("train", True)
    val_ds = _load("val", False)
    test_ds = _load("test", False)

    # Normalize pixel values to [0, 1]
    norm = layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.map(lambda x, y: (norm(x), y), num_parallel_calls=AUTOTUNE)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    test_ds = test_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


# ============================================================
# TODO 2: Augmentation
# ============================================================

def get_augmentation():
    """
    TODO:
    - Add augmentation:
        * flip
        * rotation
        * zoom
    """
    return models.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )


# ============================================================
# TODO 3: CNN from LAB5
# ============================================================

def build_cnn(augment=False):
    """
    TODO:
    - Copy or recreate CNN from LAB5
    """
    # Exact CNN architecture from LAB5: three conv blocks (the first two with
    # two stacked conv layers each), then a dense classifier head.
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    if augment:
        x = get_augmentation()(x)

    # Block 1
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # Block 2
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # Block 3
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Classifier head
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    return models.Model(inputs, outputs, name="CNN_LAB5")


# ============================================================
# TODO 4: Patch extraction
# ============================================================

def create_patches(images, patch_size=PATCH_SIZE):
    """
    TODO:
    - Split image into patches
    """
    return Patches(patch_size)(images)


class Patches(layers.Layer):
    """Split a batch of images into flattened patches."""

    def __init__(self, patch_size=PATCH_SIZE):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg


# ============================================================
# TODO 5: Patch encoding
# ============================================================

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches=NUM_PATCHES, projection_dim=PROJECTION_DIM):
        super().__init__()
        # embedding + positional encoding
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(projection_dim)               # embedding
        self.position_embedding = layers.Embedding(                  # positional
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patches) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {"num_patches": self.num_patches,
             "projection_dim": self.projection_dim}
        )
        return cfg


# ============================================================
# TODO 6: Transformer block
# ============================================================

def transformer_block(x):
    """
    TODO:
    - LayerNorm
    - MultiHeadAttention
    - MLP
    - Residual connections
    """
    # --- Attention sub-block ---
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    attn = layers.MultiHeadAttention(
        num_heads=NUM_HEADS, key_dim=PROJECTION_DIM, dropout=0.1
    )(x1, x1)
    x2 = layers.Add()([attn, x])                 # residual connection

    # --- MLP sub-block ---
    x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
    for units in MLP_UNITS:
        x3 = layers.Dense(units, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
    out = layers.Add()([x3, x2])                 # residual connection
    return out


# ============================================================
# TODO 7: Vision Transformer
# ============================================================

def build_vit(augment=False):
    """
    TODO:
    - Input
    - Patch extraction
    - Encoder
    - Transformer blocks
    - Classifier
    """
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))     # Input
    x = inputs
    if augment:
        x = get_augmentation()(x)

    patches = Patches(PATCH_SIZE)(x)                         # Patch extraction
    encoded = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(patches)   # Encoder

    for _ in range(TRANSFORMER_LAYERS):                      # Transformer blocks
        encoded = transformer_block(encoded)

    # Classifier
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.3)(representation)
    features = layers.Dense(128, activation=tf.nn.gelu)(representation)
    features = layers.Dropout(0.3)(features)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(features)

    return models.Model(inputs, outputs, name="VisionTransformer")


# ============================================================
# TODO 8: Compilation
# ============================================================

def compile_model(model, lr=1e-3):
    """
    TODO:
    - optimizer
    - loss
    - accuracy
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),   # optimizer
        loss="sparse_categorical_crossentropy",                 # loss
        metrics=["accuracy"],                                   # accuracy
    )
    return model


# ============================================================
# TODO 9: Training
# ============================================================

def train_model(model, train_ds, val_ds, epochs=EPOCHS):
    """
    TODO:
    - measure training time
    """
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    end = time.time()
    return history, end - start


# ============================================================
# TODO 10: Evaluation
# ============================================================

def evaluate_model(model, test_ds):
    """
    TODO:
    - accuracy
    - loss
    """
    return model.evaluate(test_ds)


# ============================================================
# TODO 11: Plots
# ============================================================

def plot_history(history, title="model", fname=None):
    """
    TODO:
    - accuracy
    - loss
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    h = history.history
    epochs_range = range(1, len(h["accuracy"]) + 1)

    plt.figure(figsize=(11, 4))
    # accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, h["accuracy"], "o-", label="train")
    plt.plot(epochs_range, h["val_accuracy"], "s-", label="val")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.grid(True)
    # loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, h["loss"], "o-", label="train")
    plt.plot(epochs_range, h["val_loss"], "s-", label="val")
    plt.title(f"{title} - Loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    if fname is None:
        fname = f"history_{title}.png"
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[plot] saved {path}")


# ------------------------------------------------------------
# Small helpers used by the experiments below
# ------------------------------------------------------------

def _subset(ds, fraction):
    """Keep only `fraction` of the batches (used for the reduced-data exp.)."""
    n = ds.cardinality().numpy()
    return ds.take(max(1, int(n * fraction)))


def _run(model_builder, train_ds, val_ds, test_ds, name,
         augment=False, lr=1e-3, epochs=EPOCHS):
    """Build -> compile -> train -> evaluate one model; return a result dict."""
    print(f"\n--- {name} ---")
    model = model_builder(augment=augment)
    compile_model(model, lr=lr)
    history, t = train_model(model, train_ds, val_ds, epochs=epochs)
    loss, acc = evaluate_model(model, test_ds)
    params = model.count_params()
    print(f"[{name}] acc={acc:.4f} loss={loss:.4f} time={t:.1f}s params={params:,}")
    plot_history(history, title=name)
    return {"name": name, "acc": float(acc), "loss": float(loss),
            "time": float(t), "params": int(params), "model": model}


# ============================================================
# TODO 12: Experiment A
# ============================================================

def experiment_small_data(train_ds, val_ds, test_ds, fraction=0.4, epochs=EPOCHS):
    """
    TODO:
    - reduce dataset
    - CNN vs ViT
    """
    print("\n===== EXPERIMENT A: reduced dataset "
          f"({int(fraction*100)}%) =====")
    small = _subset(train_ds, fraction)                 # reduce dataset
    results = []
    results.append(_run(build_cnn, small, val_ds, test_ds,
                        "A_CNN", epochs=epochs))         # CNN
    results.append(_run(build_vit, small, val_ds, test_ds,
                        "A_ViT", epochs=epochs))         # ViT
    return results


# ============================================================
# TODO 13: Experiment B
# ============================================================

def experiment_full_aug(train_ds, val_ds, test_ds, epochs=EPOCHS):
    """
    TODO:
    - augmentation
    """
    print("\n===== EXPERIMENT B: full dataset + augmentation =====")
    results = []
    results.append(_run(build_cnn, train_ds, val_ds, test_ds,
                        "B_CNN", augment=True, epochs=epochs))   # CNN + aug
    results.append(_run(build_vit, train_ds, val_ds, test_ds,
                        "B_ViT", augment=True, epochs=epochs))   # ViT + aug
    return results


# ============================================================
# TODO 14: Experiment C (pretrained)
# ============================================================

def experiment_pretrained(train_ds, val_ds, test_ds, epochs=5):
    """
    TODO:
    - load pretrained ViT
    - modify classifier
    - fine-tuning
    """
    print("\n===== EXPERIMENT C: pretrained Vision Transformer =====")

    def build_pretrained(augment=False):
        # --- load pretrained ViT (ImageNet) via tensorflow_hub ---
        try:
            import tensorflow_hub as hub
            hub_url = "https://tfhub.dev/sayakpaul/vit_b16_fe/1"
            inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
            backbone = hub.KerasLayer(hub_url, trainable=True)  # fine-tuning
            x = backbone(inputs)
            x = layers.Dropout(0.3)(x)
            # modify the classifier for our number of classes
            outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
            print("[exp C] Loaded pretrained ViT from TF-Hub.")
            return models.Model(inputs, outputs, name="Pretrained_ViT")
        except Exception as e:
            # Fallback: pretrained CNN backbone (works without TF-Hub)
            print(f"[exp C] TF-Hub ViT unavailable ({type(e).__name__}); "
                  f"using a pretrained EfficientNet backbone instead.")
            base = tf.keras.applications.EfficientNetB0(
                include_top=False, weights="imagenet",
                input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
            base.trainable = True
            inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
            x = layers.Rescaling(255.0)(inputs)   # undo our 0-1 scaling
            x = base(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
            return models.Model(inputs, outputs, name="Pretrained_EffNet")

    # fine-tuning with a small learning rate
    return [_run(build_pretrained, train_ds, val_ds, test_ds,
                 "C_Pretrained", lr=1e-4, epochs=epochs)]


# ------------------------------------------------------------
# Results table + sample predictions (presentation of results)
# ------------------------------------------------------------

def print_results_table(results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    header = f"{'Model':14} {'Acc':>8} {'Loss':>9} {'Time(s)':>9} {'Params':>13}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(f"{r['name']:14} {r['acc']:>8.4f} {r['loss']:>9.4f} "
                     f"{r['time']:>9.1f} {r['params']:>13,}")
    table = "\n".join(lines)
    print("\n" + table)
    with open(os.path.join(OUTPUT_DIR, "results_table.txt"), "w") as f:
        f.write(table + "\n")


def plot_sample_predictions(model, test_ds, class_names, fname="sample_predictions.png"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    imgs, lbls = [], []
    for bx, by in test_ds.take(8):
        imgs.append(bx.numpy()); lbls.append(by.numpy())
    images = np.concatenate(imgs); labels = np.concatenate(lbls)
    idx = np.random.default_rng(SEED).permutation(len(images))[:9]
    images, labels = images[idx], labels[idx]
    preds = np.argmax(model.predict(images, verbose=0), axis=1)

    plt.figure(figsize=(9, 9))
    for i in range(min(9, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        tc, pc = class_names[int(labels[i])], class_names[int(preds[i])]
        plt.title(f"pred: {pc}\ntrue: {tc}",
                  color="green" if tc == pc else "red", fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=120); plt.close()
    print(f"[plot] saved {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    # TODO:
    # - load datasets
    # - run experiments
    # - compare results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("TensorFlow:", tf.__version__)
    print("GPU:", tf.config.list_physical_devices("GPU"))

    # load datasets
    train_ds, val_ds, test_ds = load_datasets("/Users/denizerdemozkan/Documents/PythonProject/dataset")
    class_names = ["banana", "lemon", "orange"]  # alphabetical (folder order)

    # run experiments
    results = []
    results += experiment_small_data(train_ds, val_ds, test_ds, fraction=0.4)
    results += experiment_full_aug(train_ds, val_ds, test_ds)
    results += experiment_pretrained(train_ds, val_ds, test_ds, epochs=5)

    # compare results
    print_results_table(results)

    # sample predictions from the best model + save weights
    best = max(results, key=lambda r: r["acc"])
    print(f"\nBest model: {best['name']} (acc={best['acc']:.4f})")
    plot_sample_predictions(best["model"], test_ds, class_names)

    for r in results:
        r["model"].save_weights(os.path.join(OUTPUT_DIR, f"{r['name']}.weights.h5"))
    print(f"Saved weights and plots into ./{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
