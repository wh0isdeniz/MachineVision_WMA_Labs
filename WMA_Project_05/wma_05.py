#!/usr/bin/env python3
"""
LAB5: Convolutional Neural Networks
Completed solution

Task:
- Classify three fruits: banana, orange, lemon
"""

import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ============================================================
# TODO 1: Load dataset
# ============================================================

def load_data(dataset_path):
    X, y_raw = [], []
    label_map = {"banana": 0, "orange": 1, "lemon": 2}

    for class_name in label_map.keys():
        class_path = os.path.join(dataset_path, class_name)

        if not os.path.exists(class_path):
            print(f"Warning: folder '{class_path}' not found, skipping.")
            continue

        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                continue

            img = cv2.resize(img, (128, 128))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0

            X.append(img)
            y_raw.append(label_map[class_name])

    X = np.array(X, dtype=np.float32)
    y_raw = np.array(y_raw, dtype=np.int32)

    # Shuffle before one-hot encoding
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X = X[indices]
    y_raw = y_raw[indices]

    y = to_categorical(y_raw, num_classes=3)

    print(f"Loaded {len(X)} images total.")
    return X, y


# ============================================================
# TODO 2: Data augmentation
# ============================================================

def create_augmentation():
    """
Data augmentation pipeline.
Rotation, flipping, small shifts and mild brightness changes are applied.
Heavy augmentation was avoided — Fruits 360 has clean white backgrounds
and aggressive transforms hurt validation performance in practice.
"""
    augmentation = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        zoom_range=0.05,
        brightness_range=[0.9, 1.1]
    )
    return augmentation


# ============================================================
# TODO 3: Build CNN model
# ============================================================

def build_model(input_shape, num_classes):
    """
    3 Conv2D blocks with increasing filters (32->64->128) extract features
    from edges to shapes. ReLU handles non-linearity, MaxPooling reduces
    dimensions, Dropout(0.3) prevents overfitting. Dense(128) + Softmax
    output gives class probabilities for the 3 fruit classes.
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


# ============================================================
# TODO 4: Compile model
# ============================================================

def compile_model(model):
    """
    - Loss: categorical_crossentropy - standard for multi-class with one-hot labels.
    - Optimizer: Adam with learning rate 0.001.
    - Metric: accuracy.
    """
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


# ============================================================
# TODO 5: Train model
# ============================================================

def train_model(model, X_train, y_train, augmentation=None):
    """
30 epochs with EarlyStopping (patience=5), batch size 32,
20% stratified validation split. Augmentation skipped — Fruits 360
studio images with white backgrounds caused val loss instability.
"""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42,
        stratify=np.argmax(y_train, axis=1)
    )

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    history = model.fit(
        X_tr, y_tr,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )

    return history


# ============================================================
# TODO 6: Test classification
# ============================================================

def classify_image(model, image_path):
    class_names = ["banana", "orange", "lemon"]

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not load image '{image_path}'")
        return

    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    probs = model.predict(img)[0]
    predicted_idx = np.argmax(probs)
    predicted_class = class_names[predicted_idx]

    print(f"\n=== Classification Result ===")
    print(f"Image           : {image_path}")
    print(f"Predicted class : {predicted_class}")
    print(f"Probabilities   :")
    for name, prob in zip(class_names, probs):
        print(f"  {name:8s}: {prob * 100:.2f}%")

    return predicted_class, probs


# ============================================================
# TODO 7: Plot training curves
# ============================================================

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history['accuracy'], label='Train accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val accuracy')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].plot(history.history['loss'], label='Train loss')
    axes[1].plot(history.history['val_loss'], label='Val loss')
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()
    print("Training curves saved to training_curves.png")


# ============================================================
# Main function
# ============================================================

def main():

    dataset_path = "dataset"
    input_shape = (128, 128, 3)
    num_classes = 3

    # Load data
    X, y = load_data(dataset_path)

    # Augmentation (defined but not applied in training —
    # Fruits 360 white-background images perform best without augmentation)
    augmentation = create_augmentation()

    # Build model
    model = build_model(input_shape, num_classes)

    # Compile model
    compile_model(model)

    # Show model structure
    model.summary()

    # Train model
    history = train_model(model, X, y, augmentation=augmentation)

    # Save model and plot curves
    model.save("model.keras")
    print("Model saved to model.keras")

    plot_history(history)


    test_image_path = "test.jpg"
    if not os.path.exists(test_image_path):
        all_images = glob.glob("dataset/**/*.jpg", recursive=True)
        if all_images:
            test_image_path = all_images[0]
            print(f"\ntest.jpg not found, picked from dataset: {test_image_path}")
        else:
            print("No picture for test.")
            return

    classify_image(model, test_image_path)


if __name__ == "__main__":
    main()