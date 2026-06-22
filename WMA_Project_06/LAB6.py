#!/usr/bin/env python3
"""
LAB6: YOLO Algorithm
Full starter template (completed version)

Goals:
1. Load the YOLO model.
2. Load an image, video, or camera stream.
3. Perform object detection.
4. Read bounding boxes, classes, and confidence scores.
5. Draw detection results.
6. Prepare data for model fine-tuning.
7. Run YOLO model fine-tuning.
8. Compare the base model and the fine-tuned model.
"""

import argparse
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# TODO 1: Import YOLO
from ultralytics import YOLO


# ============================================================
# TODO 2: Load the base model
# ============================================================

def load_model(model_path: str):
    """
    Loads a YOLO model from file.
    """
    try:
        model = YOLO(model_path)
        print(f"[OK] Model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Could not load model '{model_path}': {e}")
        raise


# ============================================================
# TODO 3: Load image
# ============================================================

def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from file.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image (corrupt or unsupported format): {image_path}")

    return image


# ============================================================
# TODO 4: Load video stream
# ============================================================

def load_stream(source: str):
    """
    Loads a video stream.
    """
    if source == "camera":
        cap = cv2.VideoCapture(0)
        source_desc = "default camera (index 0)"
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Video file not found: {source}")
        cap = cv2.VideoCapture(source)
        source_desc = f"video file '{source}'"

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {source_desc}")

    print(f"[OK] Stream opened: {source_desc}")
    return cap


# ============================================================
# TODO 5: Run detection
# ============================================================

def run_detection(model, image: np.ndarray):
    """
    Runs the YOLO model on a single image / frame.
    """
    results = model(image, verbose=False)
    return results


# ============================================================
# TODO 6: Read data from YOLO results
# ============================================================

def parse_results(results) -> List[Dict[str, Any]]:
    """
    Reads detection data from the model output.
    """
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = xyxy.tolist()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())

            detections.append({
                "box": [x1, y1, x2, y2],
                "class_id": class_id,
                "confidence": confidence,
            })

    return detections


# ============================================================
# TODO 7: Filter detections
# ============================================================

def filter_detections(
    detections: List[Dict[str, Any]],
    confidence_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Filters detections based on the confidence threshold.
    """
    return [
        det for det in detections
        if det["confidence"] >= confidence_threshold
    ]


# ============================================================
# TODO 8: Get class names
# ============================================================

def get_class_names(model) -> Optional[Dict[int, str]]:
    """
    Retrieves class names from the model, if available.
    """
    try:
        names = model.names
        if isinstance(names, dict):
            return names
        if isinstance(names, (list, tuple)):
            return {i: name for i, name in enumerate(names)}
        return None
    except AttributeError:
        return None


# ============================================================
# TODO 9: Draw detections
# ============================================================

def draw_detections(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    class_names: Optional[Dict[int, str]] = None
) -> np.ndarray:
    """
    Draws bounding boxes and val on the image.
    """
    output = image.copy()

    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        class_id = det["class_id"]
        conf = det["confidence"]

        if class_names is not None and class_id in class_names:
            label = class_names[class_id]
        else:
            label = f"id={class_id}"

        color = (0, 255, 0)
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        text = f"{label}: {conf:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        text_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10

        cv2.rectangle(
            output,
            (x1, text_y - text_h - baseline),
            (x1 + text_w, text_y + baseline),
            color,
            -1,
        )
        cv2.putText(
            output,
            text,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
        )

    return output


# ============================================================
# TODO 10: Diagnostic text
# ============================================================

def add_diagnostics(
    image: np.ndarray,
    num_detections: int,
    confidence_threshold: float
) -> np.ndarray:
    """
    Adds diagnostic text to the image.
    """
    output = image.copy()

    lines = [
        f"Detections: {num_detections}",
        f"Conf threshold: {confidence_threshold:.2f}",
    ]

    x, y = 10, 30
    for i, line in enumerate(lines):
        y_pos = y + i * 30

        cv2.putText(
            output, line, (x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4,
        )
        cv2.putText(
            output, line, (x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )

    return output


# ============================================================
# TODO 11: Check training data structure
# ============================================================

def check_dataset_structure(data_path: str) -> bool:
    """
    Checks the dataset structure for YOLO training.
    """
    required_dirs = [
        os.path.join(data_path, "images", "train"),
        os.path.join(data_path, "images", "val"),
        os.path.join(data_path, "labels", "train"),
        os.path.join(data_path, "labels", "val"),
    ]
    yaml_path = os.path.join(data_path, "data.yaml")

    print(f"\n[INFO] Checking dataset structure at: {data_path}")

    all_ok = True

    for d in required_dirs:
        if os.path.isdir(d):
            num_files = len(os.listdir(d))
            print(f"  [OK]      {d}  ({num_files} files)")
        else:
            print(f"  [MISSING] {d}")
            all_ok = False

    if os.path.isfile(yaml_path):
        print(f"  [OK]      {yaml_path}")
    else:
        print(f"  [MISSING] {yaml_path}")
        all_ok = False

    if all_ok:
        print("[OK] Dataset structure looks correct.\n")
    else:
        print("[ERROR] Dataset structure is incomplete.\n")

    return all_ok


# ============================================================
# TODO 12: Annotation data information
# ============================================================

def print_annotation_info():
    """
    Displays information for the student about preparing training data.
    """
    print("=" * 60)
    print("Preparing training data for YOLO")
    print("=" * 60)
    print()
    print("Recommended annotation tools:")
    print("  - LabelImg     (simple desktop tool)")
    print("  - CVAT         (advanced web-based tool)")
    print("  - Makesense.ai (browser, no installation)")
    print("  - Roboflow     (with augmentation and export)")
    print()
    print("YOLO label file format (one line per object):")
    print("  <class_id> <x_center> <y_center> <width> <height>")
    print()
    print("All values are normalized to the range [0, 1].")
    print("Example:")
    print("  0 0.523 0.441 0.210 0.317")
    print()
    print("Expected dataset structure:")
    print("  dataset/")
    print("      images/")
    print("          train/")
    print("          val/")
    print("      val/")
    print("          train/")
    print("          val/")
    print("      data.yaml")
    print()
    print("Example data.yaml:")
    print("  path: dataset")
    print("  train: images/train")
    print("  val: images/val")
    print("  names:")
    print("    0: object")
    print("=" * 60)


# ============================================================
# TODO 13: Fine-tune the model
# ============================================================

def fine_tune_model(
    model,
    data_path: str,
    num_epochs: int = 20,
    image_size: int = 640
):
    """
    Runs YOLO model training / fine-tuning.
    """
    if not check_dataset_structure(data_path):
        raise ValueError(
            f"Invalid dataset structure at '{data_path}'. "
            "Fix the issues above before training."
        )

    yaml_path = os.path.join(data_path, "data.yaml")

    print(f"\n[INFO] Starting fine-tuning")
    print(f"  data:   {yaml_path}")
    print(f"  epochs: {num_epochs}")
    print(f"  imgsz:  {image_size}\n")

    results = model.train(
        data=yaml_path,
        epochs=num_epochs,
        imgsz=image_size,
    )

    save_dir = getattr(results, "save_dir", None)
    if save_dir is not None:
        best_path = os.path.join(str(save_dir), "weights", "best.pt")
    else:
        best_path = "runs/detect/train/weights/best.pt"

    print(f"\n[OK] Training finished.")
    print(f"  Best model: {best_path}")

    return best_path


# ============================================================
# TODO 14: Load fine-tuned model
# ============================================================

def load_fine_tuned_model(model_path: str):
    """
    Loads the model saved after training.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Fine-tuned model not found: {model_path}"
        )

    try:
        model = YOLO(model_path)
        print(f"[OK] Fine-tuned model loaded: {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Could not load fine-tuned model: {e}")
        raise


# ============================================================
# TODO 15: Compare models
# ============================================================

def compare_models(
    base_model,
    fine_tuned_model,
    image: np.ndarray,
    confidence_threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compares the base model and the fine-tuned model on the same image.
    """
    base_results = run_detection(base_model, image)
    base_detections = parse_results(base_results)
    base_detections = filter_detections(base_detections, confidence_threshold)
    base_names = get_class_names(base_model)
    base_image = draw_detections(image, base_detections, base_names)
    base_image = add_diagnostics(base_image, len(base_detections), confidence_threshold)

    ft_results = run_detection(fine_tuned_model, image)
    ft_detections = parse_results(ft_results)
    ft_detections = filter_detections(ft_detections, confidence_threshold)
    ft_names = get_class_names(fine_tuned_model)
    ft_image = draw_detections(image, ft_detections, ft_names)
    ft_image = add_diagnostics(ft_image, len(ft_detections), confidence_threshold)

    print(f"\n[INFO] Comparison:")
    print(f"  Base model        : {len(base_detections)} detections")
    print(f"  Fine-tuned model  : {len(ft_detections)} detections")

    return base_image, ft_image


# ============================================================
# Image processing
# ============================================================

def process_image(model, image_path: str, confidence_threshold: float):
    """
    Processes a single image.
    """
    image = load_image(image_path)

    results = run_detection(model, image)
    detections = parse_results(results)
    detections = filter_detections(detections, confidence_threshold)

    class_names = get_class_names(model)

    output_image = draw_detections(image, detections, class_names)
    output_image = add_diagnostics(output_image, len(detections), confidence_threshold)

    cv2.imshow("Detection result - image", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Video / camera processing
# ============================================================

def process_video(model, source: str, confidence_threshold: float):
    """
    Processes video or camera input.
    """
    cap = load_stream(source)
    class_names = get_class_names(model)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = run_detection(model, frame)
        detections = parse_results(results)
        detections = filter_detections(detections, confidence_threshold)

        output_frame = draw_detections(frame, detections, class_names)
        output_frame = add_diagnostics(output_frame, len(detections), confidence_threshold)

        cv2.imshow("Detection result - video", output_frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# Comparison mode
# ============================================================

def run_comparison(
    base_model,
    fine_tuned_model,
    image_path: str,
    confidence_threshold: float
):
    """
    Runs a comparison of two models on a single image.
    """
    image = load_image(image_path)

    base_image, fine_tuned_image = compare_models(
        base_model,
        fine_tuned_model,
        image,
        confidence_threshold
    )

    cv2.imshow("Base model", base_image)
    cv2.imshow("Fine-tuned model", fine_tuned_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ============================================================
# Main function
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LAB6 - YOLO, full starter template")
    parser.add_argument("--model", required=True, help="Path to the base model, e.g. yolov8n.pt")
    parser.add_argument("--image", help="Path to the image")
    parser.add_argument("--video", help="Path to the video file")
    parser.add_argument("--camera", action="store_true", help="Use camera")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")

    parser.add_argument("--train", action="store_true", help="Run model fine-tuning")
    parser.add_argument("--train-data", help="Path to YOLO training data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--trained-model", help="Path to the fine-tuned model")
    parser.add_argument("--compare", action="store_true", help="Compare base and fine-tuned model")
    parser.add_argument("--show-annotation-help", action="store_true", help="Show annotation data information")

    args = parser.parse_args()

    # TODO 16: Load the base model
    base_model = load_model(args.model)

    # TODO 17: Annotation help
    if args.show_annotation_help:
        print_annotation_info()
        return 0

    # TODO 18: Training mode
    if args.train:
        if not args.train_data:
            print("[ERROR] --train requires --train-data <path to dataset>")
            return 1

        best_path = fine_tune_model(
            base_model,
            args.train_data,
            num_epochs=args.epochs,
            image_size=args.imgsz,
        )
        print(f"\n[OK] Training complete. Best weights at: {best_path}")
        return 0

    # TODO 19: Comparison mode
    if args.compare:
        if not args.trained_model:
            print("[ERROR] --compare requires --trained-model <path to fine-tuned model>")
            return 1
        if not args.image:
            print("[ERROR] --compare requires --image <path to image>")
            return 1

        fine_tuned_model = load_fine_tuned_model(args.trained_model)
        run_comparison(
            base_model,
            fine_tuned_model,
            args.image,
            args.confidence,
        )
        return 0

    # TODO 20: Detection modes
    if args.image:
        process_image(base_model, args.image, args.confidence)
        return 0

    if args.video:
        process_video(base_model, args.video, args.confidence)
        return 0

    if args.camera:
        process_video(base_model, "camera", args.confidence)
        return 0

    print("[ERROR] No mode selected. Provide one of: "
          "--image, --video, --camera, --train, --compare, --show-annotation-help")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())