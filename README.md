# MachineVision WMA Labs

Lab projects for the **Machine Vision (WMA)** course — implemented in Python.

---

## Projects

### [WMA_Project_01 – Red Object Detection & Horizontal Tracking](./WMA_Project_01)

Real-time detection of a red object in a video stream using HSV color segmentation and image moments. Tracks the object's horizontal position relative to the frame center and visualizes deviation with an on-screen bar.

**Key techniques:** HSV segmentation · morphological cleaning (OPEN/CLOSE) · image moments · deviation overlay

---

### [WMA_Project_02 – Coin Detection & Tray Localization](./WMA_Project_02)

Detects Polish coins (5 PLN and 5 gr) in static images, localizes an orange tray, and computes the total monetary value of coins both on and off the tray.

**Key techniques:** HoughCircles · HoughLinesP · HSV tray segmentation · ring edge-density filter · radius + colour classification

---

### [WMA_Project_03 – Feature-Based Object Detection & Tracking](./WMA_Project_03)

Detects and tracks an object in static images and video using local feature descriptors, Lowe's ratio test, and homography-based localization. Supports ORB, BRISK, and SIFT.

**Key techniques:** ORB / BRISK / SIFT descriptors · BFMatcher + Lowe's ratio test · RANSAC homography · perspective transform

---


### [WMA_Project_04 – Optical Flow Feature Tracking](./WMA_Project_04)

Tracks feature points across video frames using the Lucas-Kanade optical flow algorithm. Accumulates motion trajectories as colored trails and automatically re-detects points when tracking is lost.

**Key techniques:** Shi-Tomasi corner detection · Lucas-Kanade pyramidal optical flow · status filtering · trajectory accumulation

---


### [WMA_Project_05 – CNN Fruit Classification](./WMA_Project_05)

Classifies images into three fruit classes (banana, orange, lemon) using a custom convolutional neural network trained from scratch on a subset of the Fruits 360 dataset. Includes preprocessing, training with early stopping, and single-image inference.

**Key techniques:** Conv2D blocks (32→64→128) · MaxPooling + Dropout · Adam + categorical crossentropy · EarlyStopping with stratified split · Softmax classification

---

### [WMA_Project_06 – YOLO Object Detection & Fine-Tuning](./WMA_Project_06)

Fine-tunes a YOLOv8n object detector on a custom avocado class (not part of COCO) and compares it against the base model on the same image. Supports inference on images, video files, and live camera streams, plus dataset structure validation before training.

**Key techniques:** Ultralytics YOLOv8n · custom-class fine-tuning · YOLO-format dataset preparation · bounding-box parsing + confidence filtering · base vs fine-tuned comparison

---

### [WMA_Project_07 – Vision Transformer vs CNN](./WMA_Project_07)

Implements a Vision Transformer from scratch and benchmarks it against the LAB5 CNN baseline on the same 3-class fruit dataset across three experiments: reduced data, full data with augmentation, and a pretrained ImageNet backbone fine-tuned end-to-end. The ViT matches the CNN's accuracy with ~32× fewer parameters.

**Key techniques:** 16×16 patch embedding · learnable positional encoding · 4-block multi-head self-attention · MLP with GELU · EfficientNetB0 transfer learning · CNN vs ViT comparison

---
