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
