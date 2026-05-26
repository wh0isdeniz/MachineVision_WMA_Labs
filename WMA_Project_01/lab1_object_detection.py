import argparse
import sys
import cv2
import numpy as np


def build_red_mask(bgr: np.ndarray) -> np.ndarray:
    """Red color segmentation in HSV + mask cleaning (OPEN and CLOSE).
    Returns 0/255 mask."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 150, 80], dtype=np.uint8)
    upper_red1 = np.array([8, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([172, 150, 80], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open,  iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    return mask


def object_from_moments(
    mask_0_255: np.ndarray,
    min_area: float = 200.0,
    max_area: float = 20000.0,
):
    """
    Detect the most likely red object using image moments (no contours).
    Returns: (cx, cy), area_px, radius_px  or  (None, None, None)
    """
    m = cv2.moments(mask_0_255, binaryImage=True)
    area = m["m00"]

    if area < min_area or area > max_area:
        return None, None, None

    cx = int(m["m10"] / area)
    cy = int(m["m01"] / area)
    radius = int(np.sqrt(area / np.pi))

    return (cx, cy), area, radius


def draw_deviation_bars(
    img: np.ndarray,
    cx: int,
    width: int,
    y: int = 90,
    bar_height: int = 16,
    margin: int = 10,
):
    """Draw bars showing left/right deviation from the image center."""
    center_x = width // 2
    x1, x2 = margin, width - margin
    bar_y1, bar_y2 = y, y + bar_height

    cv2.rectangle(img, (x1, bar_y1), (x2, bar_y2), (255, 255, 255), 1)
    cv2.line(img, (center_x, bar_y1 - 4), (center_x, bar_y2 + 4), (255, 255, 255), 1)

    cx_clamped = max(x1, min(cx, x2))
    if cx_clamped < center_x:
        cv2.rectangle(img, (cx_clamped, bar_y1), (center_x, bar_y2), (255, 0, 0), -1)
    elif cx_clamped > center_x:
        cv2.rectangle(img, (center_x, bar_y1), (cx_clamped, bar_y2), (0, 255, 0), -1)

    deviation = cx - center_x
    direction = "LEFT" if deviation < 0 else "RIGHT" if deviation > 0 else "CENTER"
    cv2.putText(
        img,
        f"Deviation: {deviation:+d} px ({direction})",
        (margin, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Lab 1 – Red object detection and horizontal tracking"
    )
    p.add_argument("--video", required=True, help="Path to input video file")
    p.add_argument(
        "--min-area",
        type=float,
        default=120.0,
        help="Minimum segmented area to accept object detection",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {args.video}")
        sys.exit(1)

    cv2.namedWindow("Detection (Original | Mask)", cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        vis = frame.copy()
        h, w = vis.shape[:2]
        frame_center_x = w // 2

        mask = build_red_mask(frame)

        obj_center, area, radius = object_from_moments(
            mask, min_area=args.min_area, max_area=20000.0
        )

        cv2.line(vis, (frame_center_x, 0), (frame_center_x, h), (255, 255, 255), 1)

        if obj_center is not None:
            cx, cy = obj_center

            cv2.circle(vis, (cx, cy), radius, (0, 255, 255), 2)
            cv2.circle(vis, (cx, cy), 4, (0, 0, 255), -1)
            cv2.line(vis, (frame_center_x, cy), (cx, cy), (0, 255, 255), 2)

            deviation = cx - frame_center_x
            cv2.putText(vis, f"Object center: ({cx}, {cy})",            (10, 30),     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"Area: {area:.0f} px",                    (10, 55),     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"Radius: {radius} px",                    (10, 80),     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(vis, f"Horizontal deviation: {deviation:+d} px",(10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            draw_deviation_bars(vis, cx, w, y=100)
        else:
            cv2.putText(vis, "No red object detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if mask_bgr.shape[:2] != vis.shape[:2]:
            mask_bgr = cv2.resize(mask_bgr, (w, h))

        combined = np.hstack((vis, mask_bgr))
        cv2.imshow("Detection (Original | Mask)", combined)

        key = cv2.waitKey(20) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()