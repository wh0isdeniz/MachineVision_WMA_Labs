#!/usr/bin/env python3
"""
LAB4: Optical Flow
Lucas-Kanade Feature Tracking

Usage:
    python lab4_tracking.py --video film.mp4
"""

import argparse
import sys

import cv2
import numpy as np


# ============================================================
# TODO 1: Load video
# ============================================================

def load_video(video_path: str):
    """
    Opens a video file and verifies it was loaded correctly.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {video_path}", file=sys.stderr)
        sys.exit(1)
    return cap


# ============================================================
# TODO 2: Detect good features to track
# ============================================================

def detect_features(gray_frame: np.ndarray):
    """
    Detects feature points using Shi-Tomasi (Good Features to Track).
    """
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    points = cv2.goodFeaturesToTrack(gray_frame, mask=None, **feature_params)
    return points


# ============================================================
# TODO 3: Optical flow (Lucas-Kanade)
# ============================================================

def track_points(prev_gray, curr_gray, prev_points):
    """
    Tracks points between frames using Lucas-Kanade optical flow.
    Returns new_points and status array.
    """
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **lk_params
    )
    return new_points, status


# ============================================================
# TODO 4: Visualization of points
# ============================================================

def draw_points(frame, points):
    """
    Draws circles at current positions of tracked points.
    """
    if points is None:
        return frame
    for pt in points:
        x, y = pt.ravel()
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    return frame


# ============================================================
# TODO 5: Trajectory visualization
# ============================================================

def draw_trajectories(mask, prev_points, curr_points):
    """
    Draws lines from previous to current point positions,
    accumulating trajectories on mask.
    """
    if prev_points is None or curr_points is None:
        return mask
    for prev, curr in zip(prev_points, curr_points):
        x0, y0 = prev.ravel()
        x1, y1 = curr.ravel()
        cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
    return mask


# ============================================================
# Main processing loop
# ============================================================

def process_video(video_path: str):

    cap = load_video(video_path)

    # Read first frame
    ret, first_frame = cap.read()
    if not ret:
        print("ERROR: Cannot read video", file=sys.stderr)
        return

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Detect initial features
    prev_points = detect_features(prev_gray)

    # Mask for trajectories
    trajectory_mask = np.zeros_like(first_frame)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track points
        curr_points, status = track_points(prev_gray, curr_gray, prev_points)

        # TODO 6: Filter only good points using status
        if curr_points is not None and status is not None:
            good_curr = curr_points[status == 1]
            good_prev = prev_points[status == 1]
        else:
            good_curr = None
            good_prev = None

        # If no points remain, re-detect
        if good_curr is None or len(good_curr) == 0:
            prev_points = detect_features(prev_gray)
            trajectory_mask = np.zeros_like(first_frame)  # reset trails
            prev_gray = curr_gray.copy()
            continue

        # Draw trajectories
        trajectory_mask = draw_trajectories(trajectory_mask, good_prev, good_curr)

        # Draw current points
        output = draw_points(frame.copy(), good_curr)

        # Combine frame and trajectories
        output = cv2.add(output, trajectory_mask)

        # TODO 7: Add text showing number of tracked points
        num_points = len(good_curr)
        cv2.putText(
            output,
            f"Tracked points: {num_points}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Optical Flow Tracking", output)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break

        prev_gray = curr_gray.copy()
        prev_points = good_curr.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()

# Main function

def main():
    parser = argparse.ArgumentParser(description="LAB4 - Optical Flow Tracking")
    parser.add_argument("--video", required=True, help="Path to video file")
    args = parser.parse_args()
    process_video(args.video)


if __name__ == "__main__":
    main()