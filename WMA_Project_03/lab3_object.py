import argparse
import sys
from typing import Optional, Tuple

import cv2
import numpy as np


# ============================================================
# TODO 1: Create detector and matcher
# ============================================================

def create_detector(method_name: str = "ORB"):
    """
    Creates a feature detector / descriptor extractor.
    Supports ORB, BRISK, and SIFT.
    """
    method_name = method_name.upper()
    if method_name == "ORB":
        return cv2.ORB_create(nfeatures=5000)
    elif method_name == "BRISK":
        return cv2.BRISK_create()
    elif method_name == "SIFT":
        return cv2.SIFT_create()
    else:
        print(f"WARNING: Unknown method '{method_name}', falling back to ORB.", file=sys.stderr)
        return cv2.ORB_create(nfeatures=2000)


def create_matcher(method_name: str = "ORB"):
    """
    Creates a matcher compatible with the descriptor type.
    - ORB / BRISK  → BFMatcher with NORM_HAMMING
    - SIFT         → BFMatcher with NORM_L2
    """
    method_name = method_name.upper()
    if method_name in ("ORB", "BRISK"):
        return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:  # SIFT
        return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


# ============================================================
# TODO 2: Compute keypoints and descriptors
# ============================================================

def compute_features(
    gray_image: np.ndarray,
    detector
) -> Tuple[list, Optional[np.ndarray]]:
    """
    Detects keypoints and computes descriptors using detectAndCompute.
    Returns (keypoints, descriptors). descriptors may be None if none found.
    """
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)
    return keypoints, descriptors


# ============================================================
# TODO 3: Match descriptors
# ============================================================

def match_descriptors(
    reference_descriptors: np.ndarray,
    image_descriptors: np.ndarray,
    matcher,
    ratio: float = 0.80
):
    """
    Matches descriptors using kNN (k=2) and Lowe's ratio test.
    Returns a list of good DMatch objects.
    """
    # Guard: need at least 2 descriptors on each side for knnMatch with k=2
    if reference_descriptors is None or image_descriptors is None:
        return []
    if len(reference_descriptors) < 2 or len(image_descriptors) < 2:
        return []

    raw_matches = matcher.knnMatch(reference_descriptors, image_descriptors, k=2)

    good_matches = []
    for pair in raw_matches:
        # knnMatch can occasionally return only 1 neighbour
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    return good_matches


# TODO 4: Localize the object

MIN_MATCH_COUNT = 8

def localize_object(
    reference_keypoints,
    image_keypoints,
    matches,
    reference_shape
):
    """
    Localises the object in the scene image using homography.

    Returns (corners, inlier_mask) where corners is a (4,1,2) float32 array
    of the projected reference-image corners, or (None, None) on failure.
    """
    if len(matches) < MIN_MATCH_COUNT:
        return None, None

    src_pts = np.float32(
        [reference_keypoints[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [image_keypoints[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)

    if H is None or mask is None:
        return None, None


    det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
    if det < 0.01 or det > 100:
        return None, None

    inlier_count = int(mask.sum())
    inlier_ratio = inlier_count / len(matches)
    if inlier_count < 4 or inlier_ratio < 0.20:
        return None, None

    h, w = reference_shape[:2]
    ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    scene_corners = cv2.perspectiveTransform(ref_corners, H)


    pts = scene_corners.reshape(4, 2)
    area = cv2.contourArea(pts)
    if area < 100:
        return None, None

    return scene_corners, mask


# ============================================================
# TODO 5: Visualize the result
# ============================================================

def draw_object(image_bgr: np.ndarray, corners) -> np.ndarray:
    """
    Draws a green polygon around the detected object.
    If corners is None the original image is returned unchanged.
    """
    result = image_bgr.copy()
    if corners is None:
        return result

    # corners shape: (4, 1, 2) — convert to int for drawing
    pts = np.int32(corners)
    cv2.polylines(result, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
    return result


def draw_matches(
    reference_image,
    reference_keypoints,
    test_image,
    test_keypoints,
    matches,
    max_count: int = 50
):
    """
    Creates a side-by-side visualisation of the best matched keypoints.
    """
    # Limit the number of displayed matches
    display_matches = sorted(matches, key=lambda m: m.distance)[:max_count]

    match_img = cv2.drawMatches(
        reference_image, reference_keypoints,
        test_image,      test_keypoints,
        display_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    return match_img


def resize_to_fit(img, max_w=800, max_h=600):
    """Scales image down to fit within max_w x max_h, preserving aspect ratio."""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _put_text(img, text, org, color=(0, 255, 0), scale=1.4):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, org, font, scale,     (0, 0, 0), 6, cv2.LINE_AA)   # shadow
    cv2.putText(img, text, org, font, scale,     color,     3, cv2.LINE_AA)

def process_image(reference_path: str, test_image_path: str, method: str):
    """
    Loads reference + test image, computes descriptors, finds the object,
    and displays three windows: reference, detected object, matched keypoints.
    """

    reference_bgr = cv2.imread(reference_path)
    test_bgr      = cv2.imread(test_image_path)

    if reference_bgr is None:
        print(f"ERROR: Cannot open reference image: {reference_path}", file=sys.stderr)
        return
    if test_bgr is None:
        print(f"ERROR: Cannot open test image: {test_image_path}", file=sys.stderr)
        return

    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    test_gray      = cv2.cvtColor(test_bgr,      cv2.COLOR_BGR2GRAY)

    detector = create_detector(method)
    matcher  = create_matcher(method)

    reference_keypoints, reference_descriptors = compute_features(reference_gray, detector)
    test_keypoints,      test_descriptors      = compute_features(test_gray,      detector)

    good_matches = match_descriptors(reference_descriptors, test_descriptors, matcher)

    corners, inlier_mask = localize_object(
        reference_keypoints,
        test_keypoints,
        good_matches,
        reference_gray.shape
    )

    result_image  = draw_object(test_bgr, corners)
    matches_image = draw_matches(
        reference_bgr, reference_keypoints,
        test_bgr,      test_keypoints,
        good_matches
    )

    # TODO 6: overlay text with match count and detection status
    object_found = corners is not None
    status_text  = "OBJECT FOUND" if object_found else "OBJECT NOT FOUND"
    status_color = (0, 255, 0)    if object_found else (0, 0, 255)

    _put_text(result_image,  f"Matches: {len(good_matches)}",  (10, 45))
    _put_text(result_image,  status_text,                      (10, 105), status_color, scale=2.2)
    _put_text(matches_image, f"Good matches: {len(good_matches)}", (10, 45))

    cv2.imshow("Reference image",    resize_to_fit(reference_bgr))
    cv2.imshow("Detected object",    resize_to_fit(result_image))
    cv2.imshow("Matched keypoints",  resize_to_fit(matches_image, max_w=1400, max_h=600))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(reference_path: str, video_path: str, method: str):
    """
    For each video frame: detects features, matches with reference,
    and draws the object outline only when reliably detected.
    """

    reference_bgr = cv2.imread(reference_path)
    if reference_bgr is None:
        print(f"ERROR: Cannot open reference image: {reference_path}", file=sys.stderr)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {video_path}", file=sys.stderr)
        return

    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)

    detector = create_detector(method)
    matcher  = create_matcher(method)

    reference_keypoints, reference_descriptors = compute_features(reference_gray, detector)

    fps   = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps and fps > 1 else 20

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_keypoints, frame_descriptors = compute_features(frame_gray, detector)
        good_matches = match_descriptors(reference_descriptors, frame_descriptors, matcher)

        corners, inlier_mask = localize_object(
            reference_keypoints,
            frame_keypoints,
            good_matches,
            reference_gray.shape
        )

        # TODO 7: draw result with status text
        result = draw_object(frame, corners)

        object_found = corners is not None
        status_text  = "DETECTED"     if object_found else "NOT DETECTED"
        status_color = (0, 255, 0)    if object_found else (0, 0, 255)

        _put_text(result, f"Matches: {len(good_matches)}", (10, 45))
        _put_text(result, status_text,                     (10, 105), status_color, scale=2.2)

        cv2.imshow("Object tracking in video", resize_to_fit(result))

        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="LAB3 - complete solution: descriptors")
    parser.add_argument("--reference", required=True,
                        help="Path to the reference image, e.g. saw1.jpg")
    parser.add_argument("--image",  help="Path to the test image")
    parser.add_argument("--video",  help="Path to the video file, e.g. sawmovie.mp4")
    parser.add_argument("--method", default="ORB",
                        help="Descriptor method: ORB, BRISK, SIFT")

    args = parser.parse_args()

    if args.image is None and args.video is None:
        print("ERROR: Provide --image or --video", file=sys.stderr)
        return 1

    if args.image is not None:
        process_image(args.reference, args.image, args.method)

    if args.video is not None:
        process_video(args.reference, args.video, args.method)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())