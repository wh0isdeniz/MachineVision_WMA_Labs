import cv2
import numpy as np
import os
import sys


HOUGH_CIRCLE_PARAMS = dict(
    dp=1.2,
    minDist=35,
    param1=80,
    param2=33,
    minRadius=20,
    maxRadius=120,
)

RADIUS_THRESHOLD_5PLN = 34

COIN_MAX_VALID_RADIUS = 55
CANNY_LOW = 30
CANNY_HIGH = 100
RING_WIDTH = 6
COIN_MIN_RING_DENSITY = 0.10

TRAY_HSV_LOWER = np.array([5, 120, 120])
TRAY_HSV_UPPER = np.array([25, 255, 255])

HOUGH_LINE_THRESHOLD = 80
HOUGH_LINE_MIN_LENGTH = 150
HOUGH_LINE_MAX_GAP = 30

VALUE_5PLN = 5.00
VALUE_5GR = 0.05

COLOR_TRAY = (0, 200, 255)
COLOR_5PLN_ON = (0, 255, 0)
COLOR_5PLN_OFF = (0, 128, 0)
COLOR_5GR_ON = (255, 180, 0)
COLOR_5GR_OFF = (200, 100, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX

def circle_mask(shape, cx, cy, r):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def coin_features(img, cx, cy, r):
    """
    Compute simple colour / brightness features inside the detected coin.
    Used only as support when radius is borderline.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = circle_mask(img.shape, cx, cy, r)

    mean_h, mean_s, mean_v, _ = cv2.mean(hsv, mask=mask)
    mean_gray = cv2.mean(gray, mask=mask)[0]

    return {
        "mean_s": mean_s,
        "mean_v": mean_v,
        "mean_gray": mean_gray,
    }

def detect_tray(img):
    """
    Returns:
        (x, y, w, h, contour, box_points, lines)

    Uses:
      - HSV mask for orange tray
      - HoughLinesP for edge support
      - contour + minAreaRect for robust tray box
    """
    h_img, w_img = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TRAY_HSV_LOWER, TRAY_HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=HOUGH_LINE_THRESHOLD,
        minLineLength=HOUGH_LINE_MIN_LENGTH,
        maxLineGap=HOUGH_LINE_MAX_GAP
    )

    top_ys, bottom_ys, left_xs, right_xs = [], [], [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

            if abs(angle) < 25 or abs(angle) > 155:
                mid_y = (y1 + y2) / 2
                if mid_y < h_img / 2:
                    top_ys.append(mid_y)
                else:
                    bottom_ys.append(mid_y)
            else:
                mid_x = (x1 + x2) / 2
                if mid_x < w_img / 2:
                    left_xs.append(mid_x)
                else:
                    right_xs.append(mid_x)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    tray_contour = max(contours, key=cv2.contourArea)
    cx, cy, cw, ch = cv2.boundingRect(tray_contour)

    if top_ys and bottom_ys and left_xs and right_xs:
        t = int(np.median(top_ys))
        b = int(np.median(bottom_ys))
        l = int(np.median(left_xs))
        r = int(np.median(right_xs))

        x = min(cx, l)
        y = min(cy, t)
        w = max(cx + cw, r) - x
        h = max(cy + ch, b) - y
    else:
        x, y, w, h = cx, cy, cw, ch

    rect = cv2.minAreaRect(tray_contour)
    box_pts = np.int32(cv2.boxPoints(rect))

    return x, y, w, h, tray_contour, box_pts, lines

def detect_coins(img):
    """
    Returns list of (x, y, radius) for validated coins.
    Uses HoughCircles, then filters candidates with edge density on ring.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 2)

    raw = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, **HOUGH_CIRCLE_PARAMS)
    if raw is None:
        return []

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    result = []
    for x, y, r in np.round(raw[0]).astype(int):
        if r > COIN_MAX_VALID_RADIUS:
            continue

        outer = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(outer, (x, y), r, 255, -1)

        inner = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(inner, (x, y), max(1, r - RING_WIDTH), 255, -1)

        ring_mask = cv2.subtract(outer, inner)
        ring_area = np.count_nonzero(ring_mask)

        if ring_area == 0:
            continue

        ring_edges = cv2.bitwise_and(edges, ring_mask)
        ring_density = np.count_nonzero(ring_edges) / ring_area

        if ring_density < COIN_MIN_RING_DENSITY:
            continue

        result.append((x, y, r))

    return result

def classify_coin(radius, mean_s, mean_v, mean_gray):
    """
    Main rule: radius.
    Borderline radii are refined using simple colour / brightness features.
    """
    if radius >= 34:
        return "5PLN"

    if radius <= 31:
        return "5gr"

    if radius in [32, 33]:
        if mean_s < 130 and mean_v > 95:
            return "5PLN"
        if mean_gray > 105 and mean_s < 145:
            return "5PLN"
        return "5gr"

    return "5gr"


def coin_value(coin_type):
    return VALUE_5PLN if coin_type == "5PLN" else VALUE_5GR


def is_on_tray(x, y, tray_bbox):
    tx, ty, tw, th = tray_bbox
    return tx <= x <= tx + tw and ty <= y <= ty + th

def draw_results(img, tray_result, coins, tray_bbox):
    out = img.copy()
    tx, ty, tw, th = tray_bbox

    if tray_result is not None:
        _, _, _, _, _, box_pts, lines = tray_result

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(out, (x1, y1), (x2, y2), (255, 255, 0), 1)

        cv2.drawContours(out, [box_pts], 0, COLOR_TRAY, 3)
        cv2.putText(out, "TRAY", (tx, max(ty - 8, 20)), FONT, 0.7, COLOR_TRAY, 2)

    on_tray_counts = {"5PLN": 0, "5gr": 0}
    off_tray_counts = {"5PLN": 0, "5gr": 0}

    for coin in coins:
        x, y, r = coin["cx"], coin["cy"], coin["r"]
        ctype = coin["name"]
        on = coin["on_tray"]

        if on:
            color = COLOR_5PLN_ON if ctype == "5PLN" else COLOR_5GR_ON
            on_tray_counts[ctype] += 1
        else:
            color = COLOR_5PLN_OFF if ctype == "5PLN" else COLOR_5GR_OFF
            off_tray_counts[ctype] += 1

        cv2.circle(out, (x, y), r, color, 2)
        cv2.circle(out, (x, y), 3, color, -1)
        cv2.putText(out, ctype, (x - 22, y - r - 6), FONT, 0.5, color, 2)

    val_on = on_tray_counts["5PLN"] * VALUE_5PLN + on_tray_counts["5gr"] * VALUE_5GR
    val_off = off_tray_counts["5PLN"] * VALUE_5PLN + off_tray_counts["5gr"] * VALUE_5GR

    summary_lines = [
        f"ON tray : {on_tray_counts['5PLN'] + on_tray_counts['5gr']} coins = {val_on:.2f} PLN",
        f"OFF tray: {off_tray_counts['5PLN'] + off_tray_counts['5gr']} coins = {val_off:.2f} PLN",
        f"TOTAL   : {sum(on_tray_counts.values()) + sum(off_tray_counts.values())} coins = {val_on + val_off:.2f} PLN",
    ]

    overlay = out.copy()
    cv2.rectangle(overlay, (5, 5), (380, 105), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    for idx, txt in enumerate(summary_lines):
        y_pos = 30 + idx * 28
        cv2.putText(out, txt, (12, y_pos), FONT, 0.7, (255, 255, 255), 2)
        cv2.putText(out, txt, (12, y_pos), FONT, 0.7, (0, 0, 0), 1)

    return out, on_tray_counts, off_tray_counts, val_on, val_off


def process_image(image_path, display=True, save_dir=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Cannot read: {image_path}")
        return None

    name = os.path.basename(image_path)

    print("\n" + "═" * 55)
    print(f"  Image: {name}")
    print("═" * 55)

    tray_result = detect_tray(img)
    if tray_result is None:
        print(f"[WARN] {name}: tray not found, using full image as fallback")
        h, w = img.shape[:2]
        tray_bbox = (0, 0, w, h)
    else:
        tx, ty, tw, th = tray_result[:4]
        tray_bbox = (tx, ty, tw, th)
        print(f"  Tray bounding box : ({tx},{ty}) size=({tw}x{th})")

    detected_circles = detect_coins(img)
    print(f"  Valid circles found : {len(detected_circles)}")

    all_coins = []
    on_tray_coins = []
    off_tray_coins = []

    for cx, cy, r in detected_circles:
        feats = coin_features(img, cx, cy, r)
        ctype = classify_coin(r, feats["mean_s"], feats["mean_v"], feats["mean_gray"])
        value = coin_value(ctype)
        on_tray = is_on_tray(cx, cy, tray_bbox)

        coin = {
            "cx": cx,
            "cy": cy,
            "r": r,
            "name": ctype,
            "value": value,
            "on_tray": on_tray,
            "mean_s": feats["mean_s"],
            "mean_v": feats["mean_v"],
            "mean_gray": feats["mean_gray"],
        }

        all_coins.append(coin)
        if on_tray:
            on_tray_coins.append(coin)
        else:
            off_tray_coins.append(coin)

    total_on = sum(c["value"] for c in on_tray_coins)
    total_off = sum(c["value"] for c in off_tray_coins)

    print(f"\n  ── Coins ON the tray ({len(on_tray_coins)}) ─────────────")
    for c in on_tray_coins:
        print(
            f"     {c['name']}  r={c['r']}px  centre=({c['cx']},{c['cy']})  "
            f"S={c['mean_s']:.1f} V={c['mean_v']:.1f} G={c['mean_gray']:.1f}  "
            f"{c['value']:.2f} PLN"
        )
    print(f"     Subtotal ON  : {total_on:.2f} PLN")

    print(f"\n  ── Coins OFF the tray ({len(off_tray_coins)}) ───────────")
    for c in off_tray_coins:
        print(
            f"     {c['name']}  r={c['r']}px  centre=({c['cx']},{c['cy']})  "
            f"S={c['mean_s']:.1f} V={c['mean_v']:.1f} G={c['mean_gray']:.1f}  "
            f"{c['value']:.2f} PLN"
        )
    print(f"     Subtotal OFF : {total_off:.2f} PLN")

    print(f"\n  ── Grand total ───────────────────────────────")
    print(f"     ON  tray : {len(on_tray_coins):2d} coins = {total_on:.2f} PLN")
    print(f"     OFF tray : {len(off_tray_coins):2d} coins = {total_off:.2f} PLN")
    print(f"     TOTAL    : {len(all_coins):2d} coins = {total_on + total_off:.2f} PLN")

    result_img, on_counts, off_counts, val_on, val_off = draw_results(
        img, tray_result, all_coins, tray_bbox
    )

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"result_{name}")
        cv2.imwrite(out_path, result_img)
        print(f"\n  Saved → {out_path}")

    if display:
        cv2.imshow(f"Result - {name}", result_img)
        print("  Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return {
        "filename": name,
        "tray_bbox": tray_bbox,
        "coins_on_tray": on_tray_coins,
        "coins_off_tray": off_tray_coins,
        "total_on": total_on,
        "total_off": total_off,
        "annotated": result_img,
    }


def main():
    if len(sys.argv) > 1:
        img_dir = sys.argv[1]
    else:
        img_dir = "."

    save_dir = os.path.join(img_dir, "results")
    display = "--no-display" not in sys.argv

    image_files = sorted([
        os.path.join(img_dir, f"tray{i}.jpg") for i in range(1, 9)
    ])

    all_results = []

    print("Coin Detection - LAB2 (Hough Transform)")
    print(f"Processing images from: {img_dir}")
    print(f"Radius threshold 5PLN/5gr: {RADIUS_THRESHOLD_5PLN}px")

    for path in image_files:
        if not os.path.exists(path):
            print(f"[SKIP] Not found: {path}")
            continue

        result = process_image(path, display=display, save_dir=save_dir)
        if result is not None:
            all_results.append(result)

    if all_results:
        total_coins = sum(len(r["coins_on_tray"]) + len(r["coins_off_tray"]) for r in all_results)
        total_value = sum(r["total_on"] + r["total_off"] for r in all_results)

        print("\n" + "═" * 55)
        print("  GRAND SUMMARY - ALL IMAGES")
        print("═" * 55)
        print(f"  Images processed : {len(all_results)}")
        print(f"  Total coins found: {total_coins}")
        print(f"  Total value      : {total_value:.2f} PLN")
        print("═" * 55)

    print("\nDone.")


if __name__ == "__main__":
    main()