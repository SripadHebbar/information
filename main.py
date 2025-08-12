import time
import cv2
import numpy as np
import os
from datetime import datetime

from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH = "models/efficientdet_lite0.tflite"
BASE_LOG_DIR = "logs"
CAMERA_ID = 0
SCORE_THRESHOLD = 0.35
MAX_RESULTS = 10

# ---- Create a unique log directory for each run ----
run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_DIR = os.path.join(BASE_LOG_DIR, f"run_{run_timestamp}")
os.makedirs(LOG_DIR, exist_ok=True)
# ---------------------------------------------------

material_composition = {
    "steel": {"iron_%": 70, "carbon_%": 0.2, "chromium_%": 18, "nickel_%": 8, "others_%": 3.8},
    "cast_iron": {"iron_%": 94, "carbon_%": 3.5, "silicon_%": 1.5, "others_%": 1.0},
    "aluminum": {"aluminum_%": 98, "magnesium_%": 1, "others_%": 1},
    "copper": {"copper_%": 99.9, "others_%": 0.1},
    "brass": {"copper_%": 65, "zinc_%": 35},
    "bronze": {"copper_%": 88, "tin_%": 12},
    "plastic": {"carbon_%": 50, "hydrogen_%": 50},
    "polyethylene": {"carbon_%": 86, "hydrogen_%": 14},
    "polypropylene": {"carbon_%": 85.7, "hydrogen_%": 14.3},
    "polycarbonate": {"carbon_%": 74, "hydrogen_%": 5, "oxygen_%": 21},
    "nylon": {"carbon_%": 63, "hydrogen_%": 9, "oxygen_%": 13, "nitrogen_%": 15},
    "ceramic": {"aluminum_oxide_%": 70, "silicon_oxide_%": 25, "others_%": 5},
    "glass": {"silicon_dioxide_%": 72, "sodium_oxide_%": 14, "calcium_oxide_%": 10, "others_%": 4},
    "rubber": {"carbon_%": 85, "hydrogen_%": 15},
    "silicone": {"silicon_%": 30, "oxygen_%": 30, "carbon_%": 20, "hydrogen_%": 20},
    "wood": {"carbon_%": 50, "oxygen_%": 42, "hydrogen_%": 6, "nitrogen_%": 1, "others_%": 1}
}

def guess_material_from_label(label):
    label = label.lower()
    if any(word in label for word in ['steel', 'knife', 'spoon', 'fork', 'pan', 'pot', 'grater']):
        return 'steel'
    if 'aluminum' in label or 'foil' in label:
        return 'aluminum'
    if any(word in label for word in ['bottle', 'cup', 'bowl', 'glass', 'jar', 'wine glass']):
        if 'glass' in label: return 'glass'
        if 'plastic' in label: return 'plastic'
        return 'glass'
    if any(word in label for word in ['plate', 'mug', 'ceramic']):
        return 'ceramic'
    if any(word in label for word in ['wood', 'spatula', 'chopping board']):
        return 'wood'
    if any(word in label for word in ['rubber', 'mat']):
        return 'rubber'
    if any(word in label for word in ['silicone']):
        return 'silicone'
    if 'plastic' in label:
        return 'plastic'
    return None

def geometry_from_contour(cnt):
    if cnt is None or len(cnt) < 3:
        return {}
    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / h if h else None
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull) if hull is not None else 0
    solidity = area / hull_area if hull_area > 0 else None
    extent = area / (w * h) if w > 0 and h > 0 else None
    eqd = np.sqrt(4 * area / np.pi) if area > 0 else None
    orientation = None
    major_axis = minor_axis = None
    if len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)
        orientation = float(ellipse[2])
        major_axis = float(max(ellipse[1]))
        minor_axis = float(min(ellipse[1]))
    min_rect = cv2.minAreaRect(cnt)
    min_rect_area = min_rect[1][0] * min_rect[1][1]
    min_rect_angle = min_rect[2]
    roundness = 4 * area / (np.pi * (peri ** 2)) if peri > 0 else None
    eccentricity = None
    if major_axis and minor_axis and major_axis > 0:
        a = major_axis / 2
        b = minor_axis / 2
        eccentricity = np.sqrt(1 - (b ** 2) / (a ** 2))
    return {
        "area_px": area,
        "perimeter_px": peri,
        "aspect_ratio": aspect,
        "solidity": solidity,
        "extent": extent,
        "equivalent_diameter_px": eqd,
        "orientation_deg": orientation,
        "major_axis_length_px": major_axis,
        "minor_axis_length_px": minor_axis,
        "min_area_rect_area_px": min_rect_area,
        "min_area_rect_angle_deg": min_rect_angle,
        "roundness": roundness,
        "eccentricity": eccentricity,
        "bbox_xywh_px": [int(x), int(y), int(w), int(h)],
    }

def mask_from_color(frame, bbox):
    x, y, w, h = [int(v) for v in bbox]
    crop = frame[y:y+h, x:x+w]
    if crop.size == 0:
        return np.zeros(frame.shape[:2], dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = thresh
    return mask

def write_detection_log(logdir, info_dict, counter):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(logdir, f"log_{ts}_{counter}.txt")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{'-'*60}\n")
        f.write(f"Timestamp: {info_dict['timestamp']}\n")
        f.write(f"Label: {info_dict['label']}\n")
        f.write(f"Score: {info_dict['score']}\n")
        f.write(f"BBox: {info_dict['bbox']}\n")
        f.write(f"Geometry:\n")
        for k, v in info_dict['geometry'].items():
            f.write(f"  {k}: {v}\n")
        f.write(f"Material: {info_dict['material']}\n")
        f.write(f"Composition:\n")
        for k, v in info_dict['composition'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

def build_detector(model_path, score_thresh=0.35, max_results=15):
    opts = mp_vision.ObjectDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        score_threshold=score_thresh,
        max_results=max_results,
    )
    return mp_vision.ObjectDetector.create_from_options(opts)

def run():
    detector = build_detector(MODEL_PATH, score_thresh=SCORE_THRESHOLD, max_results=MAX_RESULTS)
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Press 'q' to quit.")
    log_counter = 1
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        frame_bgr_small = cv2.resize(frame_bgr, (640, 480))
        frame_rgb = cv2.cvtColor(frame_bgr_small, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_img)

        if result.detections:
            for d in result.detections:
                bb = d.bounding_box
                x, y, w, h = float(bb.origin_x), float(bb.origin_y), float(bb.width), float(bb.height)
                bbox = (x, y, w, h)
                label = d.categories[0].category_name if d.categories else "object"
                score = d.categories[0].score if d.categories else 0.0

                mask = mask_from_color(frame_bgr_small, bbox)
                cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt = max(cnts, key=cv2.contourArea) if cnts else None
                geom = geometry_from_contour(cnt)
                mat = guess_material_from_label(label)
                comp = material_composition.get(mat, {})

                # Logging per detection
                log_info = {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "label": label,
                    "score": score,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "geometry": geom,
                    "material": mat if mat else "Unknown",
                    "composition": comp if comp else {"Unknown": "N/A"}
                }
                write_detection_log(LOG_DIR, log_info, log_counter)
                log_counter += 1

                # Draw on image
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame_bgr_small, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame_bgr_small, f"{label} {score:.2f}", (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                if cnt is not None and len(cnt) > 3:
                    cv2.drawContours(frame_bgr_small, [cnt], -1, (255, 200, 0), 2)

                # Collect info lines for vertical display
                info = []
                if geom.get("aspect_ratio") is not None:
                    info.append(f"aspect_ratio:{geom['aspect_ratio']:.2f}")
                if geom.get("solidity") is not None:
                    info.append(f"solidity:{geom['solidity']:.2f}")
                if geom.get("extent") is not None:
                    info.append(f"extent:{geom['extent']:.2f}")
                if geom.get("equivalent_diameter_px") is not None:
                    info.append(f"equivalent_diameter:{geom['equivalent_diameter_px']:.0f}px")
                if geom.get("orientation_deg") is not None:
                    info.append(f"orientation_deg:{geom['orientation_deg']:.0f}°")
                if geom.get("roundness") is not None:
                    info.append(f"roundness:{geom['roundness']:.2f}")
                if geom.get("eccentricity") is not None:
                    info.append(f"eccentricity:{geom['eccentricity']:.2f}")

                # DRAW VERTICALLY
                if info:
                    for i, line in enumerate(info):
                        y_offset = y + h + 18 + i * 22
                        cv2.putText(
                            frame_bgr_small,
                            line,
                            (x, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (0, 200, 255),
                            2
                        )

                # Material (next line after geometry info)
                if mat:
                    y_offset = y + h + 18 + len(info) * 22 + 5
                    cv2.putText(
                        frame_bgr_small,
                        f"Material: {mat}",
                        (x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.53,
                        (255, 255, 0),
                        2
                    )

                # Composition (below material)
                if comp:
                    compstr = ", ".join(f"{k[:-2]}:{v}%" for k, v in comp.items())
                    y_offset = y + h + 18 + len(info) * 22 + 30
                    cv2.putText(
                        frame_bgr_small,
                        f"Comp: {compstr}",
                        (x, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (150, 255, 150),
                        2
                    )

        cv2.imshow("Kitchen Detector (geometry+material+composition)", frame_bgr_small)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
