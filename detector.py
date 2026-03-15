"""
Gender & Age Detection using OpenCV DNN + Levi-Hassner Caffe models
--------------------------------------------------------------------
Usage:
    python detector.py --image path/to/image.jpg
    python detector.py --video path/to/video.mp4
    python detector.py --webcam
    python detector.py --webcam --save output.avi
"""

import cv2
import numpy as np
import argparse
import time
import os
import sys

# ── Model configuration ───────────────────────────────────────────────────────

AGE_BUCKETS  = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST  = ['Male', 'Female']
MEAN_VALUES  = (78.4263377603, 87.7689143744, 114.895847746)

MODEL_FILES = {
    "face_proto":   "models/opencv_face_detector.pbtxt",
    "face_model":   "models/opencv_face_detector_uint8.pb",
    "age_proto":    "models/deploy_age.prototxt",
    "age_model":    "models/age_net.caffemodel",
    "gender_proto": "models/deploy_gender.prototxt",
    "gender_model": "models/gender_net.caffemodel",
}

# ── Colors (BGR) ──────────────────────────────────────────────────────────────

COLORS = {
    "Male":   (219, 152,  52),   # blue-ish
    "Female": ( 72, 101, 241),   # pink-ish
    "box":    (  0, 220, 120),
    "text_bg":( 30,  30,  30),
}


# ── Loader ────────────────────────────────────────────────────────────────────

def load_models():
    """Load all DNN models. Raises FileNotFoundError with a helpful message."""
    missing = [v for v in MODEL_FILES.values() if not os.path.exists(v)]
    if missing:
        print("\n[ERROR] Missing model files:")
        for f in missing:
            print(f"  • {f}")
        print("\nRun  python download_models.py  to download them automatically.\n")
        sys.exit(1)

    face_net   = cv2.dnn.readNet(MODEL_FILES["face_model"],   MODEL_FILES["face_proto"])
    age_net    = cv2.dnn.readNet(MODEL_FILES["age_model"],    MODEL_FILES["age_proto"])
    gender_net = cv2.dnn.readNet(MODEL_FILES["gender_model"], MODEL_FILES["gender_proto"])

    # Use CUDA if available, else CPU
    for net in (face_net, age_net, gender_net):
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return face_net, age_net, gender_net


# ── Inference helpers ─────────────────────────────────────────────────────────

def detect_faces(frame, face_net, conf_threshold=0.7):
    """Returns list of (x1, y1, x2, y2) bounding boxes."""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                  [104, 117, 123], swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > conf_threshold:
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w - 1, int(detections[0, 0, i, 5] * w))
            y2 = min(h - 1, int(detections[0, 0, i, 6] * h))
            faces.append((x1, y1, x2, y2, float(conf)))
    return faces


def predict_age_gender(face_img, age_net, gender_net):
    """Returns (gender, age_bucket, gender_conf, age_conf)."""
    blob = cv2.dnn.blobFromImage(
        face_img, 1.0, (227, 227), MEAN_VALUES, swapRB=False
    )

    gender_net.setInput(blob)
    g_preds = gender_net.forward()[0]
    gender  = GENDER_LIST[g_preds.argmax()]
    g_conf  = float(g_preds.max())

    age_net.setInput(blob)
    a_preds = age_net.forward()[0]
    age     = AGE_BUCKETS[a_preds.argmax()]
    a_conf  = float(a_preds.max())

    return gender, age, g_conf, a_conf

# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_results(frame, faces_data):
    """
    faces_data: list of dicts with keys:
        box, gender, age, gender_conf, age_conf
    """
    overlay = frame.copy()

    for fd in faces_data:
        x1, y1, x2, y2 = fd["box"]
        gender   = fd["gender"]
        age      = fd["age"]
        g_conf   = fd["gender_conf"]
        a_conf   = fd["age_conf"]
        color    = COLORS[gender]

        # Semi-transparent filled rect
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        overlay = frame.copy()

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        label     = f"{gender}  |  {age}  ({g_conf:.0%} / {a_conf:.0%})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        bg_y1 = max(0, y1 - th - 12)
        cv2.rectangle(frame, (x1, bg_y1), (x1 + tw + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_stats(frame, fps, face_count):
    """HUD overlay in top-left corner."""
    h, w = frame.shape[:2]
    stats = [
        f"FPS   : {fps:.1f}",
        f"Faces : {face_count}",
    ]
    for i, line in enumerate(stats):
        y = 28 + i * 24
        cv2.putText(frame, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# ── Core processing loop ──────────────────────────────────────────────────────

def process_frame(frame, face_net, age_net, gender_net,
                  conf_threshold=0.7, padding=20):
    """Detect faces and predict age/gender for a single frame."""
    faces_raw = detect_faces(frame, face_net, conf_threshold)
    h, w = frame.shape[:2]
    faces_data = []

    for (x1, y1, x2, y2, face_conf) in faces_raw:
        # Padded crop
        fx1 = max(0, x1 - padding)
        fy1 = max(0, y1 - padding)
        fx2 = min(w - 1, x2 + padding)
        fy2 = min(h - 1, y2 + padding)
        face_crop = frame[fy1:fy2, fx1:fx2]

        if face_crop.size == 0:
            continue

        gender, age, g_conf, a_conf = predict_age_gender(
            face_crop, age_net, gender_net
        )
        faces_data.append({
            "box":         (x1, y1, x2, y2),
            "gender":      gender,
            "age":         age,
            "gender_conf": g_conf,
            "age_conf":    a_conf,
            "face_conf":   face_conf,
        })

    return faces_data


# ── Mode runners ──────────────────────────────────────────────────────────────

def run_image(path, face_net, age_net, gender_net, args):
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] Cannot read image: {path}")
        return

    faces_data = process_frame(frame, face_net, age_net, gender_net,
                                args.confidence, args.padding)
    out = draw_results(frame, faces_data)

    for fd in faces_data:
        print(f"  • Gender: {fd['gender']} ({fd['gender_conf']:.1%})  "
              f"Age: {fd['age']} ({fd['age_conf']:.1%})")

    cv2.imshow("Gender & Age Detection", out)
    if args.save:
        cv2.imwrite(args.save, out)
        print(f"[INFO] Saved result to {args.save}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video_or_webcam(source, face_net, age_net, gender_net, args):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        return

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.save, fourcc, fps_src, (fw, fh))

    prev_time = time.time()
    fps = 0.0

    print("[INFO] Press  q  to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces_data = process_frame(frame, face_net, age_net, gender_net,
                                   args.confidence, args.padding)
        frame = draw_results(frame, faces_data)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - prev_time + 1e-9))
        prev_time = now

        draw_stats(frame, fps, len(faces_data))
        cv2.imshow("Gender & Age Detection  [q = quit]", frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"[INFO] Saved video to {args.save}")
    cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Gender & Age Detection")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",  metavar="PATH", help="Path to an image file")
    src.add_argument("--video",  metavar="PATH", help="Path to a video file")
    src.add_argument("--webcam", action="store_true", help="Use default webcam")

    p.add_argument("--save",       metavar="PATH",  help="Save output to file")
    p.add_argument("--confidence", type=float, default=0.7,
                   help="Face detection confidence threshold (default: 0.7)")
    p.add_argument("--padding",    type=int,   default=20,
                   help="Pixels to pad around each detected face (default: 20)")
    p.add_argument("--cam-id",     type=int,   default=0,
                   help="Webcam device index (default: 0)")
    return p.parse_args()


def main():
    args = parse_args()
    print("[INFO] Loading models...")
    face_net, age_net, gender_net = load_models()
    print("[INFO] Models loaded.\n")

    if args.image:
        run_image(args.image, face_net, age_net, gender_net, args)
    elif args.video:
        run_video_or_webcam(args.video, face_net, age_net, gender_net, args)
    else:
        run_video_or_webcam(args.cam_id, face_net, age_net, gender_net, args)


if __name__ == "__main__":
    main()