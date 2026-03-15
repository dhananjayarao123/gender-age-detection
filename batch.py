"""
batch_analyze.py  –  Analyze a folder of images and export results to CSV
Usage:
    python batch_analyze.py --input ./photos --output results.csv
"""

import cv2
import os
import csv
import argparse
from detector import load_models, process_frame

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Batch gender & age analysis")
    p.add_argument("--input",  required=True, metavar="DIR",
                   help="Folder containing images")
    p.add_argument("--output", default="results.csv", metavar="FILE",
                   help="Output CSV file (default: results.csv)")
    p.add_argument("--confidence", type=float, default=0.7)
    p.add_argument("--padding",    type=int,   default=20)
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input):
        print(f"[ERROR] Not a directory: {args.input}")
        return

    images = [
        f for f in os.listdir(args.input)
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ]
    if not images:
        print("[ERROR] No supported images found in input folder.")
        return

    print(f"[INFO] Found {len(images)} images. Loading models...")
    face_net, age_net, gender_net = load_models()
    print(f"[INFO] Models loaded. Processing...\n")

    rows = []
    for idx, filename in enumerate(sorted(images), 1):
        path = os.path.join(args.input, filename)
        frame = cv2.imread(path)
        if frame is None:
            print(f"  [{idx}/{len(images)}] SKIP (unreadable): {filename}")
            continue

        faces_data = process_frame(frame, face_net, age_net, gender_net,
                                   args.confidence, args.padding)

        if not faces_data:
            print(f"  [{idx}/{len(images)}] No faces: {filename}")
            rows.append({
                "file": filename, "face_index": "-",
                "gender": "-", "gender_conf": "-",
                "age": "-", "age_conf": "-",
                "face_detect_conf": "-",
            })
        else:
            for fi, fd in enumerate(faces_data):
                print(f"  [{idx}/{len(images)}] {filename}  face#{fi+1}: "
                      f"{fd['gender']} {fd['age']}")
                rows.append({
                    "file":             filename,
                    "face_index":       fi + 1,
                    "gender":           fd["gender"],
                    "gender_conf":      f"{fd['gender_conf']:.3f}",
                    "age":              fd["age"],
                    "age_conf":         f"{fd['age_conf']:.3f}",
                    "face_detect_conf": f"{fd['face_conf']:.3f}",
                })

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "face_index", "gender", "gender_conf",
                           "age", "age_conf", "face_detect_conf"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[INFO] Results saved to {args.output}")
    print(f"[INFO] Processed {len(images)} images, {len(rows)} face records.")


if __name__ == "__main__":
    main()