"""
download_models.py  –  Download all required model files
Run once before using detector.py
"""

import os
import urllib.request

MODELS_DIR = "models"

FILES = {
    # OpenCV face detector (TensorFlow)
    "opencv_face_detector.pbtxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/opencv_face_detector.pbtxt"
    ),
    "opencv_face_detector_uint8.pb": (
        "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
        "opencv_face_detector_uint8.pb"
    ),
    # Age model (Levi & Hassner, Caffe)
    "deploy_age.prototxt": (
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/"
        "age_net_definitions/deploy.prototxt"
    ),
    "age_net.caffemodel": (
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/"
        "models/age_net.caffemodel"
    ),
    # Gender model (Levi & Hassner, Caffe)
    "deploy_gender.prototxt": (
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/"
        "gender_net_definitions/deploy.prototxt"
    ),
    "gender_net.caffemodel": (
        "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/"
        "models/gender_net.caffemodel"
    ),
}


def download(filename, url, dest_dir):
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        print(f"  [skip] {filename} already exists")
        return
    print(f"  [download] {filename} ...", end="", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f" done ({size_mb:.1f} MB)")
    except Exception as e:
        print(f" FAILED: {e}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Downloading models into ./{MODELS_DIR}/\n")
    for name, url in FILES.items():
        download(name, url, MODELS_DIR)
    print("\nAll done! Now run:\n  python detector.py --webcam\n")


if __name__ == "__main__":
    main()