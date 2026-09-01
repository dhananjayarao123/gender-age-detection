# Gender & Age Detection

Real-time gender and age prediction from faces in an image, video file, or webcam feed, using OpenCV's DNN module with pre-trained Caffe models (Levi & Hassner architecture) for age/gender classification and a TensorFlow-based face detector.

## Problem type
Classification — for each detected face, predicts:
- **Gender**: `Male` / `Female`
- **Age group**: one of 8 buckets — `(0-2)`, `(4-6)`, `(8-12)`, `(15-20)`, `(25-32)`, `(38-43)`, `(48-53)`, `(60-100)`

## How it works
1. **Face detection** — an OpenCV DNN face detector (`opencv_face_detector`) locates all faces in a frame and returns bounding boxes.
2. **Age & gender inference** — each detected face is cropped (with padding), resized to 227×227, and passed through two separate Caffe models (`age_net`, `gender_net`) trained on the Adience dataset.
3. **Visualization** — bounding boxes and labels (gender, age bucket, confidence) are drawn on the frame, color-coded by predicted gender, with an FPS/face-count HUD for video and webcam modes.

## Setup

**1. Install dependencies**
```bash
pip install opencv-python numpy
```

**2. Download the pre-trained models** (one-time setup — fetches the face detector and the age/gender Caffe models into `models/`)
```bash
python models.py
```

## Usage

**Single image**
```bash
python detector.py --image path/to/image.jpg
```

**Video file**
```bash
python detector.py --video path/to/video.mp4
```

**Webcam (live)**
```bash
python detector.py --webcam
```

**Save the output** (works with any mode above)
```bash
python detector.py --webcam --save output.avi
```

**Other options**
| Flag | Default | Description |
|---|---|---|
| `--confidence` | 0.7 | Face detection confidence threshold |
| `--padding` | 20 | Pixels of padding around each detected face before cropping |
| `--cam-id` | 0 | Webcam device index (for multi-camera setups) |

Press `q` to quit video/webcam mode.

**Batch processing a folder of images**
```bash
python batch.py --input ./photos --output results.csv
```
Runs detection on every image in a folder and exports gender, age, and confidence scores for every detected face to a CSV file.

## Project structure
```
gender-age-detection/
├── detector.py     # Core detection pipeline + CLI (image / video / webcam)
├── batch.py        # Batch-process a folder of images to CSV
├── models.py       # One-time script to download the required model files
├── models/         # Downloaded model weights (created by models.py)
│   ├── opencv_face_detector.pbtxt
│   ├── opencv_face_detector_uint8.pb
│   ├── deploy_age.prototxt
│   ├── age_net.caffemodel
│   ├── deploy_gender.prototxt
│   └── gender_net.caffemodel
└── .gitignore
```

## Models used
- **Face detector**: OpenCV's SSD-based face detector (TensorFlow `.pb`)
- **Age & gender classifiers**: [Levi & Hassner](https://github.com/GilLevi/AgeGenderDeepLearning) Caffe models, trained on the Adience benchmark dataset

## Limitations
- Age is predicted as a coarse 8-bucket range, not an exact number — this is a known characteristic of the underlying Adience-trained model, not a bug.
- Accuracy drops on extreme angles, poor lighting, occlusion, or faces far from the camera.
- Like most face-analysis models trained on Adience-era data, predictions can be less reliable across the full diversity of ages, skin tones, and lighting conditions found in real-world use — treat outputs as approximate, not authoritative, especially in any sensitive or decision-making context.
