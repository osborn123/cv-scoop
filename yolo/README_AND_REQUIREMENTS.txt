# Object Detection + Stereo Camera Distance Measurement (YOLO + Stereo Vision)

## ✅ Overview
This project integrates **YOLO object detection** and **stereo camera distance measurement**, allowing **real-time object detection with distance estimation**.

## 📌 Workflow
1. **Camera Calibration (`stereo_calibration.py`)**
   - Computes intrinsic/extrinsic parameters of **stereo cameras**.
   - Saves **calibration parameters** as `stereo_params.npz`.

2. **Object Detection + Distance Estimation (`stereo_object_detection.py`)**
   - Reads **stereo camera video stream**.
   - Computes **disparity map** and **depth map**.
   - Uses **YOLO object detection** and estimates object distance.

---

## 🛠 Environment Setup
### Install Dependencies
```bash
pip install -r README_AND_REQUIREMENTS.txt
```

### Ensure you have the following hardware
- ✅ **Stereo Camera (Dual Cameras)**
- ✅ **Chessboard calibration images** (stored in `left/` and `right/` folders)

---

## 1️⃣ Run Camera Calibration
```bash
python stereo_calibration.py
```
**Input**:
- Chessboard images in `left/` and `right/` folders.

**Output**:
- Generates **camera calibration parameters file** `stereo_params.npz`.

---

## 2️⃣ Run Object Detection + Distance Estimation
```bash
python stereo_object_detection.py
```
**Input**:
- **Real-time stereo camera video stream**.

**Output**:
- ✅ **Real-time Object Detection** (YOLO labels object name + confidence score)
- ✅ **Real-time Distance Estimation** (Displays object distance in cm)
- ✅ **Disparity Map Display**
- ✅ **Depth Map Display**

📌 **Press `q` to exit**

---

## 📂 Project Directory Structure
```
📂 Project Directory
│── README_AND_REQUIREMENTS.txt  # Instructions & dependencies
│── stereo_calibration.py        # Camera Calibration
│── stereo_object_detection.py   # Object Detection + Distance Estimation
│── stereo_params.npz            # Camera Parameters (Generated after calibration)
│── left/                        # Left camera chessboard images
│── right/                       # Right camera chessboard images
```

---

## ✅ Summary
- 🎯 **Supports real-time distance estimation for autonomous driving, robotics, AR applications**.
- 🎯 **Uses YOLO for high-precision object detection**.
- 🎯 **Employs stereo vision for depth estimation and improved accuracy**.

🚀 **Run the scripts to achieve YOLO-based object detection + real-time distance measurement!**

---

## 📜 Required Dependencies (Can be installed using this file)
```txt
opencv-python
numpy
ultralytics
```
