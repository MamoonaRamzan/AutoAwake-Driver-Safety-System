# Driver-Drowsiness-Detection-App
A YOLOv8-based real-time drowsiness detection system through facial expressions such as closed eyes and yawning, trained on yawn-eye-dataset from kaggle. Supports live webcam inference for detecting drowsiness with sound alerts.

---

## 📁 Project Structure
DriverDrowsinessDetector:
  data:
    - train/
    - test/
  model.pt:"Trained YOLOv8 model weights"
  Alert.wav:"Audio alert for drowsiness detection"
  training.ipynb:"Notebook for model training and evaluation"
  drowy detector.py:"Script for real-time drowsiness detection using webcam"
  
---

## 🔍 Features

- 🧠 YOLOv8-based model for drowsiness detection
- 🎥 Real-time webcam inference
- 📊 Evaluates with metrics like mAP, Precision, and Recall
- ⚠️ Sound alerts when drowsiness is detected (closed eyes/yawning)

---
