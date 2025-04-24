# Driver-Drowsiness-Detection-App
A YOLOv8-based real-time drowsiness detection system through facial expressions such as closed eyes and yawning, trained on yawn-eye-dataset from kaggle. Supports live webcam inference for detecting drowsiness with sound alerts.

## 📁 Project Structure
DriverDrowsinessDetector/

├── data/

│   ├── train/
│   └── test/
├──model.pt    
├── Alert.wav  
├── training.ipynb            
├── drowy detector.py

## 🔍 Features
- 🧠 Trained YOLOv8 model for drowsiness detection
- 🎥 Real-time detection via webcam
- 📊 Evaluation with mAP, Precision, Recall
- ⚠️ Alerts when drowsiness is detected (closed eyes/yawn)
