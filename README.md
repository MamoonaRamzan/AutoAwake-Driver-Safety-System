# 🚗 AutoAwake-Driver-Drowsiness-Detection-System

This Driver Drowsiness Detection System is an AI-powered application that monitors driver fatigue in real-time using computer vision. The system detects eye closure and yawning to identify potential drowsiness, providing visual and audio alerts to prevent accidents caused by driver fatigue.

---

## 🔍 Features

- 🧠 YOLOv8-based model for real-time drowsiness detection using computer vision and deep learning
- 🎥 Real-time webcam inference
- 📊 Evaluates with metrics like mAP, Precision, and Recall
- 🔊 Sound alerts when drowsiness is detected (closed eyes/yawning).
- ⚠️ Multi-level alerting system, Warning and critical alerts based on detection thresholds
- ⏲ Session tracking: Records drowsy events throughout the monitoring period

---
## 💻 Usage 
Run the application:
bash python drowsy_detector.py

Position your face within the camera view at an appropriate distance
The system will automatically begin detecting eye states and yawning
The status panel shows current detection status and session metrics
To exit:

Click the "QUIT" button to close the application

---
## 👁️ Preview
<img width="706" alt="Screenshot 2025-04-26 142224" src="https://github.com/user-attachments/assets/dd895fee-5aaa-4237-abed-2773f0e5c0c5" />
<img width="706" alt="Screenshot 2025-04-26 142825" src="https://github.com/user-attachments/assets/1e45ab62-9fe0-4225-9e99-f5e8725debdb" />



