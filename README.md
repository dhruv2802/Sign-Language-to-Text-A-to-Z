# Sign-Language-to-Text-A-to-Z

# Sign Language Detection using Computer Vision

A beginner-friendly **Computer Vision and Machine Learning practice project** that detects hand gestures through a webcam and extracts hand landmark features using **MediaPipe** and **OpenCV**.

This project is being developed as a learning step toward building a more advanced **Sign Language to Text** system.

## 🎯 Objective

The main objective of this project is to understand how computer vision can be used to:

- Detect a human hand using a webcam
- Identify **21 hand landmarks**
- Extract landmark coordinates
- Normalize the extracted features
- Prepare the data for gesture/sign classification
- Recognize different hand gestures as letters or signs

## 🛠️ Technologies Used

- **Python**
- **OpenCV** – Webcam access and image processing
- **MediaPipe** – Hand detection and landmark extraction
- **NumPy** – Numerical operations and feature processing
- **Machine Learning** – For future gesture classification

## 🔄 Project Workflow

```text
Webcam
   ↓
Capture Hand Image
   ↓
Hand Detection using MediaPipe
   ↓
Extract 21 Hand Landmarks
   ↓
Extract Landmark Coordinates
   ↓
Normalize Features
   ↓
Train Classification Model
   ↓
Predict Sign / Gesture
   ↓
Display Output
```

## 📁 Project Structure

```text
SignLanguageProject/
│
├── function.py
├── features.py
├── normalization.py
├── hand_landmarker.task
└── README.md
```

## ✋ Hand Landmark Detection

MediaPipe detects **21 landmarks** on a hand.

Each landmark contains:

- X coordinate
- Y coordinate
- Z coordinate

Therefore:

```text
21 landmarks × 3 coordinates = 63 features
```

These 63 values can be used as input features for a machine learning classification model.

## 🚀 Current Progress

- [ ] Webcam integration
- [ ] Hand detection
- [ ] 21 hand landmark detection
- [ ] Landmark coordinate extraction
- [ ] Feature normalization
- [ ] Model training
- [ ] Gesture classification
- [ ] Real-time sign prediction
- [ ] Sign-to-text conversion

## 🔮 Future Scope

The project can be extended to recognize multiple sign-language gestures and convert them into readable text in real time.

Possible future improvements include:

- Training a classification model on ASL/sign-language datasets
- Real-time alphabet recognition
- Word formation from recognized signs
- Sentence generation
- Improved accuracy using deep learning
- Support for more complex dynamic gestures

## 📌 Note

This project is primarily developed as a **learning and practice project** to understand computer vision, hand landmark extraction, feature engineering, and gesture classification before working on a larger final-year project.

## 👨‍💻 Author

**Dhruv Marathe**

IT Engineering Student
