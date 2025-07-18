😃 EmmotiSense: Real-Time Emotion Detection
EmmotiSense is a Python-based real-time emotion detection system that uses deep learning to classify facial expressions into 7 emotions using a webcam or video feed. The model is trained on the FER-2013 dataset, which contains grayscale images of facial expressions labeled as:

😠 Angry

🤢 Disgust

😨 Fear

😀 Happy

😐 Neutral

😭 Sad

😲 Surprise


🧠 Key Features
Real-time emotion detection using webcam

Custom CNN trained on grayscale emotion images

60%+ accuracy on test set

Uses OpenCV for face detection and TensorFlow for classification

Fully offline and works on mid-range systems


🗂 Folder Structure

EmmotiSense/
├── archive/              # Dataset (from Kaggle)
│   ├── train/            # 7 emotion folders
│   └── test/             # 7 emotion folders
├── emotion_model.h5      # Saved Keras model (after training)
├── train_emotion_model.py
├── real_time_emotion.py
├── README.md             # This file

