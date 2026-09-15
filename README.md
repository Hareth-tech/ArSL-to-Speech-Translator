# ArSL-to-Speech-Translator
## Overview

This project is a real-time, end-to-end assistive system that translates Arabic Sign Language (ArSL) gestures into text and spoken Arabic, developed as a senior capstone project at Al-Rasheed University College (Medical Instrumentation Techniques Engineering Dept.), supervised by Dr. Dhifaf Aziz Kazem.

The system uses a hybrid deep learning pipeline: MediaPipe Hands extracts 21 3D hand landmarks per frame, a CNN captures spatial features from each frame, and an LSTM models the temporal sequence across frames to classify dynamic gestures. Recognized signs are converted to natural Arabic speech in real time using gTTS, played asynchronously via a background thread so the live camera feed never freezes.

Since no public benchmark dataset adequately covers Arabic Sign Language, a custom ArSL video dataset was recorded and annotated by the team to train and evaluate the model.

### Results
- Static gesture accuracy: 80-85%
- Dynamic gesture accuracy: 76–80%
- Overall accuracy on the custom ArSL corpus: 80%
- Real-time inference accuracy: ~90%
- Live end-to-end testing (team): 81% accuracy, ~1.3s average response time (capture → classify → speech output)

### Tech Stack
Python 3.11, OpenCV, MediaPipe Hands, TensorFlow (CNN-LSTM), gTTS, playsound3, threading

### Team
Hareth Hasan Ibrahim, Bareq Hussain Jassim, Mustafa Khalil Zahir, Ali Faeq Abd, Khalid Walid
