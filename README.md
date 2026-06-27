🚦Traffic Sign Detection and Classification

This project implements two separate deep learning models for traffic sign analysis:

YOLO for traffic sign detection, which locates traffic signs in images by predicting bounding boxes and object classes.
CNN for traffic sign classification, which classifies traffic sign images into their corresponding categories.

Both models are served through a FastAPI backend, allowing users to perform detection or classification via REST API endpoints. The project also integrates MLflow for experiment tracking, model management, and performance monitoring.

Features
Traffic sign detection using YOLO
Traffic sign classification using CNN
FastAPI REST API
MLflow experiment tracking
Model evaluation
Deployment-ready backend
Tech Stack
Python
YOLO (Ultralytics)
TensorFlow / Keras
FastAPI
OpenCV
MLflow
NumPy
Pandas
Uvicorn
