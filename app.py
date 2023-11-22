import streamlit as st
import os
from PIL import Image
from roboflow import Roboflow
import torch
import glob
from IPython.display import display

# Step 1: Install Requirements
st.title("Detecting & Tracking Vespa Velutina (Yellow Legged Hornet) Streamlit App")

# Step 2: Assemble Our Dataset
assemble_dataset_button = st.button("Assemble Dataset")
if assemble_dataset_button:
    st.write("Setting up environment...")
    os.environ["/content/Bees-and-Hornets-1"] = "/content/datasets"
    rf = Roboflow(api_key="TAQXNlttGMH92B2rCZpo")
    project = rf.workspace("bees-and-hornets-20yku").project("bees-and-hornets")
    dataset = project.version(2).download("yolov5")

# Step 3: Train Our Custom YOLOv5 model
train_model_button = st.button("Train Model")
if train_model_button:
    st.write("Training the Model...")
    train_code = f"!python train.py --img 640 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache"
    st.code(train_code)

# Run Inference With Trained Weights
inference_button = st.button("Run Inference")
if inference_button:
    st.write("Running Inference...")
    inference_code = f"!python detect.py --weights runs/train/exp3/weights/best.pt --img 640 --conf 0.5 --source {dataset.location}/test/images"
    st.code(inference_code)

    # Display inference on ALL test images
    for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'):
        image = Image.open(imageName)
        st.image(image, caption=imageName, use_column_width=True)
        st.write("\n")
        
# Export model's weights for future use
export_weights_button = st.button("Export Weights")
if export_weights_button:
    st.write("Exporting model's weights for future use...")
    from google.colab import files
    files.download('./runs/train/exp/weights/best.pt')

# Inferencing on videos
infer_on_video_button = st.button("Infer on Video")
if infer_on_video_button:
    st.write("Inferencing on videos...")
    st.code("%pip install supervision")
    infer_video_code = """
import cv2
import torch
import supervision as sv
import numpy as np
from ultralytics import YOLO

VIDEO_PATH = "/content/drive/MyDrive/BeesAndHornets2.mp4"
model = YOLO("/content/yolov5/runs/train/exp3/weights/best.pt")

!python detect.py --weights "/content/yolov5/runs/train/exp3/weights/best.pt" --source "/content/drive/MyDrive/BeesAndHornets2.mp4"
"""
    st.code(infer_video_code)