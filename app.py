import streamlit as st
import os
from IPython.display import Image, display
from roboflow import Roboflow

def install_requirements():
    st.code("%pip install roboflow")
    st.code("!git clone https://github.com/ultralytics/yolov5")
    st.code("%cd yolov5")
    st.code("%pip install -qr requirements.txt")
    st.code("%pip install -q roboflow")
    
def assemble_dataset():
    st.write("Step 2: Assemble Our Dataset")
    os.environ["/content/Bees-and-Hornets-1"] = "/content/datasets"
    rf = Roboflow(api_key="YOUR_API_KEY")
    project = rf.workspace("bees-and-hornets-20yku").project("bees-and-hornets")
    dataset = project.version(2).download("yolov5")
    return dataset

def train_model(dataset):
    st.write("Step 3: Train Our Custom YOLOv5 model")
    train_code = f"!python train.py --img 640 --batch 16 --epochs 100 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache"
    st.code(train_code)
    
def run_inference(dataset):
    st.write("Run Inference With Trained Weights")
    inference_code = f"!python detect.py --weights runs/train/exp3/weights/best.pt --img 640 --conf 0.5 --source {dataset.location}/test/images"
    st.code(inference_code)
    
def display_images():
    st.write("Display inference on ALL test images")
    for imageName in glob.glob('/content/yolov5/runs/detect/exp/*.jpg'):
        display(Image(filename=imageName))
        st.write("\n")
        
def export_weights():
    st.write("Export model's weights for future use")
    export_code = "from google.colab import files\nfiles.download('./runs/train/exp/weights/best.pt')"
    st.code(export_code)

def infer_on_video():
    st.write("Inferencing on videos")
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
       
def main():
    st.title("YOLOv5 Streamlit App")
    
    install_requirements_button = st.button("Step 1: Install Requirements")
    if install_requirements_button:
        install_requirements()

    assemble_dataset_button = st.button("Step 2: Assemble Dataset")
    if assemble_dataset_button:
        dataset = assemble_dataset()

    train_model_button = st.button("Step 3: Train Model")
    if train_model_button:
        train_model(dataset)

    inference_button = st.button("Run Inference")
    if inference_button:
        run_inference(dataset)
        display_images()

    export_weights_button = st.button("Export Weights")
    if export_weights_button:
        export_weights()

    infer_on_video_button = st.button("Infer on Video")
    if infer_on_video_button:
        infer_on_video()
        
if __name__ == "__main__":
    main()