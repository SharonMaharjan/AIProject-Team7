import streamlit as st
import os
import glob
import tempfile
import cv2
from IPython.display import display
#from roboflow import Roboflow
import numpy as np
from PIL import Image
import torch
import base64
from yolov5 import detect
import yolov5
from obj_det_and_trk import 
number_uploaded_images = 0
uploaded_files = None
model_path = "./weights/bestv2.pt"
# model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

st.title("Detecting & Tracking Yellow Legged Hornet Streamlit App:female-detective::honeybee:")
st.caption('''
Created by [Asir Faysal Rafin](https://github.com/asirfaysal), [Daan Michielsen](https://github.com/DaanMichielsen) and [Sharon Maharjan](https://github.com/SharonMaharjan)
           ''')
st.subheader("Back story:book:", divider='orange')
bee, hornet = st.columns(2)
with bee:
    img_bee = Image.open('./streamlit_images/bee.jpg')
    st.image(img_bee)
with hornet:
    img_hornet = Image.open('./streamlit_images/hornet.jpg')
    st.image(img_hornet)

with st.expander(label="A beekeeper in Belgium recently reached out...", expanded=False):
    st.markdown("A beekeeper in Belgium recently reached out to us to see if we could help solve his yellow-legged hornet problem using object tracking. The idea is to track the path of the hornet after it eats, as it will fly directly back to the nest. By tracking the hornet’s path, we can locate the nest and remove it. Without the bees the agriculture would suffer tramendously and cause lower yield numbers.")

st.sidebar.header("Configuration:gear:", divider='orange')
confidence = st.sidebar.slider("Confidence",min_value=0.0,max_value=1.0,value=0.50, help="The confidence of the model must exceed the threshold in order for the object to be detected.")
st.sidebar.subheader("Video	:film_frames:", divider='orange')
run = st.sidebar.toggle('Webcam:movie_camera:')
st.sidebar.markdown('----')
custom_classes = st.sidebar.checkbox('Use custom classes:round_pushpin:')
assigned_class_id=[]
names = ["Bees","Hornets"]

assigned_class = st.sidebar.multiselect('Select the custom classes',list(names),default='Hornets', disabled=False if custom_classes else True)
for each in assigned_class:
    assigned_class_id.append(names.index(each))
st.sidebar.markdown('----')
#demo video
video_file_buffer = st.sidebar.file_uploader("Upload a video:outbox_tray:",type=["mp4","mov",'avi','asf','m4v'], disabled=True if run else False)
st.sidebar.subheader("Image:frame_with_picture:", divider='orange')
uploaded_files = st.sidebar.file_uploader("Upload image(s):outbox_tray:", type=["jpg","png"], accept_multiple_files=False, disabled=True if run else False)
# number_uploaded_images = len(uploaded_files)
st.sidebar.write("Uploaded images:",number_uploaded_images)
# for uploaded_file in uploaded_files:
#     bytes_data = uploaded_file.read()
#     st.sidebar.write("filename:", uploaded_file.name)

# ##get input video here:
# if not video_file_buffer:
#     vid = cv2.VideoCapture(Demo_video)
#     tfflie.name = Demo_video
#     dem_vid = open(tfflie.name,'rb')
#     demo_bytes = dem_vid.read()
    
#     st.sidebar.text('Input video')
#     st.sidebar.video(demo_bytes)
    
# else:
#     tfflie.write(video_file_buffer.read())
#     dem_vid = open(tfflie.name,'rb')
#     demo_bytes=dem_vid.read()
    
#     st.sidebar.text('Input Video')
#     st.sidebar.video(demo_bytes)
# print(tfflie.name)

# def draw_boxes(img, bbox, cls, offset=(0, 0)):
#     for i, box in enumerate(bbox):
#         try:
#             x1, y1, x2, y2, conf = box["xmin"], box["ymin"], box["xmax"], box["ymax"], box["confidence"]
#             x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
#             x1 += offset[0]
#             x2 += offset[0]
#             y1 += offset[1]
#             y2 += offset[1]

#             label = f'{names[int(categories[i])]} {conf:.2f}'

#             data = ((x1 + x2) // 2, (y1 + y2) // 2)

#             (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 191, 0), 2)
#             cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (255, 191, 0), -1)
#             cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [255, 255, 255], 1)
#             cv2.circle(img, data, 3, (255, 191, 0), -1)
#         except (ValueError, KeyError):
#             # Handle errors or missing keys gracefully
#             pass
#     return img



    # If a file is uploaded, run the object detection
# if uploaded_files is not None:
#     # Read the image as an array
#     image = cv2.imdecode(np.frombuffer(uploaded_files.read(), np.uint8), 1)
#     # Convert the image from BGR to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     # Run the model on the image
#     results = model(image)
#     # Get the predictions as a list of dictionaries
#     predictions = results.pandas().xyxy[0].to_dict("records")
    
#     # Filter out rows with non-numeric values in bounding box coordinates
#     predictions = [box for box in predictions if all(isinstance(box[key], (int, float)) for key in ['xmin', 'ymin', 'xmax', 'ymax'])]
    
#     # Draw the boxes and labels on the image
#     image = draw_boxes(image, predictions)
#     print(predictions)
#     # Display the image
#     st.image(image, use_column_width=True)

# Upload Image
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# Perform Object Detection
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    results = detect.run(model_path, image, conf_thres=confidence, nosave=False)

    # Display Original Image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Display Detected Objects
    if results is not None:
        st.write("Detected Objects:")
        for result in results:
            st.write(result)

stframe=st.empty()
kpi1,kpi2,kpi3 = st.columns(3)
with kpi1:
    st.markdown("**Frame Rate**")
    kpi1_text = st.markdown("0")
with kpi2:
    st.markdown("**Tracked Objects**")
    kpi2_text = st.markdown("0")
with kpi1:
    st.markdown("**Width**")
    kpi1_text = st.markdown("0")