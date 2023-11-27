import streamlit as st
import os
import glob
import tempfile
import cv2
from IPython.display import display
#from roboflow import Roboflow
from PIL import Image


st.title("Detecting & Tracking Vespa Velutina (Yellow Legged Hornet) Streamlit App")
img = Image.open('C:/AI/AI Project/AIProject-Team7/Images/Bees/Checked and uploaded 13Nov23/abeille_1.jpg')

st.image(img)

st.sidebar.slider("Confidence",min_value=0.0,max_value=1.0,value=0.25)
st.sidebar.markdown('---')


custom_classes = st.sidebar.checkbox('Use custom classes')
assigned_class_id=[]

if custom_classes:
    assigned_class = st.sidebar.multiselect('Select the custom classes',list(names),default='hornet')
    for each in assigned_class:
        assigned_class_id.append(names.index(each))

        
        
    

#demo video
video_file_buffer = st.sidebar.file_uploader("Upload a video",type=["mp4","mov",'avi','asf','m4v'])
Demo_video = 'BeesAndHornets.mp4'
tfflie = tempfile.NamedTemporaryFile(suffix='.mp4',delete=False)

##get input video here:
if not video_file_buffer:
    vid = cv2.VideoCapture(Demo_video)
    tfflie.name = Demo_video
    dem_vid = open(tfflie.name,'rb')
    demo_bytes = dem_vid.read()
    
    st.sidebar.text('Input video')
    st.sidebar.video(demo_bytes)
    
else:
    tfflie.write(video_file_buffer.read())
    dem_vid = open(tfflie.name,'rb')
    demo_bytes=dem_vid.read()
    
    st.sidebar.text('Input Video')
    st.sidebar.video(demo_bytes)
print(tfflie.name)

stframe=st.empty()
st.sidebar.markdown('----')
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

