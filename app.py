import streamlit as st

from PIL import Image
st.title("Detecting & Tracking Vespa Velutina (Yellow Legged Hornet) Streamlit App")
img = Image.open('C:/AI/AI Project/AIProject-Team7/Images/Bees/Checked and uploaded 13Nov23/abeille_1.jpg')

st.image(img)
video_file = open('C:/AI/AI Project/AIProject-Team7/BeesAndHornets.mp4','rb')

#video_bytes = video_file.read()

st.video(video_file)

