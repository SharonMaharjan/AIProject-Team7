import streamlit as st
# from obj_det_and_trk import *
from ob_detect import *
from obj_det_and_trk import detect, draw_boxes
from PIL import Image
import shutil
import numpy as np
import psutil
import datetime
import glob
import moviepy.editor as mp
#---------------------------Main Function for Execution--------------------------

def hex_to_bgr (hex_color):
    # Remove the # sign, if present
    hex_color = hex_color.lstrip ('#')
    # Split the hex color into three parts
    red, green, blue = hex_color [0:2], hex_color [2:4], hex_color [4:6]
    # Convert each part from hexadecimal to decimal
    red, green, blue = int (red, 16), int (green, 16), int (blue, 16)
    # Manually assign the colors to the right order in the tuple
    bgr_color = tuple ([blue, green, red])
    # Return the bgr color
    return bgr_color

def hex_to_rgb (hex_color):
    # Remove the # sign, if present
    hex_color = hex_color.lstrip ('#')
    # Split the hex color into three parts
    red, green, blue = hex_color [0:2], hex_color [2:4], hex_color [4:6]
    # Convert each part from hexadecimal to decimal
    red, green, blue = int (red, 16), int (green, 16), int (blue, 16)
    # Put the parts into a tuple
    rgb_color = tuple ([red, green, blue])
    # Return the rgb color
    return rgb_color

color_bee = "#00FFFF"
color_hornet = "#0000FF"
def clear_selections():
    assigned_class_id = []
st.title("Detecting & Tracking Yellow Legged Hornet Streamlit App:female-detective::honeybee:")
st.caption('''
Created by [Asir Faysal Rafin](https://github.com/asirfaysal), [Daan Michielsen](https://github.com/DaanMichielsen) and [Sharon Maharjan](https://github.com/SharonMaharjan)
           ''')
st.subheader("Back story:book:", divider='orange')
bee, hornet = st.columns(2)
with bee:
    img_bee = Image.open('../streamlit_images/bee.jpg')
    st.image(img_bee)
with hornet:
    img_hornet = Image.open('../streamlit_images/hornet.jpg')
    st.image(img_hornet)

with st.expander(label="A beekeeper in Belgium recently reached out...", expanded=False):
    st.markdown("A beekeeper in Belgium recently reached out to us to see if we could help solve his yellow-legged hornet problem using object tracking. The idea is to track the path of the hornet after it eats, as it will fly directly back to the nest. By tracking the hornet’s path, we can locate the nest and remove it. Without the bees the agriculture would suffer tramendously and cause lower yield numbers.")

inference_msg = st.empty()    

st.sidebar.header("Configuration:gear:", divider='orange')
confidence = st.sidebar.slider("Confidence",min_value=0.0,max_value=1.0,value=0.50, help="The confidence of the model must exceed the threshold in order for the object to be detected.")
custom_classes = st.sidebar.checkbox('Use custom classes:round_pushpin:')
assigned_class_id=[]
names = ["Bees","Hornets"]

assigned_class = st.sidebar.multiselect('Select the custom classes',list(names),default='Hornets', disabled=False if custom_classes else True)

for each in assigned_class:
    assigned_class_id.append(names.index(each))

print(assigned_class)
print(assigned_class_id)
if "Bees" in assigned_class:
    color_bee = st.sidebar.color_picker("Bees", value=color_bee, key=None, help="Choose color for bounding boxes of bees", disabled=False if custom_classes else True)
    color_bee_BGR = hex_to_bgr(color_bee)
if "Hornets" in assigned_class:
    color_hornet = st.sidebar.color_picker("Hornets", value=color_hornet, key=None, help="Choose color for bounding boxes of hornets", disabled=False if custom_classes else True)
    color_hornet_BGR = hex_to_bgr(color_hornet)
input_source = st.sidebar.radio("Source",
                                        ('Video:film_frames:', 'WebCam:movie_camera:','Image:frame_with_picture:'))
        
weights = "../weights/bestv4.pt"
device="cpu"

# ------------------------- LOCAL VIDEO ------------------------
if input_source == "Video:film_frames:": 
    st.sidebar.subheader("Video	:film_frames:", divider='orange')
    
    uploaded_video = st.sidebar.file_uploader("Upload a video:outbox_tray:",
                                                 type=["mp4","mov",'avi','asf','m4v'], 
                                                 accept_multiple_files=False)
    save_output_video = st.sidebar.radio("Save output video?:inbox_tray:",
                                ('Yes', 'No'))
    if save_output_video == 'Yes':
        nosave = False
        display_labels=False
    else:
        nosave = True
        display_labels = True 
    video_file = "temp.mp4"
    # video.save(video_file)
    pred_view = st.empty()
    warning = st.empty()
    if uploaded_video != None:
        # Check if the folder exists
        if os.path.exists ("./runs/detect/exp"):
            # Remove the folder and its contents
            shutil.rmtree ("./runs/detect/exp")
        # Save video to disk
        uploaded_video_path = "temp.mp4"
        with open(uploaded_video_path, mode='wb') as f:
            f.write(uploaded_video.read())

        # Display uploaded video
        with open(uploaded_video_path, 'rb') as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.caption(uploaded_video.name)
        is_tracking = st.sidebar.toggle('Tracking')
        if st.sidebar.button(f"Start {'Tracking' if is_tracking else 'Detecting'} {'hornets and bees' if len(assigned_class) == 2 else ('hornets' if 'Hornets' in assigned_class else 'bees')}", disabled=True if len(assigned_class) == 0 else False):
            with st.spinner(f"{'Tracking' if is_tracking else 'Detecting'} objects...:male-detective:"):
                if is_tracking:
                    detect(weights=weights,
                        source=uploaded_video_path,
                        hide_labels=False,
                        hide_conf=False,
                        conf_thres=confidence,
                        device='cpu',
                        classes=assigned_class_id,
                        color_bee=hex_to_bgr(color_bee),
                        color_hornet=hex_to_bgr(color_hornet))
                else:
                    run(weights=weights, 
                        source=uploaded_video_path,  
                        hide_labels=False,
                        hide_conf=False,
                        conf_thres=confidence,
                        device="cpu",
                        classes=assigned_class_id,nosave=nosave,
                        color_bee=hex_to_bgr(color_bee),
                        color_hornet=hex_to_bgr(color_hornet))
                # Define the source and destination files
                source_file = "./runs/detect/exp/temp.mp4"
                destination_file = "./result.mp4"

                # Read the source video file
                clip = mp.VideoFileClip(source_file)

                # Write the destination video file with the WebM codec
                # You can also specify other arguments such as bitrate, audio_codec, etc.
                clip.write_videofile(destination_file, codec="libvpx-vp9")

                # Close the clip
                clip.close()
                st.video(destination_file)
            st.success('Done!', icon="✅")
        if len(assigned_class) == 0:
            st.sidebar.warning("No classes selected")
        

# ------------------------- Webcam ------------------------
if input_source == "WebCam:movie_camera:":
    st.sidebar.subheader("Webcam:movie_camera:", divider='orange')
    run = st.sidebar.toggle('Webcam')
    is_tracking = st.sidebar.toggle('Tracking')
    if st.sidebar.button(f"Start {'Tracking' if is_tracking else 'Detecting'}"):
        
        stframe = st.empty()
        
        st.markdown("""<h4 style="color:black;">
                        Memory Overall Statistics</h4>""", 
                        unsafe_allow_html=True)
        kpi5, kpi6 = st.columns(2)

        with kpi5:
            st.markdown("""<h5 style="color:black;">
                        CPU Utilization</h5>""", 
                        unsafe_allow_html=True)
            kpi5_text = st.markdown("0")
        
        with kpi6:
            st.markdown("""<h5 style="color:black;">
                        Memory Usage</h5>""", 
                        unsafe_allow_html=True)
            kpi6_text = st.markdown("0")
        
        run(weights=weights, 
            source="0",  
            stframe=stframe, 
            kpi5_text=kpi5_text,
            kpi6_text = kpi6_text,
            conf_thres=confidence,
            device="cpu",
            classes=assigned_class_id,nosave=nosave, 
            display_labels=display_labels)
        kpi5_text = str(psutil.virtual_memory()[2])+"%"
        kpi6_text = str(psutil.cpu_percent())+'%'

        inference_msg.success("Inference Complete!")

# ------------------------- Image ------------------------
if input_source == "Image:frame_with_picture:":
    st.sidebar.subheader("Image:frame_with_picture:", divider='orange')
    uploaded_files = st.sidebar.file_uploader("Upload image(s):outbox_tray:", type=["jpg","png"], 
                                              accept_multiple_files=False)
    save_output_image = st.sidebar.radio("Save output video?:inbox_tray:",
                                ('Yes', 'No'))
    if save_output_image == 'Yes':
        nosave = False
        display_labels=False
    else:
        nosave = True
        display_labels = True
    # Check if the file is valid
    if uploaded_files is not None:
        # Check if the folder exists
        if os.path.exists ("./runs/detect/exp"):
            # Remove the folder and its contents
            shutil.rmtree ("./runs/detect/exp")
        # Open the image file
        img = Image.open(uploaded_files)

        # Get the names of the image bands
        bands = img.getbands()

        # Check if the image has an alpha channel
        if 'A' in bands:
            # Convert the image to RGB mode, which removes the alpha channel
            img = img.convert('RGB')
        img_file = "temp.jpg"
        # Save the image as a PIL object, which removes the alpha channel
        img.save(img_file)
        # Display the original image
        original, detected = st.columns(2)
        with original:
            st.subheader("Input", divider='orange')
            st.image(img, caption="Original image", use_column_width=True)
        if st.sidebar.button("Detect objects"):
            with st.spinner(text="Detecting objects:sleuth_or_spy:"):            
                run(weights=weights, 
                source=img_file,
                conf_thres=confidence,
                device="cpu",
                classes=assigned_class_id,nosave=nosave, 
                hide_labels=False,
                hide_conf=False,
                color_bee=hex_to_bgr(color_bee),
                color_hornet=hex_to_bgr(color_hornet))
                # Display the result image with bounding boxes and labels
                with detected:
                    st.subheader("Output", divider='orange')
                    st.image("./runs/detect/exp/temp.jpg", caption="Result image", use_column_width=True)
    else:
        st.write("Please upload a valid image file.")
           
    # --------------------------------------------------------------       
    torch.cuda.empty_cache()
    # --------------------------------------------------------------

