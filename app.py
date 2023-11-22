import streamlit as st
import torch
import os
import glob
from IPython.display import display
from roboflow import Roboflow
from PIL import Image
# Step 1: Install Requirements
st.title("Detecting & Tracking Vespa Velutina (Yellow Legged Hornet) Streamlit App")

