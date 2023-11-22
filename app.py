import streamlit as st
import os
from PIL import Image
from roboflow import Roboflow
import torch
import glob
from IPython.display import display

# Step 1: Install Requirements
st.title("Detecting & Tracking Vespa Velutina (Yellow Legged Hornet) Streamlit App")

