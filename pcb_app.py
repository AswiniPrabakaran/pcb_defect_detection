import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load model
model = load_model("C:/Users/Asus/Desktop/pcb/pcb_model.h5")

# App title
st.set_page_config(page_title="PCB Defect Detector", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 PCB Defect Detection</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Upload a PCB image to check if it's defected</h4>", unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("📂 About")
st.sidebar.info("""
This AI model detects if a PCB (Printed Circuit Board) is **defected** or **non-defected**.

- Built using Keras & TensorFlow  
- Input: PCB image (.jpg, .png)
- Output: AI prediction result

""")

# File uploader
uploaded_file = st.file_uploader("📸 Upload a PCB Image", type=["jpg", "jpeg", "png"])

# Prediction logic
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='🔍 Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((200, 200))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    label = "🛑 Defected" if prediction[0][0] < 0.5 else "✅ Non-Defected"
    color = "#ff4c4c" if "Defected" in label else "#4CAF50"

    st.markdown(f"<h2 style='text-align: center; color: {color};'>{label}</h2>", unsafe_allow_html=True)

else:
    st.warning("Please upload a PCB image to begin.")

# Footer
st.markdown("""
<hr>
<p style='text-align: center; font-size: 13px;'>Developed with ❤️ for PCB quality assurance</p>
""", unsafe_allow_html=True)
