import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import cv2

# Page Configuration
st.set_page_config(page_title="RT-DETR Object Detection", layout="wide")

st.title("🚀 RT-DETR Real-Time Object Detection")
st.write("Image upload karein aur dekhein RT-DETR kaise objects pehchanta hai.")

# 1. Model Load Karein (Caching use kar rahe hain taaki baar baar load na ho)
@st.cache_resource
def load_model():
    return RTDETR("rtdetr-l.pt")

model = load_model()

# 2. Sidebar for Settings
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)

# 3. File Uploader
uploaded_file = st.file_uploader("Image chuniye...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Image ko open karein
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Result")
        
        # Inference (Prediction)
        results = model(img_array, conf=conf_threshold)
        
        # Plotting the results on image
        # plot() function boxes aur labels draw kar deta hai
        res_plotted = results[0].plot()
        
        # RGB format mein convert karke dikhayein
        st.image(res_plotted, channels="BGR", use_container_width=True)

    # 4. Detailed Results Table
    st.divider()
    st.subheader("Detections Data")
    
    detections = []
    for box in results[0].boxes:
        detections.append({
            "Object": model.names[int(box.cls)],
            "Confidence": f"{float(box.conf):.2f}"
        })
    
    if detections:
        st.table(detections)
    else:
        st.write("Koi object nahi mila. Confidence threshold kam karke dekhein.")