import streamlit as st
from ultralytics import RTDETR
from PIL import Image
import numpy as np
import cv2
import easyocr

st.set_page_config(page_title="Number Plate Detector", layout="wide")
st.title("🆔 License Plate Recognition (RT-DETR)")

@st.cache_resource
def load_models():
    # 'rtdetr-l.pt' ki jagah hum Number Plate specific model use karenge
    # Industry tip: Agar aapke paas custom model nahi hai, 
    # toh 'keremberke/license-plate-object-detection-rtdetr' jaise models best hote hain.
    model = RTDETR("rtdetr-l.pt") # Yahan apna custom model .pt file rakhein
    reader = easyocr.Reader(['en'])
    return model, reader

model, reader = load_models()

uploaded_file = st.file_uploader("Gaadi ki photo upload karein...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Prediction (Keval Number Plate detect karne ke liye)
    results = model(img_array, conf=0.4)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Number Plate Detection")
        
        # Sirf Number Plate ko dhoondna
        for result in results[0].boxes:
            # Agar aapka model trained hai, toh class 0 plate hogi
            # Hum yahan crop kar rahe hain
            x1, y1, x2, y2 = map(int, result.xyxy[0])
            plate_crop = img_array[y1:y2, x1:x2]
            
            if plate_crop.size > 0:
                # Image processing for better OCR
                gray_plate = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
                
                # Text Read karna
                ocr_res = reader.readtext(gray_plate)
                plate_no = " ".join([res[1] for res in ocr_res]).upper()
                
                # Box Draw karna (Sirf Plate par)
                cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_array, plate_no, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                st.image(plate_crop, caption="Cropped Plate", width=300)
                st.success(f"Number Plate Text: {plate_no}")

        st.image(img_array, caption="Final Detection", use_container_width=True)