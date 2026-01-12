<img width="993" height="409" alt="Screenshot 2026-01-12 143812" src="https://github.com/user-attachments/assets/f4e73eea-3aa4-4bb7-bb97-cf571f188b60" />
# 🚀 RT-DETR Real-Time Object Detection App

This project demonstrates the power of **RT-DETR (Real-Time Detection Transformer)**, a state-of-the-art object detection model by Ultralytics. The application is wrapped in a user-friendly web interface using **Streamlit**, allowing users to upload images and detect objects instantly.

## 🌟 Features
* **Transformer Power:** Uses the RT-DETR model (rtdetr-l.pt) which combines high accuracy with real-time speed.
* **Interactive UI:** Built with Streamlit for easy image uploading and visualization.
* **Custom Controls:** Adjust the **Confidence Threshold** via a sidebar slider to filter detections.
* **Detailed Insights:** Displays the original image alongside the detection results with bounding boxes and labels.

## 🛠️ Tech Stack
* **Python** 3.x
* **Streamlit** (Web UI)
* **Ultralytics** (Model Inference)
* **OpenCV & Pillow** (Image Processing)
* **NumPy**

## ⚙️ Installation & Setup

Follow these steps to run the project locally:

**1. Clone the Repository**
```bash
git clone [https://github.com/Gopalraj248/RT-DETR-Object-detection-.git](https://github.com/Gopalraj248/RT-DETR-Object-detection-.git)
cd RT-DETR-Object-detection-

2. Create a Virtual Environment (Recommended)
# For Windows
python -m venv venv
venv\Scripts\activate

# For Linux/Mac
python3 -m venv venv
source venv/bin/activate


3. Install Dependencie
pip install -r requirements.txt

🚀 How to Run
streamlit run app_ui.py



![alt text](image-1.png)
