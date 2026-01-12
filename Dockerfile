# Python base image
FROM python:3.11-slim

# System libraries (OpenCV ke liye zaroori)
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

WORKDIR /app
COPY . /app

# Dependencies install karein
RUN pip install --no-cache-dir -r requirements.txt

# Port 8000 kholien
EXPOSE 8000

# API start karein
CMD ["uvicorn", "app_ui:app", "--host", "0.0.0.0", "--port", "8000"]