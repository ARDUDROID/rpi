import cv2
import requests
from picamera2 import Picamera2

HOST_URL = "http://<adres_ip_hosta>:5000/upload"

camera = Picamera2()
camera.configure(camera.create_still_configuration(main={"format": "RGB888", "size": (640, 480)}))
camera.start()

image = camera.capture_array()
camera.stop()

image_path = "pcb_image.jpg"
cv2.imwrite(image_path, image)

with open(image_path, "rb") as file:
    response = requests.post(HOST_URL, files={"file": file})

print("Response from host:", response.text)

def preprocess_image(image):
    image_resized = cv2.resize(image, (224, 224))
    image_normalized = image_resized / 255.0
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

from flask import Flask, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

MODEL_PATH = "model.tflite"
model = load_model(MODEL_PATH)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    input_image = preprocess_image(image)

    prediction = model.predict(input_image)

    label = "OK" if np.argmax(prediction) == 0 else "Defect"
    confidence = np.max(prediction)

    with open("label.txt", "a") as label_file:
        label_file.write(f"{label}: {confidence:.2f}\n")

    return f"{label}: {confidence:.2f}", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
