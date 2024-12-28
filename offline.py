# Kod na Raspberry Pi: Przesyłanie zdjęcia do hosta
import cv2
import requests
from picamera2 import Picamera2

# Ustawienia
HOST_URL = "http://<adres_ip_hosta>:5000/upload"  # Zmień na adres hosta

# Inicjalizacja kamery
camera = Picamera2()
camera.configure(camera.create_still_configuration(main={"format": "RGB888", "size": (640, 480)}))
camera.start()

# Zrób zdjęcie
image = camera.capture_array()
camera.stop()

# Zapisz zdjęcie do pliku
image_path = "pcb_image.jpg"
cv2.imwrite(image_path, image)

# Prześlij zdjęcie do hosta
with open(image_path, "rb") as file:
    response = requests.post(HOST_URL, files={"file": file})

print("Response from host:", response.text)

# Kod na hoście: Przetwarzanie zdjęcia
def preprocess_image(image):
    """Funkcja do przetwarzania obrazu przed przekazaniem do modelu."""
    image_resized = cv2.resize(image, (224, 224))  # Dopasuj do wymagań modelu
    image_normalized = image_resized / 255.0  # Normalizacja pikseli
    image_expanded = np.expand_dims(image_normalized, axis=0)
    return image_expanded

from flask import Flask, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Załaduj model TensorFlow
MODEL_PATH = "model.tflite"  # Zamień na ścieżkę do Twojego modelu
model = load_model(MODEL_PATH)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Wczytaj obraz
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Przetwarzanie obrazu
    input_image = preprocess_image(image)

    # Klasyfikacja obrazu
    prediction = model.predict(input_image)

    # Interpretacja wyniku
    label = "OK" if np.argmax(prediction) == 0 else "Defect"
    confidence = np.max(prediction)

    return f"{label}: {confidence:.2f}", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
