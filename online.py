import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
import time
import cv2
import tkinter as tk
from tkinter import messagebox
import json

# Function to load labels from a JSON file
def load_labels(label_file_path):
    with open(label_file_path, 'r') as file:
        data = json.load(file)
        labels = data.get("labels", [])
    return labels

def preprocess_image(frame, input_shape):
    if len(frame.shape) == 3 and frame.shape[2] == 4:  # If RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))

    if len(resized_frame.shape) == 2:
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)

    normalized_frame = (resized_frame / 255.0 * 127 - 128).astype('int8')

    if len(normalized_frame.shape) != 4:
        normalized_frame = np.expand_dims(normalized_frame, axis=0)

    return normalized_frame

def interpret_output(output_data, labels):
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)
    label = labels[predicted_class] if labels else "Unknown"
    return label, confidence

def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def show_popup(label, confidence):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    messagebox.showinfo("Prediction", f"Label: {label}\nConfidence: {confidence:.2f}")
    root.destroy()

def main():
    labels = load_labels("label.json")  # Load labels from the .json file

    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(2)

    try:
        interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
        interpreter.allocate_tensors()
    except ValueError as e:
        print(f"Nie udało się wczytać modelu: {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']

    print("Model załadowany pomyślnie.")

    try:
        while True:
            frame = picam2.capture_array()

            try:
                input_data = preprocess_image(frame, input_shape)

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                output_data = interpreter.get_tensor(output_details[0]['index'])[0]

                label, confidence = interpret_output(output_data, labels)
                print(f"Label: {label}, Precyzja: {confidence:.2f}")

                # Show pop-up window with prediction
                show_popup(label, confidence)

                cv2.putText(frame, f"Precyzja: {confidence:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, f"Label: {label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.imshow("Frame", frame)

            except Exception as e:
                print(f"Błąd podczas przetwarzania obrazu: {e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("Kamera zatrzymana.")

if __name__ == "__main__":
    main()
