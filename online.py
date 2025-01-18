import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
import time
import cv2
import json

# Function to load labels from a JSON file
def load_labels(label_file_path):
    try:
        with open(label_file_path, 'r') as file:
            data = json.load(file)
            return data.get("labels", [])
    except Exception as e:
        print(f"Error loading labels: {e}")
        return []

# Preprocess the image for the model
def preprocess_image(frame, input_shape):
    if len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    elif len(frame.shape) == 3 and frame.shape[2] == 3:  # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    normalized_frame = (resized_frame / 255.0 * 127 - 128).astype('int8')
    return np.expand_dims(normalized_frame, axis=0)

# Interpret the output of the model
def interpret_output(output_data, labels):
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)
    label = labels[predicted_class] if labels and predicted_class < len(labels) else "Unknown"
    return label, confidence

# Main function
def main():
    labels = load_labels("label.json")  # Load labels from the .json file
    if not labels:
        print("No labels loaded. Check the label.json file.")
        return

    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(2)

    try:
        interpreter = tf.lite.Interpreter(model_path="model_int8.tflite")
        interpreter.allocate_tensors()
    except ValueError as e:
        print(f"Failed to load the model: {e}")
        return

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']

    print("Model loaded successfully.")

    try:
        while True:
            frame = picam2.capture_array()

            try:
                input_data = preprocess_image(frame, input_shape)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]

                label, confidence = interpret_output(output_data, labels)
                print(f"Label: {label}, Confidence: {confidence:.2f}")

                # Display label and confidence on the frame
                cv2.putText(frame, f"Label: {label}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow("Frame", frame)

            except Exception as e:
                print(f"Error processing frame: {e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("Camera stopped.")

if __name__ == "__main__":
    main()

