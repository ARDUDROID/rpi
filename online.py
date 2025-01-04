import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
import time
import cv2

#def convert_model_to_int8(input_model_path, output_model_path):
    # Ładowanie modelu float32
#    converter = tf.lite.TFLiteConverter.from_saved_model(input_model_path)
#    converter.optimizations = [tf.lite.Optimize.DEFAULT]
#    converter.target_spec.supported_types = [tf.int8]
#    converter.inference_input_type = tf.int8
#    converter.inference_output_type = tf.int8

    # Konwersja modelu
#    tflite_model = converter.convert()

    # Zapis modelu INT8
#    with open(output_model_path, "wb") as f:
#        f.write(tflite_model)
#    print(f"Model zapisany jako {output_model_path}")

# Funkcja do przeskalowania obrazu na odpowiedni format dla modelu INT8
def preprocess_image(frame, input_shape):
    # Convert BGR to RGB if needed
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


def interpret_output(output_data):
    
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)
    return predicted_class, confidence


def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

# Funkcja do wykonania autofocus
#def autofocus(picam2):
#    try:
#        # Set AF mode to Auto
#        picam2.set_controls({"AfMode": controls.AfModeEnum.Auto})
#        time.sleep(2)  # Give time for AF to work

        # Capture frame and calculate sharpness
#        frame = picam2.capture_array()
#        sharpness = calculate_sharpness(frame)
#        print(f"Autofocus completed. Sharpness: {sharpness:.2f}")

        # Set AF mode back to Manual to maintain focus
#        picam2.set_controls({"AfMode": controls.AfModeEnum.Manual})

#    except Exception as e:
#        print(f"Autofocus error: {e}")


def main():
    
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(2)  

    
#    autofocus(picam2)

    
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
               
                predicted_class, confidence = interpret_output(output_data)
                print(f"Predykcja: {predicted_class}, Pewność: {confidence:.2f}")

            except Exception as e:
                print(f"Błąd podczas przetwarzania obrazu: {e}")

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        picam2.stop()
        print("Kamera zatrzymana.")

if __name__ == "__main__":
    # convert_model_to_int8("path_to_saved_model", "model_int8.tflite")
    main()
