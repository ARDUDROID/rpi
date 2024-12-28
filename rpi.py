import numpy as np
import tensorflow as tf
from picamera2 import Picamera2
import time
import cv2

def convert_model_to_int8(input_model_path, output_model_path):
    # Ładowanie modelu float32
    converter = tf.lite.TFLiteConverter.from_saved_model(input_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Konwersja modelu
    tflite_model = converter.convert()

    # Zapis modelu INT8
    with open(output_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model zapisany jako {output_model_path}")

# Funkcja do przeskalowania obrazu na odpowiedni format dla modelu INT8
def preprocess_image(frame, input_shape):
    # Zmiana rozmiaru obrazu na wymagany przez model
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    # Normalizacja i konwersja do INT8
    normalized_frame = (resized_frame / 255.0 * 127 - 128).astype('int8')
    return normalized_frame

# Funkcja do interpretacji wyników
def interpret_output(output_data):
    # Przykład interpretacji wyników (dostosuj do swojego modelu)
    predicted_class = np.argmax(output_data)
    confidence = np.max(output_data)
    return predicted_class, confidence

def main():
    # Inicjalizacja kamery
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(2)  # Poczekaj, aż kamera się ustabilizuje

    # Wczytanie modelu
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
            # Pobranie klatki z kamery
            frame = picam2.capture_array()

            # Przygotowanie obrazu
            try:
                input_data = preprocess_image(frame, input_shape)
                input_data = np.expand_dims(input_data, axis=0)  # Dodanie wymiaru batch

                # Klasyfikacja obrazu
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()

                # Pobranie wyników
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]

                # Interpretacja wyników
                predicted_class, confidence = interpret_output(output_data)
                print(f"Predykcja: {predicted_class}, Pewność: {confidence:.2f}")

            except Exception as e:
                print(f"Błąd podczas przetwarzania obrazu: {e}")

            # Wyjście po naciśnięciu klawisza 'q'
            if hasattr(cv2, 'waitKey'):
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("cv2.waitKey is not available in the cv2 module")
                break

    finally:
        picam2.stop()
        print("Kamera zatrzymana.")

if __name__ == "__main__":
    # Jeśli potrzeba skonwertować model, odkomentuj poniższą linię i ustaw ścieżki
    # convert_model_to_int8("path_to_saved_model", "model_int8.tflite")
    main()
