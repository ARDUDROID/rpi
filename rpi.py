import cv2
import numpy as np
import tensorflow as tf
from picamera2 import Picamera2, Preview

# Ścieżka do modelu TensorFlow Lite
MODEL_PATH = "model.tflite"

# Funkcja do ładowania modelu TensorFlow Lite
def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Funkcja do przetwarzania obrazu (skalowanie, normalizacja)
def preprocess_frame(frame, img_size=(224, 224)):
    resized_frame = cv2.resize(frame, img_size)
    normalized_frame = resized_frame.astype(np.float32) / 255.0  # Normalizacja
    return np.expand_dims(normalized_frame, axis=0)  # Dodanie wymiaru batcha

# Funkcja do detekcji uszkodzeń
def detect_defects(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Funkcja do rysowania wyników na klatce obrazu
def draw_results(frame, result, defect_types):
    predicted_index = np.argmax(result)
    label = defect_types[predicted_index]
    confidence = result[0][predicted_index] * 100

    # Dodanie tekstu na obrazie
    text = f"{label} ({confidence:.2f}%)"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame

# Główna funkcja analizy na żywo
def live_analysis():
    # Inicjalizacja kamery Raspberry Pi
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration()
    picam2.configure(preview_config)

    # Ładowanie modelu
    interpreter = load_model(MODEL_PATH)

    # Typy defektów
    defect_types = ["Brak defektu", "Przerwane ścieżki", "Mostki lutownicze", "Zimne luty"]

    # Start kamery
    picam2.start_preview(Preview.QTGL)
    picam2.start()

    try:
        while True:
            # Przechwycenie klatki
            frame = picam2.capture_array()

            # Przetwarzanie obrazu
            input_image = preprocess_frame(frame)
            result = detect_defects(interpreter, input_image)

            # Rysowanie wyników na klatce
            frame_with_results = draw_results(frame, result, defect_types)

            # Wyświetlenie obrazu
            cv2.imshow("Analiza na żywo - Wykrywanie uszkodzeń PCB", frame_with_results)

            # Przerwanie pętli po naciśnięciu klawisza 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("\nZatrzymano analizę na żywo.")
    finally:
        # Zatrzymanie kamery i zamknięcie okien
        picam2.stop()
        cv2.destroyAllWindows()

# Uruchomienie analizy na żywo
if __name__ == "__main__":
    live_analysis()
