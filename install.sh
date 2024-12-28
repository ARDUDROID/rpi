#!/bin/bash

# Funkcja do animacji "trójkąt" podczas instalacji
install_animation() {
    while true; do
        for i in {1..3}; do
            echo -n "."
            sleep 0.5
        done
        break
    done
}

# Instalacja pakietów systemowych
echo "Instalowanie pakietów systemowych..."
install_animation

# Upewnienie się, że system jest zaktualizowany
sudo apt update -y
sudo apt upgrade -y
install_animation

# Instalowanie bibliotek potrzebnych do obsługi kamery Raspberry Pi
sudo apt install -y \
    libcamera-apps \
    libcamera0 \
    libcamera-dev \
    python3-picamera2 \
    python3-opencv \
    python3-dev \
    python3-picamera \
    libatlas-base-dev \
    libprotobuf-dev \
    libprotoc-dev \
    protobuf-compiler

# Instalowanie pakietów Qt i PyQt5
sudo apt install -y \
    qt5-qmake qtbase5-dev qtchooser qtbase5-dev-tools \
    libqt5webkit5-dev libqt5svg5-dev

echo "Pakiety systemowe zostały zainstalowane."

# Instalacja pip
echo "Instalowanie pip..."
install_animation
python3 -m ensurepip --upgrade
install_animation

# Instalacja wymaganych bibliotek Python
echo "Instalowanie bibliotek Python..."
install_animation
pip install --upgrade pip --break-system-packages
pip install  numpy tensorflow picamera2 opencv-python flask --break-system-packages

echo "Wszystkie wymagane biblioteki zostały zainstalowane."

# Potwierdzenie zakończenia
echo "Gotowe! Program jest gotowy do uruchomienia."

echo "Uruchamianie programu .." 

read -p "Wprowadź swoje wybory: " choice

for aplikacja in (echo $choice | tr "," "\n"); do
    case $aplikacja in
        "1") echo 'Uruchamianie aplikacji w trybie live'
            python3 online.py
            ;;
        "2") echo 'Uruchamianie aplikacji w trybie offline (rpi jako serwer a komputer jako klient)'
            python3 offline.py
            ;;
            *)
            echo "Nieprawidłowy wybór."
            ;;
    esac
done
