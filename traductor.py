import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# Function to map letters to gestures
def text_to_sign(text):
    text = text.upper()
    gestures = []
    for char in text:
        if char.isalpha():
            gestures.append(char)
    return gestures

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

# Nuevas variables para el manejo de la palabra formada
current_word = ""

start_time = time.time()

while True:
    success, img = cap.read()
    imgOutput = img.copy()

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if not imgCrop.size:
            continue

        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)

            imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape

            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)

            imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)

        # Convertir la predicción a cadena
        prediction_str = str(prediction)

        cv2.putText(imgOutput, prediction_str, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.rectangle(imgOutput, (10, 10), (220, 80), (255, 0, 255), cv2.FILLED)
    cv2.putText(imgOutput, "Deteccion de Gestos", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Deteccion de gestos", imgOutput)

    key = cv2.waitKey(1)

    # Agregar letra a la palabra formada si se detecta una nueva letra
    if key == ord("s") and hands:
        counter += 1
        cv2.imwrite(f'{folder}/Image_{counter}.jpg', imgWhite)
        print(f"Guardada la imagen {counter}")

        current_word += prediction_str  # Agregar la letra a la palabra formada

        # Reiniciar el temporizador
        start_time = time.time()

    # Mostrar la palabra formada en la consola después de 4 segundos
    if time.time() - start_time >= 4 and current_word:
        print(f"Palabra formada: {current_word}")

        # Reiniciar la palabra formada
        current_word = ""

    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
