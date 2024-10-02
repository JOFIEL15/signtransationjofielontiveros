import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3

# Function to map letters to gestures
def text_to_sign(text):
    text = text.upper()
    gestures = []
    for char in text:
        if char.isalpha():
            gestures.append(char)
    return gestures

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

folder = "Data/C"
counter = 0

labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "HOLA", "ADIOS", "BIEN", "MAL", "AVION"]

cv2.namedWindow("Deteccion de gestos", cv2.WINDOW_NORMAL)
cv2.namedWindow("Referencia de Gestos", cv2.WINDOW_NORMAL)

# Load a fixed reference image using a raw string
reference_image = cv2.imread("imagen.png")

# Check if the reference image is loaded successfully
if reference_image is None or reference_image.shape[0] == 0 or reference_image.shape[1] == 0:
    print("Error: Unable to load the reference image or invalid image dimensions.")
    exit()

# Print the shape of the loaded image for debugging
print(reference_image.shape)

# Explicitly resize the window
cv2.namedWindow("Referencia de Gestos", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Referencia de Gestos", reference_image.shape[1], reference_image.shape[0])

start_time = None
show_letter = False
current_word = ""  # Variable para almacenar la palabra actual

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

            if 0 <= index < len(labels):
                if start_time is None:
                    start_time = time.time()
                    show_letter = True

                    # Agregar la letra a la palabra actual
                    current_word = labels[index] + current_word
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)

            imgCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape

            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if 0 <= index < len(labels):
                if start_time is None:
                    start_time = time.time()
                    show_letter = True

                    # Agregar la letra a la palabra actual
                    current_word = labels[index] + current_word
            else:
                if start_time is None:
                    start_time = time.time()
                    show_letter = True

                    # Agregar la letra "A" a la palabra actual
                    current_word = "A" + current_word

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

        if show_letter and time.time() - start_time > 2:
            cv2.imshow("Letra", cv2.putText(np.zeros_like(imgOutput), current_word,
                                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2))

            # Después de 2 segundos, reiniciar para la siguiente letra
            show_letter = False
            start_time = None

    cv2.rectangle(imgOutput, (10, 10), (220, 80), (255, 0, 255), cv2.FILLED)
    cv2.putText(imgOutput, "Deteccion de Gestos", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                cv2.LINE_AA)

    # Display the fixed reference image
    cv2.imshow("Referencia de Gestos", reference_image)

    cv2.imshow("Deteccion de gestos", imgOutput)

    key = cv2.waitKey(1)

    if key == ord("s") and hands:
        counter += 1
        cv2.imwrite(f'{folder}/Image_{counter}.jpg', imgWhite)
        print(f"Guardada la imagen {counter}")

    if key == ord("f"):
        # Leer el texto acumulado al presionar la tecla "f"
        speak_text(current_word)

    # Restablecer la palabra actual si se presiona la tecla "x"
    if key == ord("x"):
        current_word = ""

    if key == 27:
        break

cv2.destroyAllWindows()
cap.release()
