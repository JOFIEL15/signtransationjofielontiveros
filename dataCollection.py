import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

captura = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 500  # Ajusta el tamaño de la imagen blanca
counter = 0
folder = "Data/HOLA"

capturing = False
capture_frames = []

while True:
    success, img = captura.read()
    manos, img = detector.findHands(img)

    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255  # Inicializar la imagen blanca
    imgcrop = np.zeros_like(img)  # Inicializar imgcrop para evitar el error

    for mano in manos:
        x, y, w, h = mano['bbox']

        # Verificar si la mano está cerca del borde de la pantalla
        if x - offset >= 0 and y - offset >= 0 and x + w + offset <= img.shape[1] and y + h + offset <= img.shape[0]:
            imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgcrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgcrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

    cv2.imshow("imagen recortada", imgcrop)
    cv2.imshow("imagen blanco", imgWhite.astype(np.uint8))

    if capturing:
        capture_frames.append(imgWhite)
        print("Capturing...")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == ord("a") and len(manos) == 2:  # Captura solo si hay dos manos presentes
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

    if key == ord("s"):
        if not capturing:
            capturing = True
            capture_frames = []  # Limpiar los fotogramas anteriores
            print("Capture started.")
        else:
            capturing = False
            # Guardar los fotogramas capturados
            for i, frame in enumerate(capture_frames):
                cv2.imwrite(f'{folder}/Capture_{time.time()}_{i}.jpg', frame)
            capture_frames = []  # Limpiar la lista después de guardar las imágenes
            print("Capture stopped.")

    if key == 27:  # Tecla Esc para salir
        if capturing:
            capturing = False
            capture_frames = []  # Limpiar los fotogramas anteriores
        break

cv2.destroyAllWindows()
captura.release()
