import cv2
import numpy as np
import os

def obtener_componentes_conectados(frame, contador, etiqueta, capturar=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    cv2.imshow('Imagen Binarizada', thresh)
    
    if capturar and not os.path.exists("imagenesDigitos"):
        os.makedirs("imagenesDigitos")
    
    for i in range(1, num_labels):  
        x, y, w, h, area = stats[i]
        if w > 20 and h > 20:
            subimg = frame[y:y+h, x:x+w]
            subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
            subimg = cv2.resize(subimg, (28, 28))
            if capturar:
                cv2.imwrite(f"imagenesDigitos/digito_{etiqueta}_{contador}.png", subimg)
                contador += 1
            cv2.putText(frame, f'Etiqueta {etiqueta}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return contador

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
else:
    contador = 0
    etiqueta = 0  # Asigna el número de la hoja aquí
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el cuadro.")
            break
        
        contador = obtener_componentes_conectados(frame, contador, etiqueta, capturar=False)
        cv2.imshow('Video en vivo', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Presiona 'c' para capturar
            contador = obtener_componentes_conectados(frame, contador, etiqueta, capturar=True)
        elif key == ord('q'):  # Presiona 'q' para salir
            break
cap.release()
cv2.destroyAllWindows()