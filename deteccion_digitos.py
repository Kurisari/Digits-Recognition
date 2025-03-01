import cv2
import numpy as np
import pickle
import os

# Obtener la ruta absoluta del archivo modelo_digitos.pkl
ruta_modelo = os.path.join(os.path.dirname(__file__), "modelo_digitos.pkl")

# Cargar el modelo previamente entrenado
with open(ruta_modelo, "rb") as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

def obtener_componentes_conectados(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    cv2.imshow('Imagen Binarizada', thresh)
    
    for i in range(1, num_labels):  
        x, y, w, h, area = stats[i]
        if w > 10 and h > 10:
            subimg = frame[y:y+h, x:x+w]
            subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
            subimg = cv2.resize(subimg, (28, 28))
            subimg = np.reshape(subimg, (1, 784))  # Convertir a formato adecuado
            etiqueta = modelo.predict(subimg)[0]  # Predecir etiqueta
            
            cv2.putText(frame, str(etiqueta), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)         
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    print("Error: No se pudo abrir la cámara.")
else:
    while True:
        # Capturar cuadro por cuadro
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar el cuadro.")
            break
        
        obtener_componentes_conectados(frame)
        cv2.imshow('Video en vivo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()