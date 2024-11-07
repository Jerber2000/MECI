import cv2
from ultralytics import YOLO

# Cargar el modelo entrenado
model = YOLO('last.pt')  # Asegúrate de que este archivo esté en el directorio actual

# Obtener los nombres de las clases del modelo
class_names = model.model.names
print("Clases detectadas:", class_names)

# Definir colores para cada clase detectada
colors = [(0, 255, 0), (255, 0, 0)]  # Verde y Azul (en el orden esperado)

# Iniciar la captura de video desde la webcam
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara.")
    exit()

# Bucle principal para procesar la entrada de la cámara
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: No se puede leer el frame de la cámara.")
        break

    # Realizar la predicción
    results = model.predict(source=frame, save=False, conf=0.25)

    # Procesar los resultados y dibujar las cajas delimitadoras
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])  # Índice de la clase
            conf = box.conf[0]     # Nivel de confianza
            class_name = class_names[cls]  # Nombre de la clase

            # Asignar color basado en la clase detectada
            color = colors[cls % len(colors)]  # Selecciona un color basado en el índice

            # Obtener las coordenadas de la caja delimitadora
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Dibujar la caja delimitadora en la imagen
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Preparar el texto a mostrar (nombre de la clase y confianza)
            text = f'{class_name}: {conf:.2f}'

            # Colocar el texto encima de la caja delimitadora
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

    # Mostrar el frame con las anotaciones
    cv2.imshow('Detección de Objetos', frame)

    # Salir del bucle si se presiona 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()