from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.core.window import Window
import cv2
from ultralytics import YOLO
import threading

class MainApp(App):
    def build(self):
        # Determinar si el dispositivo es móvil o de escritorio
        is_mobile = Window.width < 600  # Cambia este valor según el tamaño que consideres móvil

        # Crear layout principal
        layout = BoxLayout(orientation='horizontal')

        # Crear el contenedor de botones
        button_container = BoxLayout(size_hint=(None, None), size=(200, 100))

        # Si es computadora, los botones estarán en el lado izquierdo y en la parte superior
        if not is_mobile:
            button_container.orientation = 'vertical'
            button_container.pos_hint = {'x': 0, 'top': 1}
            button_container.spacing = 20
        # Si es móvil, los botones estarán centrados
        else:
            button_container.orientation = 'horizontal'
            button_container.pos_hint = {'center_x': 0.5, 'center_y': 0.5}

        # Botón "detección en tiempo real"
        real_time_button = Button(text='Detección en tiempo real', background_color=(0.5, 0, 0.5, 1), color=(1, 1, 1, 1))
        real_time_button.bind(on_press=self.run_detector)  # Vinculamos la acción del botón
        button_container.add_widget(real_time_button)

        # Botón "detección en foto"
        photo_button = Button(text='Detección en foto', background_color=(0.5, 0, 0.5, 1), color=(1, 1, 1, 1))
        button_container.add_widget(photo_button)

        # Añadir los botones al layout
        layout.add_widget(button_container)

        # Añadir la imagen de fondo desde la ruta img/img1.png
        background = Image(source='images/background.jpg', allow_stretch=True, keep_ratio=False)
        layout.add_widget(background)

        return layout

    def run_detector(self, instance):
        """Este método se ejecuta al presionar el botón de detección en tiempo real."""
        # Ejecutamos el detector en un hilo para no bloquear la interfaz de usuario
        detector_thread = threading.Thread(target=self.start_detector)
        detector_thread.daemon = True  # Hacemos que el hilo termine cuando el programa principal termine
        detector_thread.start()

    def start_detector(self):
        """Método que se ejecuta en un hilo separado para iniciar la detección."""
        self.detect_objects()

    def detect_objects(self):
        """Función de detección en tiempo real con OpenCV y YOLO."""
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
            return

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

# Corregido para que se ejecute correctamente cuando el archivo se ejecute directamente
if __name__ == '__main__':
    MainApp().run()