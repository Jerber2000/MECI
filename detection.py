from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.label import Label
import cv2
from ultralytics import YOLO
import threading
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture

class IconButton(Button):
    def __init__(self, icon_source, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'horizontal'
        # Añadir el icono fuera del botón
        self.icon = Image(source=icon_source, size_hint=(None, None), size=(30, 30))
        self.text_label = Label(text=self.text, color=(1, 1, 1, 1), size_hint=(None, None))
        self.clear_widgets()
        self.add_widget(self.icon)
        self.add_widget(self.text_label)

class MainApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Cargar el modelo YOLO una sola vez
        self.model = YOLO('last.pt')  # Asegúrate de que este archivo esté en el directorio actual

    def build(self):
        # Determinar si el dispositivo es móvil o de escritorio
        is_mobile = Window.width < 600

        # Crear layout principal
        self.main_layout = BoxLayout(orientation='horizontal')

        # Crear el contenedor de botones y configurarlo como sidebar
        self.sidebar_expanded = True
        self.sidebar_width = 200 if self.sidebar_expanded else 50
        self.button_container = BoxLayout(size_hint=(None, 1), width=self.sidebar_width, orientation='vertical',
                                          spacing=10)

        # Configurar color de fondo blanco para el sidebar
        with self.button_container.canvas.before:
            Color(1, 1, 1, 1)  # Color blanco
            self.bg_rect = Rectangle(size=self.button_container.size, pos=self.button_container.pos)

        self.button_container.bind(size=self.update_rect, pos=self.update_rect)

        # Botón para expandir o retraer el sidebar
        self.toggle_button = Button(text='<<' if self.sidebar_expanded else '>>', size_hint=(1, None), height=50,
                                    background_color=(0, 1, 0, 1), color=(1, 1, 1, 1))  # Verde con texto blanco
        self.toggle_button.bind(on_press=self.toggle_sidebar)
        self.button_container.add_widget(self.toggle_button)

        # Botón "detección en tiempo real" con icono
        real_time_button = IconButton(text='Detección en tiempo real', icon_source='camara.jpg',
                                      background_color=(0, 1, 0, 1), color=(1, 1, 1, 1))
        real_time_button.bind(on_press=self.run_detector)
        self.button_container.add_widget(real_time_button)

        # Botón "detección en foto" con icono
        photo_button = IconButton(text='Detección en foto', icon_source='img.jpg',
                                  background_color=(0, 1, 0, 1), color=(1, 1, 1, 1))
        self.button_container.add_widget(photo_button)

        # Botón para cerrar la cámara
        close_camera_button = Button(text='Cerrar Cámara', size_hint=(1, None), height=50,
                                     background_color=(1, 0, 0, 1), color=(1, 1, 1, 1))
        close_camera_button.bind(on_press=self.close_camera)
        self.button_container.add_widget(close_camera_button)

        # Añadir el sidebar al layout principal
        self.main_layout.add_widget(self.button_container)

        # Imagen de fondo y de detección
        self.background = Image(source='images/background.jpg', allow_stretch=True, keep_ratio=False)
        self.main_layout.add_widget(self.background)

        return self.main_layout

    def update_rect(self, *args):
        """Actualizar el tamaño y la posición del fondo blanco en el sidebar."""
        self.bg_rect.size = self.button_container.size
        self.bg_rect.pos = self.button_container.pos

    def toggle_sidebar(self, instance):
        """Alterna entre los modos expandido y retraído del sidebar."""
        self.sidebar_expanded = not self.sidebar_expanded
        self.sidebar_width = 200 if self.sidebar_expanded else 50
        self.button_container.width = self.sidebar_width
        self.toggle_button.text = '<<' if self.sidebar_expanded else '>>'

    def run_detector(self, instance):
        """Iniciar la detección en tiempo real."""
        # Si la cámara ya está en funcionamiento, no hacemos nada
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            return

        # Cambiar a detección en tiempo real y ejecutar en un hilo separado
        detector_thread = threading.Thread(target=self.start_detector)
        detector_thread.daemon = True
        detector_thread.start()

    def start_detector(self):
        """Iniciar la cámara."""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: No se puede abrir la cámara.")
            return

        # Iniciar el bucle de actualización en el hilo de la interfaz
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

    def update_frame(self, dt):
        """Actualizar el frame con detección en tiempo real."""
        # Verificar si la cámara está disponible
        if not hasattr(self, 'cap') or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Realizar la predicción
        results = self.model.predict(source=frame, save=False, conf=0.25)
        class_names = self.model.model.names
        colors = [(0, 255, 0), (255, 0, 0)]

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                class_name = class_names[cls]
                color = colors[cls % len(colors)]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f'{class_name}: {conf:.2f}'
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convertir el frame a una textura y mostrarla en la imagen de Kivy
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.background.texture = texture

    def close_camera(self, instance):
        """Cerrar la cámara y restaurar el estado inicial de la app."""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()  # Liberar la cámara
            self.cap = None

        # Anular el registro de update_frame del reloj de Kivy
        Clock.unschedule(self.update_frame)

        # Restaurar la imagen de fondo original en la interfaz de Kivy
        self.background.texture = None  # Limpiar la textura de detección
        self.background.source = 'images/background.jpg'
        self.background.reload()  # Recargar la imagen de fondo

    def on_stop(self):
        # Liberar la cámara al cerrar la app
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

# Ejecutar la aplicación
if __name__ == '__main__':
    MainApp().run()
