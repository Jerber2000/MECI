from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.filechooser import FileChooserListView
from kivy.properties import NumericProperty
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
import cv2
from ultralytics import YOLO
import threading

class IconButton(BoxLayout):
    def __init__(self, icon_source, text='', **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (1, 1)
        self.spacing = 5
        self.padding = 5

        # Imagen del ícono
        self.icon = Image(source=icon_source, size_hint=(1, 0.8), allow_stretch=True)
        # Etiqueta de texto centrada
        self.label = Label(text=text, size_hint=(1, 0.2), color=(0, 0, 0, 1), font_size='14sp', halign='center')
        self.label.bind(size=self.label.setter('text_size'))

        self.add_widget(self.icon)
        self.add_widget(self.label)

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            if hasattr(self, 'on_press'):
                self.on_press()
            return True
        return super().on_touch_down(touch)

class MainApp(App):
    # Contadores para cada clase
    reciclable_count = NumericProperty(0)
    no_reciclable_count = NumericProperty(0)
    organico_count = NumericProperty(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = YOLO('last.pt')  # Asegúrate de que este archivo esté en el directorio actual

    def build(self):
        # Layout principal
        self.main_layout = BoxLayout(orientation='horizontal')

        # Sidebar izquierdo
        self.left_layout = BoxLayout(orientation='vertical', size_hint=(0.2, 1))

        # Fondo del sidebar izquierdo (blanco)
        with self.left_layout.canvas.before:
            Color(1, 1, 1, 1)
            self.bg_rect_left = Rectangle(size=self.left_layout.size, pos=self.left_layout.pos)

        self.left_layout.bind(size=self.update_rect_left, pos=self.update_rect_left)

        # Botón de ocultar/mostrar el sidebar izquierdo
        self.left_toggle_button = Button(text='<<', size_hint=(1, None), height=50,
                                         background_color=(0, 1, 0, 1), color=(1, 1, 1, 1))
        self.left_toggle_button.bind(on_press=self.toggle_left_sidebar)
        self.left_layout.add_widget(self.left_toggle_button)

        # Botones que ocupan todo el espacio disponible
        button_layout = GridLayout(cols=1, spacing=10, padding=10)
        button_layout.size_hint = (1, 1)

        # Botón de cámara en tiempo real con ícono y texto
        real_time_button = IconButton(icon_source='images/camara.png', text='Cámara')
        real_time_button.on_press = self.run_detector
        button_layout.add_widget(real_time_button)

        # Botón de carga de imagen con ícono y texto
        photo_button = IconButton(icon_source='images/img.png', text='Cargar Imagen')
        photo_button.on_press = self.detect_image
        button_layout.add_widget(photo_button)

        # Botón para cerrar la cámara con ícono y texto
        close_camera_button = IconButton(icon_source='images/close.png', text='Cerrar Cámara')
        close_camera_button.on_press = self.close_camera
        button_layout.add_widget(close_camera_button)

        # Añadir los botones al sidebar izquierdo
        self.left_layout.add_widget(button_layout)

        # Añadir el sidebar izquierdo al layout principal
        self.main_layout.add_widget(self.left_layout)

        # Layout central para la imagen
        self.center_layout = BoxLayout(orientation='vertical')
        self.background = Image(source='images/background.jpg', allow_stretch=True, keep_ratio=False)
        self.center_layout.add_widget(self.background)

        # Añadir el layout central al layout principal
        self.main_layout.add_widget(self.center_layout)

        # Sidebar derecho para mostrar los contenedores de imágenes
        self.right_layout = BoxLayout(orientation='vertical', size_hint=(0.2, 1), padding=10, spacing=10)

        # Fondo del sidebar derecho (blanco)
        with self.right_layout.canvas.before:
            Color(1, 1, 1, 1)
            self.bg_rect_right = Rectangle(size=self.right_layout.size, pos=self.right_layout.pos)

        self.right_layout.bind(size=self.update_rect_right, pos=self.update_rect_right)

        # Botón de ocultar/mostrar el sidebar derecho
        self.right_toggle_button = Button(text='>>', size_hint=(1, None), height=50,
                                          background_color=(0, 1, 0, 1), color=(1, 1, 1, 1))
        self.right_toggle_button.bind(on_press=self.toggle_right_sidebar)
        self.right_layout.add_widget(self.right_toggle_button)

        # Crear contenedores para cada clase
        # Reciclable
        self.reciclable_container = BoxLayout(orientation='vertical', spacing=5)
        self.reciclable_image = Image(source='images/reciclable.jpeg', allow_stretch=True, keep_ratio=True)
        self.reciclable_count_label = Label(text='0', color=(0, 0, 0, 1), font_size='20sp')
        self.reciclable_container.add_widget(self.reciclable_image)
        self.reciclable_container.add_widget(self.reciclable_count_label)

        # No Reciclable
        self.no_reciclable_container = BoxLayout(orientation='vertical', spacing=5)
        self.no_reciclable_image = Image(source='images/no_reciclable.jpeg', allow_stretch=True, keep_ratio=True)
        self.no_reciclable_count_label = Label(text='0', color=(0, 0, 0, 1), font_size='20sp')
        self.no_reciclable_container.add_widget(self.no_reciclable_image)
        self.no_reciclable_container.add_widget(self.no_reciclable_count_label)

        # Orgánico
        self.organico_container = BoxLayout(orientation='vertical', spacing=5)
        self.organico_image = Image(source='images/organico.jpeg', allow_stretch=True, keep_ratio=True)
        self.organico_count_label = Label(text='0', color=(0, 0, 0, 1), font_size='20sp')
        self.organico_container.add_widget(self.organico_image)
        self.organico_container.add_widget(self.organico_count_label)

        # Añadir los contenedores al sidebar derecho
        self.right_layout.add_widget(self.reciclable_container)
        self.right_layout.add_widget(self.no_reciclable_container)
        self.right_layout.add_widget(self.organico_container)

        # Añadir el sidebar derecho al layout principal
        self.main_layout.add_widget(self.right_layout)

        return self.main_layout

    def update_rect_left(self, *args):
        self.bg_rect_left.size = self.left_layout.size
        self.bg_rect_left.pos = self.left_layout.pos

    def update_rect_right(self, *args):
        self.bg_rect_right.size = self.right_layout.size
        self.bg_rect_right.pos = self.right_layout.pos

    def toggle_left_sidebar(self, instance):
        if self.left_layout.size_hint_x > 0:
            self.left_layout.size_hint_x = 0
            self.left_toggle_button.text = '>>'
        else:
            self.left_layout.size_hint_x = 0.2
            self.left_toggle_button.text = '<<'

    def toggle_right_sidebar(self, instance):
        if self.right_layout.size_hint_x > 0:
            self.right_layout.size_hint_x = 0
            self.right_toggle_button.text = '<<'
        else:
            self.right_layout.size_hint_x = 0.2
            self.right_toggle_button.text = '>>'

    def run_detector(self):
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            return

        detector_thread = threading.Thread(target=self.start_detector)
        detector_thread.daemon = True
        detector_thread.start()

    def start_detector(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: No se puede abrir la cámara.")
            return

        # Iniciar el bucle de actualización
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

    def update_frame(self, dt):
        if not hasattr(self, 'cap') or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # Restablecer los contadores
        self.reciclable_count = 0
        self.no_reciclable_count = 0
        self.organico_count = 0

        results = self.model.predict(source=frame, save=False, conf=0.25)
        class_names = self.model.model.names

        # Colores específicos para cada clase
        color_dict = {
            'Reciclable': (255, 0, 0),      # Azul en BGR
            'No_Reciclable': (0, 0, 255),   # Rojo en BGR
            'Organico': (0, 255, 0)         # Verde en BGR
        }

        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]
                class_name = class_names[cls]
                color = color_dict.get(class_name, (255, 255, 255))  # Blanco por defecto
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Dibujar rectángulo y etiqueta
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                text = f'{class_name}: {conf:.2f}'
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Actualizar contadores
                if class_name == 'Reciclable':
                    self.reciclable_count += 1
                elif class_name == 'No_Reciclable':
                    self.no_reciclable_count += 1
                elif class_name == 'Organico':
                    self.organico_count += 1

        # Actualizar las etiquetas de los contadores
        self.reciclable_count_label.text = str(self.reciclable_count)
        self.no_reciclable_count_label.text = str(self.no_reciclable_count)
        self.organico_count_label.text = str(self.organico_count)

        # Mostrar el frame en la interfaz
        frame = cv2.flip(frame, 0)
        buf = frame.tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.background.texture = texture

    def close_camera(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

        Clock.unschedule(self.update_frame)
        self.background.texture = None
        self.background.source = 'images/background.jpg'
        self.background.reload()

        # Restablecer contadores y etiquetas
        self.reciclable_count = 0
        self.no_reciclable_count = 0
        self.organico_count = 0
        self.reciclable_count_label.text = '0'
        self.no_reciclable_count_label.text = '0'
        self.organico_count_label.text = '0'

    def detect_image(self):
        """Abrir un selector de archivos para elegir una imagen y realizar la detección."""
        # Cerrar la cámara si está abierta
        self.close_camera()

        # Crear el contenido del popup
        content = BoxLayout(orientation='vertical')
        filechooser = FileChooserListView(filters=['*.png', '*.jpg', '*.jpeg'], path='.')
        content.add_widget(filechooser)

        # Crear los botones de selección y cancelación
        button_box = BoxLayout(size_hint_y=None, height=30)
        select_button = Button(text='Seleccionar', size_hint_x=0.5)
        cancel_button = Button(text='Cancelar', size_hint_x=0.5)
        button_box.add_widget(select_button)
        button_box.add_widget(cancel_button)
        content.add_widget(button_box)

        # Crear el popup
        self.popup = Popup(title='Seleccionar Imagen', content=content, size_hint=(0.9, 0.9))
        self.popup.open()

        # Vincular los botones
        select_button.bind(on_press=lambda x: self.load_image(filechooser.selection))
        cancel_button.bind(on_press=self.popup.dismiss)

    def load_image(self, selection):
        """Cargar la imagen seleccionada y realizar la detección."""
        if selection:
            image_path = selection[0]
            self.popup.dismiss()

            # Cargar la imagen
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error: No se pudo cargar la imagen {image_path}")
                return

            # Restablecer los contadores
            self.reciclable_count = 0
            self.no_reciclable_count = 0
            self.organico_count = 0

            # Realizar la predicción
            results = self.model.predict(source=frame, save=False, conf=0.25)
            class_names = self.model.model.names

            # Colores específicos para cada clase
            color_dict = {
                'Reciclable': (255, 0, 0),      # Azul en BGR
                'No_Reciclable': (0, 0, 255),   # Rojo en BGR
                'Organico': (0, 255, 0)         # Verde en BGR
            }

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    class_name = class_names[cls]
                    color = color_dict.get(class_name, (255, 255, 255))  # Blanco por defecto
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Dibujar rectángulo y etiqueta
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f'{class_name}: {conf:.2f}'
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Actualizar contadores
                    if class_name == 'Reciclable':
                        self.reciclable_count += 1
                    elif class_name == 'No_Reciclable':
                        self.no_reciclable_count += 1
                    elif class_name == 'Organico':
                        self.organico_count += 1

            # Actualizar las etiquetas de los contadores
            self.reciclable_count_label.text = str(self.reciclable_count)
            self.no_reciclable_count_label.text = str(self.no_reciclable_count)
            self.organico_count_label.text = str(self.organico_count)

            # Mostrar la imagen en la interfaz
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 0)
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.background.texture = texture
            self.background.canvas.ask_update()
        else:
            print("No se seleccionó ninguna imagen")
            self.popup.dismiss()

    def on_stop(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

# Ejecutar la aplicación
if __name__ == '__main__':
    MainApp().run()
