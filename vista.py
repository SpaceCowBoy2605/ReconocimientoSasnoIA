import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Cargar el modelo
model = load_model('C:/Users/felix/OneDrive/Documentos/reconocimiento/modelo/modelo_completo.keras')

# Lista de nombres de ingredientes
class_names = ["manzana", "naranja","platano"]


# Función para procesar la imagen antes de la predicción
def preprocess_image(img, target_size=(200, 200)):
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=0)


class FoodRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Alimentos")

        # Configurar la interfaz
        self.setup_ui()

        # Iniciar la cámara
        self.cap = cv2.VideoCapture(0)
        self.is_camera_active = True
        self.update_camera()

    def setup_ui(self):
        # Frame principal
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Canvas para mostrar la cámara
        self.camera_canvas = tk.Canvas(self.main_frame, width=640, height=480)
        self.camera_canvas.grid(row=0, column=0, rowspan=4, padx=5, pady=5)

        # Botón para capturar
        self.capture_btn = ttk.Button(self.main_frame, text="Capturar (s)", command=self.capture_image)
        self.capture_btn.grid(row=0, column=1, sticky=tk.W, pady=5)

        # Botón para salir
        self.quit_btn = ttk.Button(self.main_frame, text="Salir (q)", command=self.quit_app)
        self.quit_btn.grid(row=1, column=1, sticky=tk.W, pady=5)

        # Área para mostrar resultados
        self.result_frame = ttk.LabelFrame(self.main_frame, text="Resultados", padding="10")
        self.result_frame.grid(row=2, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.top_label = ttk.Label(self.result_frame, text="Lista:", font=('Arial', 10, 'bold'))
        self.top_label.pack(anchor=tk.W)

        self.result_text = tk.Text(self.result_frame, height=10, width=40, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        # Barra de desplazamiento para el texto
        scrollbar = ttk.Scrollbar(self.result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_text.config(yscrollcommand=scrollbar.set)

        # Configurar eventos de teclado
        self.root.bind('<KeyPress-s>', lambda e: self.capture_image())
        self.root.bind('<KeyPress-q>', lambda e: self.quit_app())

    def update_camera(self):
        if self.is_camera_active:
            ret, frame = self.cap.read()
            if ret:
                # Convertir de BGR a RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Redimensionar manteniendo aspect ratio
                h, w = frame.shape[:2]
                ratio = w / h
                new_h = 480
                new_w = int(new_h * ratio)
                frame_resized = cv2.resize(frame_rgb, (new_w, new_h))

                # Convertir a ImageTk
                img = Image.fromarray(frame_resized)
                imgtk = ImageTk.PhotoImage(image=img)

                # Actualizar canvas
                self.camera_canvas.imgtk = imgtk
                self.camera_canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

            # Llamar de nuevo después de 15 ms
            self.root.after(15, self.update_camera)

    def capture_image(self):
        if self.is_camera_active:
            ret, frame = self.cap.read()
            if ret:
                input_image = preprocess_image(frame)
                predictions = model.predict(input_image)[0]

                sorted_indices = np.argsort(predictions)[::-1]

                self.result_text.delete(1.0, tk.END)
                self.result_text.insert(tk.END, "Lista:\n\n")

                # Mostrar el top N según número de clases disponibles
                top_n = len(class_names)

                for i in range(top_n):
                    idx = sorted_indices[i]
                    prob = predictions[idx] * 100
                    self.result_text.insert(tk.END, f"{class_names[idx]}: {prob:.2f}%\n")

                # Mostrar otras probabilidades > 1% (si hubiera más clases en el futuro)
                self.result_text.insert(tk.END, "\nOtras posibilidades:\n")
                for i in range(top_n, len(sorted_indices)):
                    idx = sorted_indices[i]
                    prob = predictions[idx] * 100
                    if prob > 1.0:
                        self.result_text.insert(tk.END, f"{class_names[idx]}: {prob:.2f}%\n")

    def quit_app(self):
        self.is_camera_active = False
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FoodRecognitionApp(root)
    root.mainloop()