import os
import tensorflow as tf
from tensorflow.keras import backend as k
from tensorflow.keras.callbacks import TensorBoard, Callback
import matplotlib.pyplot as plt
import numpy as np

# 1. Callback para graficar curvas
class PlotMetrics(Callback):
    def on_train_end(self, logs=None):
        plt.figure(figsize=(12, 5))
        
        # Gráfico de Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.model.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.model.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy durante Entrenamiento')
        plt.ylabel('Accuracy')
        plt.xlabel('Época')
        plt.legend()
        
        # Gráfico de Loss
        plt.subplot(1, 2, 2)
        plt.plot(self.model.history.history['loss'], label='Train Loss')
        plt.plot(self.model.history.history['val_loss'], label='Validation Loss')
        plt.title('Loss durante Entrenamiento')
        plt.ylabel('Loss')
        plt.xlabel('Época')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
# Limpiamos cualquier sesión previa
k.clear_session()

# Cargamos los sets de datos
dataEntrenamiento = 'C:/Users/felix/OneDrive/Documentos/reconocimiento/entrenamiento/fit'
dataValidacion = 'C:/Users/felix/OneDrive/Documentos/reconocimiento/VALIDACION/validation'

# Parámetros del modelo
epocas = 10
longitud, altura = 200, 200
tamanoLote = 32
filtrosConv1 = 32
filtrosConv2 = 64
tamano_filtro1 = (3, 3)
tamano_filtro2 = (2, 2)
tamano_pool = (2, 2)
clases = 3
lr = 0.0004

# Preparamos las imágenes para entrenamiento y validación
entrenemientoDataGen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)


validacionDataGen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255
)

imagenesEntrenemiento = entrenemientoDataGen.flow_from_directory(
    dataEntrenamiento,
    target_size=(altura, longitud),
    batch_size=tamanoLote,
    class_mode='categorical',
    shuffle=True
)

imagenesValidacion = validacionDataGen.flow_from_directory(
    dataValidacion,
    target_size=(altura, longitud),
    batch_size=tamanoLote,
    class_mode='categorical',
    shuffle=False
)

print("Numero de imagenes en entrenamiento:", imagenesEntrenemiento.samples)
print("Numero de imagenes en validacion:", imagenesValidacion.samples)

# Definimos el modelo CNN
modeloCNN = tf.keras.Sequential([
    tf.keras.Input(shape=(longitud, altura, 3)),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(clases, activation='softmax')
])


# Compilamos el modelo
modeloCNN.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    metrics=['accuracy']
)

# Entrenamos el modelo (sin steps_per_epoch ni validation_steps)
tensor = TensorBoard(log_dir='logs/cnn')
plot_metrics = PlotMetrics()
history = modeloCNN.fit(
    imagenesEntrenemiento,
    epochs=epocas,
    validation_data=imagenesValidacion,
    callbacks=[tensor, plot_metrics]
)

# Guardamos el modelo
archivo = './modelo/'
if not os.path.exists(archivo):
    os.makedirs(archivo, exist_ok=True)

# Guardado en formato moderno
modeloCNN.save(os.path.join(archivo, 'modelo_completo.keras'))

# Opcional: guardar solo los pesos si lo necesitas
modeloCNN.save_weights(os.path.join(archivo, 'pesos_modelo.weights.h5'))
