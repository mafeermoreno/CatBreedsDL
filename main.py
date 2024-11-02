import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
import numpy as np  
import os
import shutil
import random

#Existen 3 carpetas con diferentes tipos de cruzas de gatos: Bengal, Bombay y Calico.

'''
Función split_dataset: Divide un dataset dado en tres subconjuntos: train, validation y test.
    Parámetros: 
        source_folder: str, ruta de la carpeta que contiene el dataset original.
        train_folder: str, ruta de la carpeta donde se almacenará el subconjunto de entrenamiento.
        val_folder: str, ruta de la carpeta donde se almacenará el subconjunto de validación.
        test_folder: str, ruta de la carpeta donde se almacenará el subconjunto de prueba.
        split_ratio: tuple, proporción en la que se dividirá el dataset original en los subconjuntos de entrenamiento, validación y prueba.
'''
def split_dataset(source_folder, train_folder, val_folder, test_folder, split_ratio=(0.6, 0.2, 0.2)):
    # Crear las carpetas de train, validation y test si no existen
    for folder in [train_folder, val_folder, test_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # Recorrer las carpetas de las razas de gatos
    for breed in os.listdir(source_folder):
        breeds_path = os.path.join(source_folder, breed)
        cats_images = os.listdir(breeds_path)
        random.shuffle(cats_images)
        
        # Calcular los tamaños de los splits
        train_size = int(len(cats_images) * split_ratio[0])
        val_size = int(len(cats_images) * split_ratio[1])
        
        # Crear los subconjuntos de acuerdo a los tamaños calculados
        train_cats = cats_images[:train_size]
        val_cats = cats_images[train_size:train_size+val_size]
        test_cats = cats_images[train_size+val_size:]
        
        # Crear carpetas para cada subconjunto
        for cat in train_cats:
            train_direction = os.path.join(train_folder, breed)
            if not os.path.exists(train_direction):
                os.makedirs(train_direction)
            shutil.copy(os.path.join(breeds_path, cat), os.path.join(train_direction, cat))
        
        for cat in val_cats:
            val_direction = os.path.join(val_folder, breed)
            if not os.path.exists(val_direction):
                os.makedirs(val_direction)
            shutil.copy(os.path.join(breeds_path, cat), os.path.join(val_direction, cat))
        
        for cat in test_cats:
            test_direction = os.path.join(test_folder, breed)
            if not os.path.exists(test_direction):
                os.makedirs(test_direction)
            shutil.copy(os.path.join(breeds_path, cat), os.path.join(test_direction, cat))
        
# Definir las rutas de las carpetas
main_folder = 'cats'
train_cats_folder = 'data/train'
val_cats_folder = 'data/validation'
test_cats_folder = 'data/test'

# Dividir el dataset en train, validation y test
split_dataset(main_folder, train_cats_folder, val_cats_folder, test_cats_folder)

# Aumentaciones de datos en train
train_aumentation = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Normalizar los datos de validación
val_aumentation = ImageDataGenerator(rescale=1./255)

# Generar los datos
train_generator = train_aumentation.flow_from_directory(
    train_cats_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

validation_generator = val_aumentation.flow_from_directory(
    val_cats_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

test_generator = val_aumentation.flow_from_directory(
    test_cats_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

# Cargar la base convolucional VGG16
convolutional_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

convolutional_base.trainable = False

# Definir el modelo de CNN
model = models.Sequential()
model.add(convolutional_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(3, activation='softmax')) # El 3 es por las 3 clases que tengo: Bengal, Bombay y Calico

model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=2e-5), 
    metrics=['accuracy']
)

# Guardar el mejor modelo
checkpoint_callback = ModelCheckpoint(
    filepath='best_cat_breed_classifier.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Calcular steps_per_epoch basado en el número de imágenes y batch_size
train_steps = len(train_generator.filenames) // train_generator.batch_size
val_steps = len(validation_generator.filenames) // validation_generator.batch_size
test_steps = len(test_generator.filenames) // test_generator.batch_size

# Entrenamiento del modelo sin incluir las métricas de prueba en cada época
history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=[checkpoint_callback]
)

# Evaluar en el conjunto de prueba solo después del entrenamiento
test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

# Graficar las métricas de entrenamiento y validación, y añadir los resultados de test al final
epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(12, 5))

# Gráfica de precisión
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], 'bo-', label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'r^-', label='Validation Accuracy')
plt.axhline(test_acc, color='g', linestyle='--', label='Test Accuracy')  # Línea constante para el test accuracy
plt.title('Training, Validation, and Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Gráfica de pérdida
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], 'bo-', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], 'r^-', label='Validation Loss')
plt.axhline(test_loss, color='g', linestyle='--', label='Test Loss')  # Línea constante para el test loss
plt.title('Training, Validation, and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
