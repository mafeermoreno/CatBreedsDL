import os
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Definir las rutas de las carpetas
train_cats_folder = 'data/train'
val_cats_folder = 'data/validation'
test_cats_folder = 'data/test'

# Aumentación y normalización de los datos de entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Normalización de los datos de validación y prueba
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_cats_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_cats_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_cats_folder,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical'
)

# Cargar la base convolucional VGG16 preentrenada
conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

# Congelar la base convolucional para no entrenar sus pesos
conv_base.trainable = False

# Definir el modelo de red neuronal
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))  # 3 clases: Bengal, Bombay, Calico

# Compilar el modelo
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.RMSprop(learning_rate=2e-5),
    metrics=['accuracy']
)

# Guardar el mejor modelo durante el entrenamiento
checkpoint_callback = ModelCheckpoint(
    filepath='best_cat_breed_classifier.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=75,          
    epochs=20,
    validation_data=val_generator,
    validation_steps=25,    
    callbacks=[checkpoint_callback]
)

# Evaluar en el conjunto de prueba 
test_loss, test_acc = model.evaluate(test_generator, steps=25)  
print(f"\nFinal Test Loss: {test_loss:.4f} - Final Test Accuracy: {test_acc:.4f}")

# Graficar las métricas de entrenamiento y validación
epochs = range(1, len(history.history['accuracy']) + 1)

plt.figure(figsize=(12, 5))

# Gráfica de precisión
plt.subplot(1, 2, 1)
plt.plot(epochs, history.history['accuracy'], color='orchid', marker='o', linestyle='-', label='Training Accuracy')
plt.plot(epochs, history.history['val_accuracy'], color='hotpink', marker='^', linestyle='-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Gráfica de pérdida
plt.subplot(1, 2, 2)
plt.plot(epochs, history.history['loss'], color='orchid', marker='o', linestyle='-', label='Training Loss')
plt.plot(epochs, history.history['val_loss'], color='hotpink', marker='^', linestyle='-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Obtener predicciones en el conjunto de prueba
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Obtener etiquetas verdaderas del conjunto de prueba
y_true = test_generator.classes

# Calcular y mostrar la matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred_classes)
class_names = list(test_generator.class_indices.keys())  # Nombres de las clases
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()