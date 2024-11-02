import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

# Normalización de los datos de validación y prueba
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Definir el tamaño del lote
batch_size = 32

# Generadores de datos
train_generator = train_datagen.flow_from_directory(
    train_cats_folder,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_test_datagen.flow_from_directory(
    val_cats_folder,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = val_test_datagen.flow_from_directory(
    test_cats_folder,
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Importante para que las predicciones se alineen con los datos de prueba
)

# Cargar la base convolucional VGG16 preentrenada
conv_base = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(150, 150, 3)
)

# Descongelar las últimas 3 capas para fine-tuning
conv_base.trainable = True
for layer in conv_base.layers[:-3]:
    layer.trainable = False

# Definir el modelo de red neuronal con una capa convolucional adicional
model = models.Sequential()
model.add(conv_base)
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Nueva capa convolucional
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))  # 3 clases: Bengal, Bombay, Calico

# Compilar el modelo con optimizador Adam y tasa de aprendizaje reducida
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.Adam(learning_rate=1e-5),
    metrics=['accuracy']
)

# Guardar el mejor modelo durante el entrenamiento
checkpoint_callback = ModelCheckpoint(
    filepath='best_cat_breed_classifier_improvement.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Implementar EarlyStopping para evitar el sobreajuste
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Entrenamiento del modelo
history = model.fit(
    train_generator,
    steps_per_epoch=75,
    epochs=20,
    validation_data=val_generator,
    validation_steps=25,
    callbacks=[checkpoint_callback, early_stopping_callback]
)

# Evaluar en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_generator)
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