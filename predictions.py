import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Definir la ruta del modelo de entrenamiento mejorado
improved_path = 'best_cat_breed_classifier_improvement.keras'

# Cargar el modelo mejorado previamente entrenado
model = load_model(improved_path)
print("El modelo se ha cargado correctamente.")

# Definir las clases de acuerdo a como fueron entrenadas
class_names = ['Bengal', 'Bombay', 'Calico']

# Función para cargar y preprocesar una imagen individual
def load_and_prepare_image(img_name, target_size=(150, 150)):
    img_path = os.path.join(os.getcwd(), img_name)
    
    if not os.path.exists(img_path):
        print(f"La imagen '{img_name}' no se encuentra en el directorio actual.")
        return None
    
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión para el batch
    return img_array

# Función para hacer la predicción en una imagen específica
def predict_single_image(img_name):
    img_array = load_and_prepare_image(img_name)
    if img_array is None:
        return

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]
    class_name = class_names[predicted_class]

    # Mostrar el resultado
    print(f"\nPredicción: {class_name} ({confidence:.2f} de confianza)")
    img = image.load_img(img_name)
    plt.imshow(img)
    plt.title(f"Predicción: {class_name} ({confidence:.2f} de confianza)")
    plt.axis('off')
    plt.show()

# Solicitar al usuario el nombre de la imagen
if __name__ == '__main__':
    img_name = input("Especifica el nombre de la imagen (incluyendo la extensión): ")
    predict_single_image(img_name)
