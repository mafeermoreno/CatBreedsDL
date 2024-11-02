import os
import shutil
import random

def split_dataset(source_folder, train_folder, val_folder, test_folder, split_ratio=(0.6, 0.2, 0.2)):
    """
    Divide un dataset dado en tres subconjuntos: train, validation y test.
    
    Parámetros:
        source_folder: str, ruta de la carpeta que contiene el dataset original.
        train_folder: str, ruta de la carpeta donde se almacenará el subconjunto de entrenamiento.
        val_folder: str, ruta de la carpeta donde se almacenará el subconjunto de validación.
        test_folder: str, ruta de la carpeta donde se almacenará el subconjunto de prueba.
        split_ratio: tuple, proporción en la que se dividirá el dataset original en los subconjuntos de entrenamiento, validación y prueba.
    """
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
main_folder = 'cats' # Carpeta que contiene las carpetas de imágenes de los tipos de gatos
train_cats_folder = 'data/train'
val_cats_folder = 'data/validation'
test_cats_folder = 'data/test'

# Dividir el dataset en train, validation y test
split_dataset(main_folder, train_cats_folder, val_cats_folder, test_cats_folder)
