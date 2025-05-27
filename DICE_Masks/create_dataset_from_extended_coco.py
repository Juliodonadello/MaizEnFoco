import os
import json
import shutil
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.draw import polygon
from tqdm import tqdm

# Paths
DATA_DIR = 'DeepAgro/datasets/test_fffb7fa69'
ANNOTATION_PATH = os.path.join('DICE_Masks', 'coco_full_masks.json')
OUTPUT_DIR = 'DICE_Masks/masks/'

# Crear directorio de salida si no existe
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Cargar anotaciones formato COCO
with open(ANNOTATION_PATH, 'r') as f:
    coco = json.load(f)

# Mapeo de IDs a nombres de archivo
image_id_to_filename = {img['id']: img['file_name'] for img in coco['images']}

# Agrupar anotaciones por imagen
annotations_by_image = {}
for ann in coco['annotations']:
    image_id = ann['image_id']
    annotations_by_image.setdefault(image_id, []).append(ann)

# Procesar todas las imágenes
data = []
for image in tqdm(coco['images'], desc="Procesando imágenes"):

    img_id = image['id']
    filename = image['file_name']
    width, height = image['width'], image['height']
    src_path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(src_path):
        print(f"Imagen no encontrada: {src_path}")
        continue

    anns = annotations_by_image.get(img_id, [])

    # Copiar imagen
    shutil.copy(src_path, OUTPUT_DIR)

    # Crear máscara vacía (con NumPy)
    mask = np.zeros((height, width), dtype=np.uint8)

    if anns:  # Solo dibujar si hay anotaciones
        for ann in anns:
            for seg in ann.get('segmentation', []):
                points = np.array(seg).reshape(-1, 2)
                rr, cc = polygon(points[:, 1], points[:, 0], shape=mask.shape)
                mask[rr, cc] = 1  # Marca como 1 en la máscara

    # Guardar máscara
    #Image.fromarray(mask * 255).save(f'{OUTPUT_DIR}{filename}')
    Image.fromarray(mask * 255).save(f'{OUTPUT_DIR}{filename}'.replace(".jpg",".png"))  # Guardamos la máscara como imagen de 0-255

print("Procesamiento de las anotaciones completo.")
