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
ANNOTATION_PATH = os.path.join(DATA_DIR, 'annotations.json')
OUTPUT_DIR = 'DeepAgro/Segmentation'

# Crear carpetas de salida
for subfolder in ['images/valid', 'images/empty', 'masks/valid', 'masks/empty']:
    os.makedirs(os.path.join(OUTPUT_DIR, subfolder), exist_ok=True)

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

    # Destinos
    if anns:
        image_dst = os.path.join(OUTPUT_DIR, 'images/valid', filename)
        mask_dst = os.path.join(OUTPUT_DIR, 'masks/valid', filename)
    else:
        image_dst = os.path.join(OUTPUT_DIR, 'images/empty', filename)
        mask_dst = os.path.join(OUTPUT_DIR, 'masks/empty', filename)

    # Copiar imagen
    shutil.copy(src_path, image_dst)

    # Crear máscara vacía (con NumPy)
    mask = np.zeros((height, width), dtype=np.uint8)

    if anns:  # Solo dibujar si hay anotaciones
        for ann in anns:
            for seg in ann.get('segmentation', []):
                points = np.array(seg).reshape(-1, 2)
                rr, cc = polygon(points[:, 1], points[:, 0], shape=mask.shape)
                mask[rr, cc] = 1  # Marca como 1 en la máscara

    # Guardar máscara
    Image.fromarray(mask * 255).save(mask_dst)
    Image.fromarray(mask * 255).save(mask_dst.replace(".jpg",".png"))  # Guardamos la máscara como imagen de 0-255

    # Guardar metadata
    data.append({
        'ID': f'valid/{filename}'.strip('.jpg'),
        'labels': f'valid/{filename}'.strip('.jpg'),
        'has_annotation': int(bool(anns))
    })

# Crear DataFrame
df = pd.DataFrame(data)

# Dividir en train/val/test
train = df.sample(frac=0.8, random_state=42)
remaining = df.drop(train.index)
val = remaining.sample(frac=0.5, random_state=42)
test = remaining.drop(val.index)

# Guardar CSVs
df.to_csv(os.path.join(OUTPUT_DIR, 'segmentation.csv'), index=False)
train.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
val.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
test.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

print("Procesamiento de las anotaciones completo.")
