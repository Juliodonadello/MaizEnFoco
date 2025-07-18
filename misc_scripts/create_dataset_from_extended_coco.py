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

# === Parámetros de tamaño fijo ===
TARGET_WIDTH = 455
TARGET_HEIGHT = 405

# Paths
DATA_DIR = '../DeepAgro/datasets/final'
ANNOTATION_PATH = os.path.join(DATA_DIR, 'annotations.json')
OUTPUT_DIR = '../DeepAgro/Segmentation'

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

# Extensiones posibles
extensions = ['.jpg', '.jpeg', '.png']

def buscar_archivo_sin_extension(base_path):
    for ext in extensions:
        full_path = base_path + ext
        if os.path.exists(full_path):
            return full_path
    return None

# Procesar imágenes
data = []
for image in tqdm(coco['images'], desc="Procesando imágenes"):

    img_id = image['id']
    original_filename = image['file_name']
    name, _ = os.path.splitext(original_filename)
    src_base = os.path.join(DATA_DIR, name)

    src_path = buscar_archivo_sin_extension(src_base)

    if not src_path:
        print(f"Imagen no encontrada: {src_base}.[jpg/png/jpeg]")
        continue

    anns = annotations_by_image.get(img_id, [])
    
    filename = name + '.jpg'  # Forzar .jpg como nombre de destino

    if anns:
        image_dst = os.path.join(OUTPUT_DIR, 'images/valid', filename)
        mask_dst = os.path.join(OUTPUT_DIR, 'masks/valid', filename)
    else:
        image_dst = os.path.join(OUTPUT_DIR, 'images/empty', filename)
        mask_dst = os.path.join(OUTPUT_DIR, 'masks/empty', filename)

    # Abrir imagen original y hacer resize
    with Image.open(src_path) as im:
        im_resized = im.convert("RGB").resize((TARGET_WIDTH, TARGET_HEIGHT), Image.BILINEAR)
        im_resized.save(image_dst, "JPEG")

    # Crear máscara vacía con tamaño original
    width, height = image['width'], image['height']
    original_mask = np.zeros((height, width), dtype=np.uint8)

    if anns:
        for ann in anns:
            for seg in ann.get('segmentation', []):
                points = np.array(seg).reshape(-1, 2)
                rr, cc = polygon(points[:, 1], points[:, 0], shape=original_mask.shape)
                original_mask[rr, cc] = 1

    # Redimensionar la máscara
    mask_pil = Image.fromarray(original_mask * 255).resize((TARGET_WIDTH, TARGET_HEIGHT), Image.NEAREST)
    mask_np = np.array(mask_pil) // 255  # Convertir de nuevo a binario (0 o 1)

    # Guardar máscara como .jpg y .png
    Image.fromarray(mask_np * 255).save(mask_dst, "JPEG")
    mask_dst_png = mask_dst.rsplit('.', 1)[0] + '.png'
    Image.fromarray(mask_np * 255).save(mask_dst_png, "PNG")

    # Guardar metadata
    data.append({
        'ID': f'valid/{name}',
        'labels': f'valid/{name}',
        'has_annotation': int(bool(anns))
    })

# Crear y guardar CSVs
df = pd.DataFrame(data)
train = df.sample(frac=0.8, random_state=42)
remaining = df.drop(train.index)
val = remaining.sample(frac=0.5, random_state=42)
test = remaining.drop(val.index)

df.to_csv(os.path.join(OUTPUT_DIR, 'segmentation.csv'), index=False)
train.to_csv(os.path.join(OUTPUT_DIR, 'train.csv'), index=False)
val.to_csv(os.path.join(OUTPUT_DIR, 'val.csv'), index=False)
test.to_csv(os.path.join(OUTPUT_DIR, 'test.csv'), index=False)

print("Procesamiento de las anotaciones completo.")
