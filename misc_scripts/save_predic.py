import os
import sys
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from haven import haven_utils as hu
from src import models
from src import datasets
from src.datasets import jcu_fish
import exp_configs
import glob

# Configuraciones
#exp_name = 'a106acdd252ec8c131d81b70a2014ffc' # testeado con lotes distintos los usadon en val y train
exp_name = '8f7844d98e8d29e93b3831b9576a7db4' #'with_affinity&lcfcn_loss'       # Nombre del experimento
base_dir = os.getcwd()                                                          # Ruta base (usualmente donde se ejecuta el script)
exp_dict_path = os.path.join(base_dir, 'results', exp_name, 'exp_dict.json')    # Ruta completa al exp_dict.json
model_path = os.path.join(base_dir, 'results', exp_name, 'model_best.pth')      # Ruta completa al modelo

# Lista de imágenes a predecir

validation_images = [
    #r'DeepAgro\Segmentation\images\valid\657d1748-839a-4e18-ab64-a6cca9ec2e26.jpg',
    #r'DeepAgro\Segmentation\images\valid\db836de8-2a4f-4e2d-81c5-50c4b6c7e874.jpg'
    #r'DeepAgro\Segmentation\images\valid\d4ba4743b_e2b4a3ce-e6da-4f9a-95e7-14cf8d61386a.jpg',
    #r'DeepAgro\Segmentation\images\valid\d4ba4743b_d527010a-a5a2-47b1-b88a-bdec19c44d36.jpg'   #con sombra
    #r'images_to_predict\522143ac-6097-476f-aaef-e14ae8f91dcb.jpg',
    #r'images_to_predict\892486fe-fc81-499e-8def-6d578cd301a3.jpg',
    r'images_to_predict\f798fa1a-76e7-438b-b488-bc8c1e649300.jpg'
]
'''
images_folder = 'images_to_predict'
validation_images = []
image_extensions = ['*.jpg', '*.jpeg', '*.png']
for extension in image_extensions:
    validation_images.extend(glob.glob(os.path.join(images_folder, extension)))
    validation_images.extend(glob.glob(os.path.join(images_folder, extension.upper())))
'''

for image_path in validation_images:
    print(repr(image_path))  # <- muestra caracteres especiales como \v

# Cargar exp_dict
#exp_dict_path = os.path.join(savedir, "exp_dict.json")
exp_dict = hu.load_json(exp_dict_path)
# Dataset (solo para transformar si es necesario)

dummy_dataset = jcu_fish.JcuFish(
    split="val",
    datadir='DeepAgro',
    exp_dict=exp_dict
)
'''
dummy_dataset = datasets.get_dataset(
    dataset_dict=exp_dict["dataset"],
    split="val",
    datadir='JCU_Fish',
    exp_dict=exp_dict,
    dataset_size=exp_dict['dataset_size']
)
'''
print(f"Dataset: {dummy_dataset.__class__.__name__}")

transform = dummy_dataset.img_transform
'''
custom_transform = T.Compose([
    T.Resize((512, 512)), # Redimensionar a 512x512
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # si es necesario
])
transform = custom_transform
'''
print(f"Transform: {transform}")

# Cargar modelo
model = models.get_model(model_dict=exp_dict['model'], exp_dict=exp_dict, train_set=dummy_dataset)
model.load_state_dict(hu.torch_load(model_path))
model = model.cuda()
model.eval()
# Crear directorio para máscaras predichas
output_dir = os.path.join(base_dir, 'predicted_masks')
os.makedirs(output_dir, exist_ok=True)
print("len images:", len(validation_images))
for image_path in validation_images:
    with torch.no_grad():
        print(f"\nProcesando {image_path}")
        image = Image.open(image_path).convert('RGB')

        # Mostrar resolución original
        print(f"Resolución original: {image.size}")  # (width, height)

        # Redimensionar a 256x256
        #image_resized = image.resize((512, 512))
        #print(f"Resolución redimensionada: {image_resized.size}")

        # Aplicar transformaciones
        image_tensor = transform(image).unsqueeze(0).cuda()  # [1, C, H, W]
        print(f"Tamaño del tensor: {image_tensor.shape}")  # (1, 3, 256, 256)

        # Construir el batch para predict_on_batch
        batch = {
            'images': image_tensor,
            'meta': [{'shape': image_tensor.shape[-2:]}]  # ya redimensionado
        }
        resized_res = f"{image_tensor.shape[2]}x{image_tensor.shape[3]}"
        print("Ejecutando inferencia...")
        pred = model.predict_on_batch(batch)  # model es tu instancia de SemSeg

        # Guardar la máscara
        mask_np = pred[0] * 255
        mask_img = Image.fromarray(mask_np.astype(np.uint8))
        out_path = os.path.join(output_dir, f"{resized_res}_{os.path.basename(image_path).replace(".jpg", "_mask.png")}")
        mask_img.save(out_path)
        print(f"Guardada: {out_path}")

    torch.cuda.empty_cache()
