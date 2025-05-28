import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def load_binary_mask(path, target_size):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"No se pudo cargar la máscara: {path}")
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)

def count_objects(mask):
    """Cuenta objetos como componentes conectados en la máscara binaria"""
    num_labels, _, _, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return num_labels - 1  # se descarta fondo

def compute_GAME(pred_mask, gt_mask, L):
    h, w = pred_mask.shape
    game = 0
    n_cells = 2 ** L
    cell_h, cell_w = h // n_cells, w // n_cells

    for i in range(n_cells):
        for j in range(n_cells):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w

            pred_crop = pred_mask[y1:y2, x1:x2]
            gt_crop = gt_mask[y1:y2, x1:x2]

            pred_count = count_objects(pred_crop)
            gt_count = count_objects(gt_crop)

            game += abs(pred_count - gt_count)

    return game

def evaluate_game(img_path, gt_mask_path, pred_mask_path):
    pred_mask_raw = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    if pred_mask_raw is None:
        raise ValueError(f"No se pudo cargar la máscara predicha: {pred_mask_path}")
    pred_mask = (pred_mask_raw > 127).astype(np.uint8)
    height, width = pred_mask.shape

    gt_mask = load_binary_mask(gt_mask_path, (width, height))

    total_objects = count_objects(gt_mask)
    if total_objects == 0:
        raise ValueError(f"La máscara GT tiene 0 objetos para la imagen: {img_path}")

    game_vals = {}
    for L in range(4):
        game_val = compute_GAME(pred_mask, gt_mask, L)
        game_vals[f"GAME(L={L})"] = game_val
        game_vals[f"GAME_norm(L={L})"] = game_val / total_objects

    result = {
        "image": os.path.basename(img_path),
        "total_objects_GT": total_objects,
    }
    result.update(game_vals)
    return result

def main():
    # 👉 Lista de tripletas de archivos (esto podés modificarlo o leerlo desde CSV)
    files = [
        {
            "img": r"DICE_Masks/masks/657d1748-839a-4e18-ab64-a6cca9ec2e26.jpg",
            "gt":  r"DICE_Masks/masks/657d1748-839a-4e18-ab64-a6cca9ec2e26.png",
            "pred": r"predicted_masks/256x455_657d1748-839a-4e18-ab64-a6cca9ec2e26_mask.png"
        },
        # Agregar más aquí...
    ]

    results = []
    for f in tqdm(files):
        try:
            result = evaluate_game(f["img"], f["gt"], f["pred"])
            results.append(result)
        except Exception as e:
            print(f"Error procesando {f['img']}: {e}")

    df = pd.DataFrame(results)
    print(df)
    # Opcional: guardar CSV
    df.to_csv("game_results_normalized.csv", index=False)

if __name__ == "__main__":
    main()


'''
Cambios clave:

En lugar de contar píxeles, ahora se cuentan los objetos detectados como componentes conectados.

Uso de cv2.connectedComponentsWithStats para segmentar y contar objetos individualmente dentro de cada subcelda.

Esto es mucho más preciso cuando tus máscaras segmentan objetos individuales (por ejemplo células, personas, coches), y cada objeto puede ocupar múltiples píxeles.

Al restar 1, descartamos la etiqueta de fondo (0).

¿Por qué es importante esta diferencia?
El primer método (np.sum) asume que la suma de píxeles es proporcional al número de objetos, válido si objetos son puntos o no están agrupados.

El segundo método es más fiel al concepto original de GAME, que evalúa cuántos objetos están en cada celda.

Si tu segmentación es de objetos con tamaño variable y bien definidos, el segundo método es más adecuado.

GAME normalizado:
Se agregó el cálculo de total_objects en la máscara GT.

Se agregó validación para evitar dividir por cero si no hay objetos GT.

Para cada nivel L, se calcula el GAME normalizado (GAME_norm(L=X)).

El resultado incluye el conteo total de objetos en GT.
'''