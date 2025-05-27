import cv2
import numpy as np
import matplotlib.pyplot as plt

def dice_coefficient(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    total = mask1.sum() + mask2.sum()
    if total == 0:
        return 1.0
    return 2. * intersection / total

def iou_score(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 1.0
    return intersection / union

def mean_iou(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    ious = []
    for cls in [0, 1]:
        pred_cls = (mask2 == cls)
        gt_cls = (mask1 == cls)
        intersection = np.logical_and(pred_cls, gt_cls).sum()
        union = np.logical_or(pred_cls, gt_cls).sum()
        if union == 0:
            ious.append(1.0)  # Si no hay píxeles de esa clase en ninguna de las dos
        else:
            ious.append(intersection / union)

    return np.mean(ious)

def mean_absolute_error(mask1, mask2):
    return np.mean(np.abs(mask1.astype(np.float32) - mask2.astype(np.float32)))

def game_score(mask_gt, mask_pred, level):
    h, w = mask_gt.shape
    num_cells = 2 ** level
    cell_h, cell_w = h // num_cells, w // num_cells
    error = 0
    for i in range(num_cells):
        for j in range(num_cells):
            y0, y1 = i * cell_h, (i + 1) * cell_h
            x0, x1 = j * cell_w, (j + 1) * cell_w
            count_gt = mask_gt[y0:y1, x0:x1].sum()
            count_pred = mask_pred[y0:y1, x0:x1].sum()
            error += abs(int(count_gt) - int(count_pred))
    return error

def load_binary_mask(path, target_size):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"No se pudo cargar la máscara: {path}")
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)

def comparar_y_mostrar(img_path, gt_mask_path, pred_mask_path):
    # Leer máscara predicha (referencia de tamaño)
    pred_mask_raw = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
    if pred_mask_raw is None:
        raise ValueError(f"No se pudo cargar la máscara predicha: {pred_mask_path}")
    pred_mask = (pred_mask_raw > 127).astype(np.uint8)
    height, width = pred_mask.shape

    # Leer imagen original y máscara manual, redimensionadas al tamaño de la predicción
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gt_mask = load_binary_mask(gt_mask_path, (width, height))

    # Calcular métricas
    dice = dice_coefficient(gt_mask, pred_mask)
    iou = iou_score(gt_mask, pred_mask)
    miou = mean_iou(gt_mask, pred_mask)
    mae = mean_absolute_error(gt_mask, pred_mask)
    game_levels = {L: game_score(gt_mask, pred_mask, L) for L in range(4)}

    print(f"DICE: {dice:.4f}")
    print(f"IoU : {iou:.4f}")
    print(f"mIoU : {miou:.4f}")
    print(f"MAE : {mae:.4f}")
    for L, game in game_levels.items():
        print(f"GAME(L={L}): {game}")

    # Mostrar resultados
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(img_rgb)
    axs[0].set_title("Imagen Original")
    axs[0].axis('off')

    axs[1].imshow(gt_mask, cmap='gray')
    axs[1].set_title("Máscara Manual")
    axs[1].axis('off')

    axs[2].imshow(pred_mask, cmap='gray')
    axs[2].set_title("Máscara Predicha")
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

# 👉 Parámetros de entrada
img_path = r'DICE_Masks\masks\657d1748-839a-4e18-ab64-a6cca9ec2e26.jpg'
gt_mask_path = r'DICE_Masks\masks\657d1748-839a-4e18-ab64-a6cca9ec2e26.png'
pred_mask_path = r'predicted_masks\512x512_657d1748-839a-4e18-ab64-a6cca9ec2e26_mask.png'

comparar_y_mostrar(img_path, gt_mask_path, pred_mask_path)
