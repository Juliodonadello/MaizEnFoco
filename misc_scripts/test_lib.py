from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import numpy as np

# Imagen de ejemplo
image = np.zeros((10, 10))
image[3:7, 3:7] = 1  # Región simple para segmentar

# Distancia
distance = ndi.distance_transform_edt(image)

# Obtener coordenadas de máximos locales
coordinates = peak_local_max(
    distance,
    labels=(image > 0).astype(int),
    footprint=np.ones((3, 3)),
    exclude_border=False
)

# Crear marcadores con etiquetas únicas
markers = np.zeros_like(image, dtype=np.int32)
for i, coord in enumerate(coordinates, start=1):
    markers[tuple(coord)] = i

# Aplicar watershed
labels = watershed(-distance, markers, mask=image)

# (Opcional) imprimir resultado
print("Segmented labels:\n", labels)
