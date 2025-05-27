'''import haven.haven_utils as hu

score_list_best = 'results/bda17a14f36f20116fc10e369d181814/score_list_best.pkl'
score_list = hu.load_pkl(score_list_best)

print(score_list)  # Mostrará el contenido del archivo
'''
import pandas as pd
import haven.haven_utils as hu

# Ruta corregida (doble barra invertida o raw string para evitar el warning)
score_list_best = r'results\with_affinity&lcfcn_loss\score_list_best.pkl'
score_list = hu.load_pkl(score_list_best)

# Convertir la lista de diccionarios a un DataFrame de pandas
df = pd.DataFrame(score_list)

# Mostrar solo algunas columnas clave si deseas resumir
columns_to_show = [
    'epoch', 'train_loss', 'val_score', 'val_mae', 'test_score', 'test_mae'
]
# Filtrar columnas disponibles en el DataFrame
columns_to_show = [col for col in columns_to_show if col in df.columns]

# Redondear los resultados para mejor visualización
df_display = df[columns_to_show].round(3)

# Imprimir como tabla
#print(df_display.to_string(index=False))
print(df.to_string(index=False))
