'''import haven.haven_utils as hu

score_list_best = 'results/bda17a14f36f20116fc10e369d181814/score_list_best.pkl'
score_list = hu.load_pkl(score_list_best)

print(score_list)  # Mostrará el contenido del archivo
'''
import pandas as pd
import haven.haven_utils as hu
import matplotlib.pyplot as plt

# Ruta corregida (doble barra invertida o raw string para evitar el warning)
score_list_best = r'.\results\8f7844d98e8d29e93b3831b9576a7db4\score_list_best.pkl'
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

# Crear plots para visualizar las métricas
metrics_to_plot = ['val_score', 'val_mae', 'val_game', 'train_loss', 'test_score', 'test_mae', 'test_game']

# Filtrar solo las métricas que están disponibles en el DataFrame
available_metrics = [metric for metric in metrics_to_plot if metric in df.columns]

if available_metrics and 'epoch' in df.columns:
    # Configurar el layout de subplots
    n_metrics = len(available_metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Redondear hacia arriba
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    
    # Si solo hay una fila, asegurar que axes sea una matriz 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Crear cada subplot
    for i, metric in enumerate(available_metrics):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot de la métrica
        ax.plot(df['epoch'], df[metric], marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        
        # Encontrar el mejor valor y marcarlo
        if 'loss' in metric.lower() or 'mae' in metric.lower() or 'game' in metric.lower():
            # Para métricas que deben minimizarse
            best_idx = df[metric].idxmin()
            best_value = df[metric].min()
            color = 'red'
            marker_label = f'Min: {best_value:.3f} (epoch {df.loc[best_idx, "epoch"]})'
        else:
            # Para métricas que deben maximizarse (scores)
            best_idx = df[metric].idxmax()
            best_value = df[metric].max()
            color = 'green'
            marker_label = f'Max: {best_value:.3f} (epoch {df.loc[best_idx, "epoch"]})'
        ''' 
        ax.scatter(df.loc[best_idx, 'epoch'], best_value, color=color, s=100, zorder=5)
        ax.annotate(marker_label, 
                   xy=(df.loc[best_idx, 'epoch'], best_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   fontsize=8)
        '''
    # Ocultar subplots vacíos
    for i in range(len(available_metrics), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Crear un plot adicional con todas las métricas normalizadas para comparación
    if len(available_metrics) > 1:
        plt.figure(figsize=(12, 8))
        
        # Normalizar las métricas entre 0 y 1 para poder compararlas
        df_normalized = df[['epoch'] + available_metrics].copy()
        for metric in available_metrics:
            min_val = df[metric].min()
            max_val = df[metric].max()
            if max_val != min_val:
                df_normalized[metric] = (df[metric] - min_val) / (max_val - min_val)
            else:
                df_normalized[metric] = 0.5  # Si todos los valores son iguales
        
        # Plot de todas las métricas normalizadas
        for metric in available_metrics:
            plt.plot(df_normalized['epoch'], df_normalized[metric], 
                    marker='o', label=metric, linewidth=2, markersize=4)
        
        plt.xlabel('Epoch')
        plt.ylabel('Valor Normalizado (0-1)')
        plt.title('Comparación de Todas las Métricas (Normalizadas)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
else:
    print("No se encontraron métricas válidas para graficar o falta la columna 'epoch'")
    print(f"Columnas disponibles: {list(df.columns)}")
