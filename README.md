# Sistema de segmentación en cultivos de maíz utilizando IA aplicada
[Abstract](https://drive.google.com/file/d/1VsidXCayVYQGN2685RFOHyLIVe9KXFcS/view?usp=sharing)

Este repositorio contiene el código desarrollado para la segmentación de plantas de maíz en estadíos fenológicos tempranos utilizando técnicas de aprendizaje débilmente supervisado, con el objetivo de analizar la desuniformidad espacial del cultivo a partir de imágenes de campo. El proyecto fue realizado en colaboración con DeepAgro como parte de la Especialización en Inteligencia Artificial.

El sistema permite entrenar y validar modelos de segmentación semántica a partir de anotaciones basadas únicamente en puntos por planta, facilitando la inferencia y el análisis cuantitativo del desempeño mediante métricas específicas para tareas de conteo y localización.

## Estructura del Proyecto

```
. src/
  ├ datasets/        # Definiciones de datasets personalizados para DeepAgro
  ├ models/          # Implementación de arquitecturas FCN8 (VGG16, ResNet)
  └ utils/           # Funciones auxiliares para métricas, visualización y procesamiento
. exp_configs/         # Configuraciones de experimentos para entrenamiento/validación
. streamlit_app/       # Interfaz gráfica para visualización y validación de resultados
. results/             # Carpeta destino para almacenar resultados de los experimentos
```

## Configuración del Entorno

1. Crear y activar un entorno virtual:

```bash
python -m venv .venv
# Activar entorno virtual:
# En Windows
.venv\Scripts\activate
# En Linux / macOS
source .venv/bin/activate
```

2. Instalar las dependencias requeridas:

```bash
pip install -r requirements.txt
```

## Ejecución del Entrenamiento

El sistema utiliza configuraciones de experimentos almacenadas en la carpeta `exp_configs/`. Estas contienen los hiperparámetros y rutas necesarias para cada entrenamiento. La ejecución se realiza mediante el siguiente comando:

```bash
python trainval.py -e <nombre_experimento> -sb results -d <nombre_dataset>
```

Por ejemplo, para reproducir un experimento estándar:

```bash
python trainval.py -e weakly_aff_DeepAgro_exp6 -sb results -d DeepAgro
```

### Parámetros principales:
- `-e`: Nombre del archivo de configuración dentro de `exp_configs/` (sin la extensión `.yaml`).
- `-sb`: Carpeta donde se guardarán los resultados (puede usarse `results`).
- `-d`: Dataset a utilizar (`DeepAgro` en este proyecto).

## Inferencia y Validación Visual

La interfaz ubicada en `streamlit_app/` permite realizar inferencias sobre imágenes individuales, ajustar parámetros de visualización y validar cuantitativamente las predicciones. Se puede lanzar con:

```bash
streamlit run streamlit_app/app.py
```

Esta herramienta facilita la visualización de:
- Segmentaciones binarias superpuestas
- Mapas de calor de predicciones
- Centroides detectados y su comparación contra anotaciones manuales
- Métricas de precisión espacial

## Descripción del Proyecto

El presente desarrollo tiene como finalidad asistir a la detección y cuantificación de la desuniformidad en cultivos de maíz, ofreciendo herramientas tanto para investigación como para futuros desarrollos aplicados en agricultura de precisión. Se trabajó sobre un dataset privado proporcionado por DeepAgro, compuesto por imágenes a nivel de suelo y anotaciones puntuales por planta.

Las arquitecturas utilizadas están basadas en redes convolucionales totalmente convolucionales (FCN8), empleando mecanismos de afinidad para el refinamiento de las predicciones bajo esquemas de supervisión débil.

## Métricas de Evaluación Utilizadas

- Dice Score (DICE)
- Precisión por clase
- Mean Absolute Error (MAE)
- Grid Average Mean Absolute Error (GAME)
- Precisión espacial mediante máscaras de puntos


## Estructura del Dataset

El dataset utilizado para entrenar y validar los modelos proviene del uso de la clase custom "DeepAgro". Este dataset se encuentra organizado de la siguiente manera:

```
DeepAgro/
├── Segmentation/
│   ├── images/
│   │   ├── empty/   # Imágenes sin instancias (útiles para validación de falsos positivos)
│   │   └── valid/   # Imágenes con instancias de plantas
│   ├── masks/       # Máscaras binarias con puntos que representan el centro aproximado de cada planta
│   │   ├── empty/   # Máscaras sin instancias (útiles para validación de falsos positivos)
│   │   └── valid/   # Máscaras con instancias de plantas
│   ├── segmentation.csv  # Relación imagen - máscara para todo el dataset
│   ├── train.csv         # Subconjunto para entrenamiento
│   ├── val.csv           # Subconjunto para validación
│   └── test.csv          # Subconjunto para prueba
```

Las máscaras contenidas en `masks/` no corresponden a segmentaciones completas por contorno, sino que son imágenes binarias donde cada instancia está representada únicamente por un píxel blanco en su posición central aproximada. Estas máscaras de puntos fueron utilizadas como supervisión débil para guiar el modelo en la detección y segmentación de las plantas.


## Licencia

Este código fue desarrollado exclusivamente para fines académicos y de investigación en el marco de la Especialización en Inteligencia Artificial. No se autoriza su uso comercial sin expresa autorización de los autores o de la empresa DeepAgro.
