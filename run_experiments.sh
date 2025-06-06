#!/bin/bash

# Activar entorno virtual (asumiendo que estás en el root del proyecto)
source ./venv/bin/activate

# Lista de nombres de experimentos
EXPERIMENTS=(
    "weakly_aff_DeepAgro_exp1"
    "weakly_aff_DeepAgro_exp2"
    "weakly_aff_DeepAgro_exp3"
    "weakly_aff_DeepAgro_exp4"
    "weakly_aff_DeepAgro_exp5"
)

# Ejecutar cada experimento secuencialmente
for EXP in "${EXPERIMENTS[@]}"; do
    echo "==============================="
    echo "Ejecutando experimento: $EXP"
    echo "==============================="
    python trainval.py -e "$EXP" -sb results -d DeepAgro
done
