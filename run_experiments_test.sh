#!/bin/bash

# Activar entorno virtual (asumiendo que estás en el root del proyecto)
source ./venv/Scripts/activate

# Lista de nombres de experimentos
EXPERIMENTS=(
    "weakly_aff_test_exp1"
    "weakly_aff_test_exp2"
    "weakly_aff_test_exp3"
    "weakly_aff_test_exp4"
    "weakly_aff_test_exp5"
    "weakly_aff_test_exp6"
)

# Ejecutar cada experimento secuencialmente
for EXP in "${EXPERIMENTS[@]}"; do
    echo "==============================="
    echo "Ejecutando experimento: $EXP"
    echo "==============================="
    python trainval.py -e "$EXP" -sb results -d DeepAgro_final
done
