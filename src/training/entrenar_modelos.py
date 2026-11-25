"""Entrenamiento y comparación unificada: Referencia, SVM ajustado, RF ajustado y CNN."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.utils.archivos import guardar_tabla
from src.models.evaluacion import (
    CALIDADES,
    FRUTAS,
    SEMILLA,
    cargar_base_v2,
    consolidar_resultados_oof,
    crear_pliegues_externos,
    guardar_json,
)
from src.models.modelos_tradicionales import evaluar_tradicionales_anidados


def ejecutar_comparacion(
    proyecto: str | Path,
    incluir_cnn: bool = True,
    max_epocas_cnn: int = 25,
) -> dict[str, pd.DataFrame]:
    """Ejecutar ajuste agrupado de ML tradicional y CNN sobre los mismos 3 pliegues externos."""
    proyecto = Path(proyecto).resolve()
    salida = proyecto / 'experiments/results/comparacion_modelos_v2'
    checkpoints = proyecto / 'experiments/checkpoints/comparacion_v2'
    salida.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    base, columnas, estado = cargar_base_v2(proyecto)
    particiones, asignacion, cobertura = crear_pliegues_externos(base, n_splits=3, semilla=SEMILLA)
    guardar_tabla(cobertura, salida / 'cobertura_pliegues.csv')

    predicciones_oof, resumen_ml = evaluar_tradicionales_anidados(
        base, columnas, particiones, estado, salida, checkpoints, semilla=SEMILLA
    )

    info_cnn = None
    if incluir_cnn:
        from src.models.cnn_calidad import evaluar_cnn_anidada, rejilla_cnn_por_defecto

        oof_cnn, info_cnn = evaluar_cnn_anidada(
            base,
            particiones,
            proyecto=proyecto,
            salida=salida,
            checkpoints=checkpoints,
            estado=estado,
            candidatos=rejilla_cnn_por_defecto(max_epocas=max_epocas_cnn),
            semilla=SEMILLA,
        )
        predicciones_oof['CNN_simple'] = oof_cnn

    tablas = consolidar_resultados_oof(base, particiones, asignacion, predicciones_oof, salida)

    metadata = {
        'fecha_utc': datetime.now(timezone.utc).isoformat(),
        'sha_base': estado['sha_base'],
        'imagenes': len(base),
        'grupos': int(base.grupo_particion.nunique()),
        'calidades': CALIDADES,
        'frutas': FRUTAS,
        'validacion_externa': 'StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)',
        'validacion_interna': 'StratifiedGroupKFold dentro de cada pliegue de entrenamiento (sin fuga)',
        'modelos_evaluados': list(predicciones_oof.keys()),
        'resumen_ml': resumen_ml,
        'resumen_cnn': info_cnn,
        'versiones': {
            'python': platform.python_version(),
            'scikit_learn': sklearn.__version__,
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'joblib': joblib.__version__,
        },
        'evaluacion_independiente': False,
    }
    guardar_json(metadata, salida / 'configuracion_comparacion.json')
    guardar_json(metadata, checkpoints / 'metadata.json')
    return tablas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proyecto', default='.')
    parser.add_argument('--solo-ml', action='store_true', help='Omitir entrenamiento de la CNN en PyTorch.')
    parser.add_argument('--epocas-cnn', type=int, default=25, help='Máximo de épocas por candidato CNN.')
    args = parser.parse_args()

    tablas = ejecutar_comparacion(
        args.proyecto,
        incluir_cnn=not args.solo_ml,
        max_epocas_cnn=args.epocas_cnn,
    )
    print(tablas['metricas_globales'].to_string(index=False))


if __name__ == '__main__':
    main()
