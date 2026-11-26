"""Entrenamiento y comparación unificada: Referencia, SVM ajustado, RF ajustado y CNN."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import json
from importlib.metadata import version as version_paquete

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.utils.archivos import guardar_tabla
from src.data.base import verificar_imagenes
from src.models.evaluacion import (
    CALIDADES,
    FRUTAS,
    SEMILLA,
    cargar_base,
    consolidar_resultados_oof,
    crear_pliegues_externos,
    guardar_json,
)


def cargar_resultados_tradicionales(proyecto, base, estado, asignacion):
    """Reutilizar los experimentos 06/07 solo si base, grupos y bloques coinciden."""
    carpeta = proyecto / 'experiments/results/ajuste_modelos'
    metadata = carpeta / 'configuracion_ajuste.json'
    regenerar = not metadata.is_file() or not (carpeta / 'predicciones_oof.csv').is_file()
    if not regenerar:
        regenerar = json.loads(metadata.read_text(encoding='utf-8'))['sha_base'] != estado['sha_base']
    if regenerar:
        from src.models.ajuste_tradicionales import ejecutar_ajuste
        ejecutar_ajuste(base, estado['caracteristicas'], estado, proyecto, carpeta,
                        proyecto / 'experiments/checkpoints/ajuste_modelos')
    datos = pd.read_csv(carpeta / 'predicciones_oof.csv')
    modelos = ['Referencia_mayoritaria', 'SVM_inicial', 'Random_Forest_inicial',
               'SVM_ajustado', 'Random_Forest_ajustado']
    if set(datos.modelo) != set(modelos):
        raise ValueError('Faltan modelos tradicionales en la referencia.')
    predicciones = {}
    for nombre in modelos:
        sub = datos.loc[datos.modelo.eq(nombre)].set_index('id_ejemplar')
        if not sub.index.is_unique or set(sub.index) != set(base.id_ejemplar):
            raise ValueError('Las imágenes de la referencia no coinciden: ' + nombre)
        sub = sub.loc[base.id_ejemplar].reset_index()
        for columna in ['grupo_particion', 'calidad_revisada', 'fruta', 'fuente']:
            if sub[columna].tolist() != base[columna].tolist():
                raise ValueError('La referencia cambió en ' + columna)
        if sub.pliegue_validacion.tolist() != asignacion.tolist():
            raise ValueError('La referencia utiliza otros bloques de evaluación.')
        if not set(sub.prediccion) <= set(CALIDADES):
            raise ValueError('La referencia contiene predicciones fuera de alcance.')
        predicciones[nombre] = sub.prediccion.to_numpy()
    seleccionados = json.loads((carpeta / 'parametros_seleccionados.json').read_text(encoding='utf-8'))
    resumen = {r['modelo'] + '_ajustado': {
        'mejores_parametros_global': r['parametros'],
        'f1_macro_cv_interno_global': r['f1_interno_seleccion'],
        'origen': 'Experimento tradicional del cuaderno 07; predicciones externas verificadas.',
    } for r in seleccionados if r['etapa'] == 'prototipo_final'}
    return predicciones, resumen


def ejecutar_comparacion(
    proyecto: str | Path,
    incluir_cnn: bool = True,
    max_epocas_cnn: int = 120,
    hilos_cpu: int = 4,
) -> dict[str, pd.DataFrame]:
    """Ejecutar ajuste agrupado de ML tradicional y CNN sobre los mismos 3 pliegues externos."""
    proyecto = Path(proyecto).resolve()
    if incluir_cnn:
        try:
            from src.models.cnn_calidad import evaluar_transferencia_anidada, candidatos_transferencia
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError('Para la CNN instalar: python -m pip install -r requirements-dl.txt. '
                               'Para entrenar SVM y Random Forest sin PyTorch añadir --solo-ml.') from error
        import torch
        torch.set_num_threads(max(1, hilos_cpu))
    salida = proyecto / 'experiments/results/comparacion_modelos'
    checkpoints = proyecto / 'experiments/checkpoints/comparacion_modelos'
    salida.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    base, columnas, estado = cargar_base(proyecto)
    if incluir_cnn:
        verificar_imagenes(base)
    particiones, asignacion, cobertura = crear_pliegues_externos(base, n_splits=3, semilla=SEMILLA)
    guardar_tabla(cobertura, salida / 'cobertura_pliegues.csv')

    predicciones_oof, resumen_ml = cargar_resultados_tradicionales(proyecto, base, estado, asignacion)
    print('Referencias tradicionales verificadas: misma base, imágenes, grupos y pliegues.', flush=True)

    # Referencias históricas verificadas: mismos originales, etiquetas, máscaras y grupos.
    historico = salida / 'cnn_referencias_historicas.csv'
    historico_meta = salida / 'cnn_referencias_historicas.json'
    if historico.is_file() and historico_meta.is_file():
        from src.data.entrada_cnn import firma_comparable
        hm = json.loads(historico_meta.read_text(encoding='utf-8'))
        if hm['firma_comparable'] != firma_comparable(base):
            raise ValueError('Las referencias históricas usan otros ejemplares, etiquetas o grupos.')
        for nombre, sub in pd.read_csv(historico).groupby('modelo', sort=False):
            sub = sub.set_index('id_ejemplar').loc[base.id_ejemplar]
            for campo in ['grupo_particion', 'calidad_revisada', 'fruta', 'fuente']:
                if sub[campo].tolist() != base[campo].tolist():
                    raise ValueError('Cambió la referencia histórica en '+campo)
            if sub.pliegue_validacion.tolist() != asignacion.tolist():
                raise ValueError('Cambió el bloque de la referencia histórica.')
            predicciones_oof[nombre] = sub.prediccion.to_numpy()

    info_cnn = None
    if incluir_cnn:
        oof_cnn, info_cnn = evaluar_transferencia_anidada(
            base,
            particiones,
            proyecto=proyecto,
            salida=salida,
            checkpoints=checkpoints,
            estado=estado,
            candidatos=candidatos_transferencia(max_epocas=max_epocas_cnn),
            semilla=SEMILLA,
        )
        predicciones_oof['CNN_transferencia'] = oof_cnn

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
        'referencias_tradicionales': 'Predicciones OOF verificadas de los cuadernos 06 y 07; no se volvieron a seleccionar parámetros.',
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
    if incluir_cnn:
        metadata['versiones'].update(torch=torch.__version__, torchvision=version_paquete('torchvision'),
                                     pillow=version_paquete('pillow'), matplotlib=version_paquete('matplotlib'))
    guardar_json(metadata, salida / 'configuracion_comparacion.json')
    guardar_json(metadata, checkpoints / 'metadata.json')
    return tablas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proyecto', default='.')
    parser.add_argument('--solo-ml', action='store_true', help='Omitir entrenamiento de la CNN en PyTorch.')
    parser.add_argument('--epocas-cnn', type=int, default=120, help='Máximo de épocas por ajuste del cabezal CNN.')
    parser.add_argument('--hilos-cpu', type=int, default=4)
    args = parser.parse_args()

    tablas = ejecutar_comparacion(
        args.proyecto,
        incluir_cnn=not args.solo_ml,
        max_epocas_cnn=args.epocas_cnn,
        hilos_cpu=args.hilos_cpu,
    )
    print(tablas['metricas_globales'].to_string(index=False))


if __name__ == '__main__':
    main()
