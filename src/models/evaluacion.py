"""Utilidades compartidas de partición por grupos y evaluación OOF."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold

from src.utils.archivos import guardar_bytes_si_cambio, guardar_tabla

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

CALIDADES = ['buena', 'intermedia', 'mala']
FRUTAS = ['manzana', 'banano', 'limon']
COLORES = ['#36856c', '#dea33c', '#ba5260']
SEMILLA = 42


def cargar_base_v2(proyecto: str | Path) -> tuple[pd.DataFrame, list[str], dict]:
    """Cargar y validar integridad de base_revisada_v2/base_svm_rf.csv."""
    proyecto = Path(proyecto).resolve()
    ruta = proyecto / 'data/processed/base_revisada_v2/base_svm_rf.csv'
    if not ruta.is_file():
        raise FileNotFoundError(
            f'No se encontró {ruta}. Ejecuta primero el flujo de datos hasta el cuaderno 04/05.'
        )
    estado = json.loads(ruta.with_name('estado_base.json').read_text(encoding='utf-8'))
    base = pd.read_csv(ruta)
    columnas = estado['caracteristicas']
    if hashlib.sha256(ruta.read_bytes()).hexdigest() != estado['sha_base']:
        raise ValueError('La base cambió: actualizar primero la exportación del cuaderno 04.')
    if len(columnas) != 43 or not all(c.startswith(('color_', 'textura_', 'forma_')) for c in columnas):
        raise ValueError('Se esperan las 43 características de imagen; no incluir identificadores.')
    if not np.isfinite(base[columnas].to_numpy(dtype=float)).all():
        raise ValueError('Hay características faltantes o no finitas.')
    if base[['id_ejemplar', 'grupo_particion', 'calidad_revisada', 'fruta']].isna().any().any():
        raise ValueError('Hay etiquetas o grupos faltantes.')
    if base.id_ejemplar.duplicated().any() or set(base.calidad_revisada) != set(CALIDADES):
        raise ValueError('Revisar identificadores únicos y las tres calidades.')
    if set(base.fruta) != set(FRUTAS):
        raise ValueError('Revisar el alcance de frutas.')
    return base, columnas, estado


def guardar_json(objeto: dict, ruta: str | Path) -> None:
    """Guardar JSON UTF-8 solo si el contenido cambió."""
    contenido = json.dumps(objeto, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    guardar_bytes_si_cambio(ruta, contenido)


def guardar_grafica(figura: plt.Figure, ruta: str | Path) -> None:
    """Serializar figura Matplotlib a PNG con fondo blanco."""
    contenido = io.BytesIO()
    figura.savefig(contenido, format='png', dpi=155, bbox_inches='tight', facecolor='white')
    guardar_bytes_si_cambio(ruta, contenido.getvalue())


def crear_pliegues_externos(
    base: pd.DataFrame,
    n_splits: int = 3,
    semilla: int = SEMILLA,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, pd.DataFrame]:
    """Partición fija estratificada por grupos, sin fuga entre entrenamiento y validación."""
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semilla)
    particiones = list(cv.split(base, base.calidad_revisada, base.grupo_particion))
    asignacion = np.full(len(base), -1, dtype=int)
    cobertura = []
    for pliegue, (entrenamiento, validacion) in enumerate(particiones, 1):
        grupos_train = set(base.iloc[entrenamiento].grupo_particion)
        grupos_val = set(base.iloc[validacion].grupo_particion)
        assert not (grupos_train & grupos_val)
        for indices, etapa in [(entrenamiento, 'entrenamiento'), (validacion, 'validacion')]:
            tabla = base.iloc[indices]
            if set(tabla.calidad_revisada) != set(CALIDADES):
                raise ValueError(f'El pliegue {pliegue} no tiene las tres calidades en {etapa}.')
            for fruta in FRUTAS:
                for calidad in CALIDADES:
                    sub = tabla.loc[(tabla.fruta == fruta) & (tabla.calidad_revisada == calidad)]
                    cobertura.append({
                        'pliegue': pliegue,
                        'etapa': etapa,
                        'fruta': fruta,
                        'calidad': calidad,
                        'imagenes': len(sub),
                        'grupos': sub.grupo_particion.nunique(),
                    })
        assert (asignacion[validacion] == -1).all()
        asignacion[validacion] = pliegue
    assert (asignacion > 0).all()
    return particiones, asignacion, pd.DataFrame(cobertura)


def crear_cv_interno(
    y_train: pd.Series | np.ndarray,
    grupos_train: pd.Series | np.ndarray,
    max_splits: int = 3,
    semilla: int = SEMILLA,
) -> StratifiedGroupKFold:
    """Elegir un número de pliegues internos seguro según los grupos disponibles."""
    tabla = pd.DataFrame({'y': np.asarray(y_train), 'g': np.asarray(grupos_train)})
    min_grupos_clase = int(tabla.groupby('y')['g'].nunique().min())
    total_grupos = int(tabla['g'].nunique())
    n_splits = max(2, min(max_splits, total_grupos, min_grupos_clase if min_grupos_clase >= 2 else 2))
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=semilla)


def calcular_metricas(real, predicha, pesos=None) -> dict[str, float]:
    """Calcular exactitud, F1 macro y recall macro (equivalente a balanced accuracy)."""
    return {
        'exactitud': float(accuracy_score(real, predicha, sample_weight=pesos)),
        'f1_macro': float(f1_score(real, predicha, labels=CALIDADES, average='macro', zero_division=0, sample_weight=pesos)),
        'recall_macro': float(recall_score(real, predicha, labels=CALIDADES, average='macro', zero_division=0, sample_weight=pesos)),
    }


def consolidar_resultados_oof(
    base: pd.DataFrame,
    particiones: list[tuple[np.ndarray, np.ndarray]],
    asignacion: np.ndarray,
    predicciones_por_modelo: dict[str, np.ndarray],
    salida: str | Path,
    generar_graficas: bool = True,
) -> dict[str, pd.DataFrame]:
    """Calcular tablas de desempeño OOF, matrices de confusión y comparativa visual."""
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    y = base.calidad_revisada
    pesos = 1 / base.groupby('grupo_particion').id_ejemplar.transform('size').to_numpy()

    asignaciones = base[['id_ejemplar', 'grupo_particion', 'fruta', 'calidad_revisada', 'fuente']].copy()
    asignaciones['pliegue_validacion'] = asignacion
    guardar_tabla(asignaciones, salida / 'asignacion_pliegues.csv')

    resultados, por_pliegue, por_fruta, por_fuente, por_clase, predicciones = [], [], [], [], [], []
    matrices = {}

    for nombre, oof in predicciones_por_modelo.items():
        oof = np.asarray(oof, dtype=object)
        if len(oof) != len(base) or (oof == '').any():
            raise ValueError(f'Predicciones OOF incompletas para {nombre}.')

        for pliegue, (train, val) in enumerate(particiones, 1):
            por_pliegue.append({
                'modelo': nombre,
                'pliegue': pliegue,
                'n_train': len(train),
                'n_validacion': len(val),
                **calcular_metricas(y.iloc[val], oof[val]),
            })

        resultados.append({
            'modelo': nombre,
            'imagenes': len(base),
            **calcular_metricas(y, oof),
            **{f'{k}_peso_grupo': v for k, v in calcular_metricas(y, oof, pesos).items()},
        })

        tabla = asignaciones.copy()
        tabla['modelo'] = nombre
        tabla['prediccion'] = oof
        tabla['acierto'] = tabla.calidad_revisada == tabla.prediccion
        predicciones.append(tabla)

        for campo, destino in [('fruta', por_fruta), ('fuente', por_fuente)]:
            for categoria, sub in tabla.groupby(campo):
                destino.append({
                    'modelo': nombre,
                    campo: categoria,
                    'imagenes': len(sub),
                    **calcular_metricas(sub.calidad_revisada, sub.prediccion),
                })

        matriz = confusion_matrix(y, oof, labels=CALIDADES)
        matrices[nombre] = matriz
        guardar_tabla(
            pd.DataFrame(matriz, columns=CALIDADES).assign(real=CALIDADES)[['real', *CALIDADES]],
            salida / f'matriz_confusion_{nombre}.csv',
        )
        for clase in CALIDADES:
            por_clase.append({
                'modelo': nombre,
                'calidad': clase,
                'imagenes': int((y == clase).sum()),
                'recall': float(recall_score(y, oof, labels=[clase], average='macro', zero_division=0)),
                'f1': float(f1_score(y, oof, labels=[clase], average='macro', zero_division=0)),
            })

    tablas = {
        'metricas_globales': pd.DataFrame(resultados),
        'metricas_pliegues': pd.DataFrame(por_pliegue),
        'metricas_por_fruta': pd.DataFrame(por_fruta),
        'metricas_por_fuente': pd.DataFrame(por_fuente),
        'metricas_por_clase': pd.DataFrame(por_clase),
        'predicciones_oof': pd.concat(predicciones, ignore_index=True),
    }
    for nombre, tabla in tablas.items():
        guardar_tabla(tabla, salida / f'{nombre}.csv')

    if generar_graficas:
        _graficar_matrices(matrices, salida / 'matrices_confusion.png')
        _graficar_comparacion(tablas['metricas_globales'], salida / 'comparacion_modelos.png')
    return tablas


def _graficar_matrices(matrices: dict[str, np.ndarray], ruta: Path) -> None:
    import matplotlib.pyplot as plt

    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4.0), layout='constrained')
    if n == 1:
        axes = [axes]
    for ax, (nombre, matriz) in zip(axes, matrices.items()):
        filas_sum = np.maximum(matriz.sum(axis=1, keepdims=True), 1)
        ax.imshow(matriz / filas_sum, vmin=0, vmax=1, cmap='Blues')
        ax.set(
            xticks=range(3),
            yticks=range(3),
            xticklabels=CALIDADES,
            yticklabels=CALIDADES,
            xlabel='Predicción',
            ylabel='Calidad revisada',
            title=nombre.replace('_', ' '),
        )
        for i in range(3):
            for j in range(3):
                proporcion = matriz[i, j] / filas_sum[i, 0]
                ax.text(
                    j,
                    i,
                    f'{matriz[i, j]}\n{proporcion:.0%}',
                    ha='center',
                    va='center',
                    color='white' if proporcion > 0.55 else '#222222',
                )
    fig.suptitle('Predicciones fuera del entrenamiento (OOF) · Validación agrupada')
    guardar_grafica(fig, ruta)
    plt.close(fig)


def _graficar_comparacion(metricas_globales: pd.DataFrame, ruta: Path) -> None:
    import matplotlib.pyplot as plt

    df = metricas_globales.set_index('modelo')[['exactitud', 'f1_macro', 'recall_macro']]
    fig, ax = plt.subplots(figsize=(9, 4.5), layout='constrained')
    df.plot.bar(ax=ax, color=['#4c72b0', '#36856c', '#dea33c'], rot=15)
    ax.set(ylim=(0, 1.0), ylabel='Puntuación', xlabel='', title='Comparación OOF de modelos')
    ax.legend(['Exactitud', 'F1 macro', 'Recall macro'], frameon=False)
    for cont in ax.containers:
        ax.bar_label(cont, fmt='%.3f', fontsize=8, padding=2)
    guardar_grafica(fig, ruta)
    plt.close(fig)
