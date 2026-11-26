"""Búsqueda de parámetros dentro de los pliegues de entrenamiento del cuaderno 06."""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import platform
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedGroupKFold

from src.data.preparacion_imagenes import guardar_bytes_si_cambio, guardar_tabla, leer_rgb
from .modelos_tradicionales import (
    CALIDADES, FRUTAS, SEMILLA, calcular_metricas, crear_modelos, crear_pliegues,
    guardar_grafica, guardar_json, entrenar_y_comparar,
)


def espacios_busqueda():
    """Espacios fijos: incluyen los parámetros del experimento inicial."""
    return {
        'SVM': [
            {'svc__kernel': ['rbf'], 'svc__C': [.1, 1., 10.],
             'svc__gamma': ['scale', .01], 'svc__class_weight': [None, 'balanced']},
            {'svc__kernel': ['linear'], 'svc__C': [.1, 1., 10.],
             'svc__class_weight': [None, 'balanced']},
        ],
        'Random_Forest': {
            'max_depth': [None, 8], 'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', .7],
        },
    }


def preparar_plan(base, estado, salida):
    """Exportar las particiones anidadas antes de consultar resultados de ajuste."""
    exteriores, asignacion, _ = crear_pliegues(base)
    internos, registros, cobertura = {}, [], []
    for exterior, (train, val) in enumerate(exteriores, 1):
        sub = base.iloc[train]
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEMILLA)
        internos[exterior] = list(cv.split(sub, sub.calidad_revisada, sub.grupo_particion))
        grupos_externos = set(base.iloc[val].grupo_particion)
        for interior, (a, b) in enumerate(internos[exterior], 1):
            grupos_a, grupos_b = set(sub.iloc[a].grupo_particion), set(sub.iloc[b].grupo_particion)
            if grupos_a & grupos_b or (grupos_a | grupos_b) & grupos_externos:
                raise ValueError('Se mezclaron grupos entre selección de parámetros y evaluación.')
            if set(sub.iloc[a].calidad_revisada) != set(CALIDADES):
                raise ValueError('El entrenamiento interno necesita las tres calidades globales.')
            for etapa, indices in [('entrenamiento', a), ('validacion', b)]:
                filas = sub.iloc[indices]
                for calidad in CALIDADES:
                    cobertura.append({'exterior': exterior, 'interior': interior, 'etapa': etapa,
                                      'calidad': calidad, 'imagenes': int(filas.calidad_revisada.eq(calidad).sum())})
                for fila in filas.itertuples():
                    registros.append({'exterior': exterior, 'interior': interior, 'etapa': etapa,
                                      'id_ejemplar': fila.id_ejemplar, 'grupo_particion': fila.grupo_particion})
    salida = Path(salida)
    guardar_tabla(pd.DataFrame(registros), salida / 'particiones_internas.csv')
    guardar_tabla(pd.DataFrame(cobertura), salida / 'cobertura_interna.csv')
    asignaciones = base[['id_ejemplar', 'grupo_particion', 'calidad_revisada', 'fruta', 'fuente']].copy()
    asignaciones['pliegue_validacion'] = asignacion
    guardar_tabla(asignaciones, salida / 'particiones_externas.csv')
    guardar_json({
        'sha_base': estado['sha_base'], 'semilla': SEMILLA, 'pliegues_externos': 3,
        'pliegues_internos': 3, 'metrica_seleccion': 'Promedio del F1 macro interno con tres etiquetas fijas.',
        'espacios': espacios_busqueda(),
        'candidatos': {k: len(ParameterGrid(v)) for k, v in espacios_busqueda().items()},
        'regla': 'Los parámetros se seleccionan dentro del entrenamiento externo; no se modifican por el resultado externo.',
    }, salida / 'plan_busqueda.json')
    return exteriores, internos, asignaciones, pd.DataFrame(cobertura)


def cargar_referencia(base, estado, proyecto, asignaciones):
    carpeta = Path(proyecto) / 'experiments/results/eda_modelos'
    metadata = carpeta / 'configuracion_entrenamiento.json'
    necesita_referencia = not metadata.is_file() or not (carpeta / 'predicciones_oof.csv').is_file()
    if not necesita_referencia:
        necesita_referencia = json.loads(metadata.read_text(encoding='utf-8'))['sha_base'] != estado['sha_base']
    if necesita_referencia:
        print('Generando la referencia inicial para esta base; sus resultados son regenerables.', flush=True)
        entrenar_y_comparar(base, estado['caracteristicas'], estado, carpeta,
                           Path(proyecto) / 'experiments/checkpoints/modelos_tradicionales')
    meta = json.loads((carpeta / 'configuracion_entrenamiento.json').read_text(encoding='utf-8'))
    if meta['sha_base'] != estado['sha_base']:
        raise ValueError('La referencia usa otra base. Ejecutar primero el cuaderno 06.')
    pred = pd.read_csv(carpeta / 'predicciones_oof.csv')
    if set(pred.modelo) != {'SVM', 'Random_Forest', 'Referencia_mayoritaria'}:
        raise ValueError('Faltan modelos en las predicciones de referencia.')
    for nombre, filas in pred.groupby('modelo'):
        columnas = ['id_ejemplar', 'grupo_particion', 'calidad_revisada', 'fruta', 'fuente', 'pliegue_validacion']
        if not filas[columnas].reset_index(drop=True).equals(asignaciones[columnas].reset_index(drop=True)):
            raise ValueError(f'Las predicciones de {nombre} no coinciden con la base y los pliegues actuales.')
    pred['modelo'] = pred.modelo.replace({'SVM': 'SVM_inicial', 'Random_Forest': 'Random_Forest_inicial'})
    return pred


def buscar(prototipo, espacio, X, y, pliegues):
    puntuacion = make_scorer(f1_score, labels=CALIDADES, average='macro', zero_division=0)
    busqueda = GridSearchCV(clone(prototipo), espacio, cv=pliegues, scoring=puntuacion,
                           n_jobs=1, refit=True, error_score='raise', return_train_score=True)
    return busqueda.fit(X, y)


def registrar_busqueda(busqueda, nombre, etapa, exterior):
    resultados = pd.DataFrame(busqueda.cv_results_)
    filas = []
    for fila in resultados.to_dict('records'):
        registro = {
            'modelo': nombre, 'etapa': etapa, 'exterior': exterior,
            'parametros': json.dumps(fila['params'], ensure_ascii=False, sort_keys=True),
            'f1_macro_interno': float(fila['mean_test_score']),
            'desv_f1_interno': float(fila['std_test_score']),
            'f1_macro_entrenamiento': float(fila['mean_train_score']),
            'rango': int(fila['rank_test_score']),
        }
        for i in range(3):
            registro[f'f1_validacion_interna_{i+1}'] = float(fila[f'split{i}_test_score'])
        filas.append(registro)
    return filas


def ejecutar_ajuste(base, columnas, estado, proyecto, salida, checkpoints):
    salida, checkpoints = Path(salida), Path(checkpoints)
    exteriores, internos, asignaciones, _ = preparar_plan(base, estado, salida)
    referencia = cargar_referencia(base, estado, proyecto, asignaciones)
    X, y = base[columnas], base.calidad_revisada
    predicciones = [referencia]
    resultados_busqueda, parametros = [], []
    for nombre, espacio in espacios_busqueda().items():
        prototipo = crear_modelos()[nombre]
        oof = np.full(len(base), '', dtype=object)
        for exterior, (train, val) in enumerate(exteriores, 1):
            print(f'{nombre}: ajuste interno del bloque {exterior}/3 ({len(ParameterGrid(espacio))} candidatos).', flush=True)
            seleccion = buscar(prototipo, espacio, X.iloc[train], y.iloc[train], internos[exterior])
            oof[val] = seleccion.predict(X.iloc[val])
            resultados_busqueda.extend(registrar_busqueda(seleccion, nombre, 'evaluacion_externa', exterior))
            parametros.append({'modelo': nombre, 'etapa': 'evaluacion_externa', 'exterior': exterior,
                               'parametros': seleccion.best_params_, 'f1_interno_seleccion': float(seleccion.best_score_)})
            print(f'  Bloque {exterior} terminado; F1 externo: {calcular_metricas(y.iloc[val], oof[val])["f1_macro"]:.3f}.', flush=True)
        if (oof == '').any():
            raise ValueError('Faltan predicciones fuera del entrenamiento.')
        tabla = asignaciones.copy()
        tabla['modelo'], tabla['prediccion'] = f'{nombre}_ajustado', oof
        tabla['acierto'] = tabla.calidad_revisada.eq(tabla.prediccion)
        predicciones.append(tabla)

        print(f'{nombre}: selección final con los grupos de toda la base de desarrollo.', flush=True)
        final = buscar(prototipo, espacio, X, y, exteriores)
        resultados_busqueda.extend(registrar_busqueda(final, nombre, 'prototipo_final', 0))
        parametros.append({'modelo': nombre, 'etapa': 'prototipo_final', 'exterior': 0,
                           'parametros': final.best_params_, 'f1_interno_seleccion': float(final.best_score_)})
        paquete = {
            'modelo': final.best_estimator_, 'caracteristicas': columnas, 'calidades': CALIDADES,
            'frutas': FRUTAS, 'sha_base': estado['sha_base'], 'parametros': final.best_params_,
            'alcance': 'Prototipo ajustado sobre toda la base de desarrollo; no es un modelo validado en prueba independiente.',
        }
        buffer = io.BytesIO()
        joblib.dump(paquete, buffer, compress=3)
        ruta = checkpoints / f'{nombre.lower()}_ajustado.joblib'
        guardar_bytes_si_cambio(ruta, buffer.getvalue())
        recargado = joblib.load(ruta)
        if not np.array_equal(recargado['modelo'].predict(X), final.predict(X)):
            raise ValueError('El modelo guardado no reproduce sus predicciones.')

    predicciones = pd.concat(predicciones, ignore_index=True)
    guardar_tabla(predicciones, salida / 'predicciones_oof.csv')
    guardar_tabla(pd.DataFrame(resultados_busqueda), salida / 'busquedas_completas.csv')
    guardar_json(parametros, salida / 'parametros_seleccionados.json')
    tablas = resumir_resultados(base, predicciones, salida)
    tablas['predicciones_oof'] = predicciones
    tablas['busquedas'] = pd.DataFrame(resultados_busqueda)
    metadata = {
        'fecha_utc': datetime.now(timezone.utc).isoformat(), 'sha_base': estado['sha_base'],
        'imagenes': len(base), 'grupos': int(base.grupo_particion.nunique()),
        'caracteristicas': columnas, 'objetivo': 'calidad_revisada',
        'metodo': 'Validación anidada: 3 bloques externos fijos y 3 internos agrupados, semilla 42.',
        'seleccion': 'F1 macro medio de validación interna; nunca puntuación externa.',
        'prototipo_final': 'Nueva búsqueda por grupos sobre toda la base; sus parámetros pueden diferir entre bloques externos.',
        'parametros_seleccionados': parametros,
        'versiones': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__},
        'evaluacion_independiente': False,
        'limitacion': 'Diseño posterior al EDA y a la referencia. Campañas desbalanceadas y algunas clases ausentes en validación interna. Métricas exploratorias, sin garantía de mejora.',
    }
    guardar_json(metadata, salida / 'configuracion_ajuste.json')
    guardar_json(metadata, checkpoints / 'metadata.json')
    return tablas


def resumir_resultados(base, predicciones, salida):
    globales, por_clase, por_fruta, por_grupo, pliegues = [], [], [], [], []
    pesos = 1 / base.groupby('grupo_particion').id_ejemplar.transform('size').to_numpy()
    for modelo, tabla in predicciones.groupby('modelo', sort=False):
        globales.append({'modelo': modelo, **calcular_metricas(tabla.calidad_revisada, tabla.prediccion),
                         **{f'{k}_peso_grupo': v for k, v in calcular_metricas(tabla.calidad_revisada, tabla.prediccion, pesos).items()}})
        for clase in CALIDADES:
            por_clase.append({'modelo': modelo, 'calidad': clase, 'imagenes': int(tabla.calidad_revisada.eq(clase).sum()),
                              'recall': float(recall_score(tabla.calidad_revisada, tabla.prediccion, labels=[clase], average='macro', zero_division=0)),
                              'f1': float(f1_score(tabla.calidad_revisada, tabla.prediccion, labels=[clase], average='macro', zero_division=0))})
        for fruta, sub in tabla.groupby('fruta'):
            por_fruta.append({'modelo': modelo, 'fruta': fruta, 'imagenes': len(sub),
                              **calcular_metricas(sub.calidad_revisada, sub.prediccion)})
        for grupo, sub in tabla.groupby('grupo_particion'):
            por_grupo.append({'modelo': modelo, 'grupo_particion': grupo, 'fruta': ','.join(sorted(sub.fruta.unique())),
                              'fuente': ','.join(sorted(sub.fuente.unique())), 'imagenes': len(sub),
                              'errores': int((~sub.acierto).sum()),
                              'intermedias': int(sub.calidad_revisada.eq('intermedia').sum()),
                              'intermedias_correctas': int((sub.calidad_revisada.eq('intermedia') & sub.acierto).sum())})
        for pliegue, sub in tabla.groupby('pliegue_validacion'):
            pliegues.append({'modelo': modelo, 'pliegue': int(pliegue), 'imagenes': len(sub),
                             **calcular_metricas(sub.calidad_revisada, sub.prediccion)})
    tablas = {'metricas_globales': pd.DataFrame(globales), 'metricas_por_clase': pd.DataFrame(por_clase),
              'metricas_por_fruta': pd.DataFrame(por_fruta), 'errores_por_grupo': pd.DataFrame(por_grupo),
              'metricas_pliegues': pd.DataFrame(pliegues)}
    for nombre, tabla in tablas.items():
        guardar_tabla(tabla, Path(salida) / f'{nombre}.csv')

    modelos = ['SVM_inicial', 'SVM_ajustado', 'Random_Forest_inicial', 'Random_Forest_ajustado']
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), layout='constrained')
    for ax, modelo in zip(axes, modelos):
        sub = predicciones.loc[predicciones.modelo.eq(modelo)]
        matriz = confusion_matrix(sub.calidad_revisada, sub.prediccion, labels=CALIDADES)
        ax.imshow(matriz / matriz.sum(axis=1, keepdims=True), vmin=0, vmax=1, cmap='Blues')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CALIDADES, yticklabels=CALIDADES,
               xlabel='Predicción', ylabel='Calidad revisada', title=modelo.replace('_', ' '))
        for i in range(3):
            for j in range(3):
                ax.text(j, i, str(matriz[i, j]), ha='center', va='center',
                        color='white' if matriz[i, j] / matriz[i].sum() > .55 else '#222222')
    fig.suptitle('Comparación con los mismos grupos externos · predicciones OOF')
    guardar_grafica(fig, Path(salida) / '01_comparacion_matrices.png')
    plt.close(fig)
    crear_galeria_errores(base, predicciones, salida)
    return tablas


def crear_galeria_errores(base, predicciones, salida):
    # Selección por identificador: sin cambiar etiquetas ni buscar casos favorables.
    sub = predicciones.loc[predicciones.modelo.eq('Random_Forest_ajustado') & ~predicciones.acierto]
    sub = sub.merge(base[['id_ejemplar', 'ruta_cnn', 'codigo_revision']], on='id_ejemplar', validate='one_to_one')
    ejemplos = []
    fig, axes = plt.subplots(3, 3, figsize=(11, 10), layout='constrained')
    for i, fruta in enumerate(FRUTAS):
        filas = sub.loc[sub.fruta.eq(fruta)].sort_values('id_ejemplar').head(3)
        for j, ax in enumerate(axes[i]):
            ax.axis('off')
            if j >= len(filas):
                ax.set_title(f'{fruta}: sin más errores para mostrar')
                continue
            fila = filas.iloc[j]
            ax.imshow(leer_rgb(fila.ruta_cnn))
            ax.set_title(f'{fruta} · {fila.codigo_revision}\nRevisada: {fila.calidad_revisada} · Predicha: {fila.prediccion}', fontsize=9)
            ejemplos.append(fila.to_dict())
    fig.suptitle('Errores de Random Forest ajustado · etiquetas conservadas')
    guardar_grafica(fig, Path(salida) / '02_ejemplos_errores.png')
    guardar_tabla(pd.DataFrame(ejemplos), Path(salida) / 'ejemplos_errores.csv')
    plt.close(fig)
