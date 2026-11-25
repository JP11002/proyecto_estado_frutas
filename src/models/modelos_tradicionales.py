"""EDA y comparación inicial de calidad externa con validación por grupos.

Las características ya se calcularon dentro de las máscaras revisadas. La única
transformación aprendida (StandardScaler) se ajusta dentro de cada pliegue.
Esta evaluación pertenece al desarrollo: no sustituye una prueba independiente.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.models.evaluacion import crear_cv_interno
from src.utils.archivos import guardar_bytes_si_cambio, guardar_tabla

CALIDADES = ['buena', 'intermedia', 'mala']
FRUTAS = ['manzana', 'banano', 'limon']
COLORES = ['#36856c', '#dea33c', '#ba5260']
SEMILLA = 42


def cargar_base(proyecto):
    proyecto = Path(proyecto).resolve()
    ruta = proyecto / 'data/processed/base_revisada_v2/base_svm_rf.csv'
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


def guardar_json(objeto, ruta):
    guardar_bytes_si_cambio(ruta, json.dumps(objeto, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def guardar_grafica(figura, ruta):
    contenido = io.BytesIO()
    figura.savefig(contenido, format='png', dpi=155, bbox_inches='tight', facecolor='white')
    guardar_bytes_si_cambio(ruta, contenido.getvalue())


def hacer_eda(base, columnas, salida):
    import matplotlib.pyplot as plt

    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    conteos = pd.crosstab(base.fruta, base.calidad_revisada).reindex(index=FRUTAS, columns=CALIDADES, fill_value=0)
    cobertura = base.groupby(['fruta', 'calidad_revisada']).agg(
        fotos=('id_ejemplar', 'size'), grupos=('grupo_particion', 'nunique')).reset_index()
    origen = pd.crosstab(base.fuente, base.calidad_revisada).reindex(columns=CALIDADES, fill_value=0)
    diagnostico = {
        'imagenes': len(base), 'caracteristicas': len(columnas),
        'grupos_conservadores': int(base.grupo_particion.nunique()),
        'faltantes_caracteristicas': int(base[columnas].isna().sum().sum()),
        'caracteristicas_constantes': base[columnas].columns[base[columnas].nunique() <= 1].tolist(),
        'filas_caracteristicas_duplicadas': int(base[columnas].duplicated(keep=False).sum()),
        'sha_original_repetidos': int(base.sha_original.duplicated().sum()),
        'fraccion_clase_mayoritaria': float(base.calidad_revisada.value_counts(normalize=True).max()),
        'alcance_tamano': 'Las formas y tamaños son relativos a la imagen, sin escala en centímetros.',
    }
    guardar_json(diagnostico, salida / 'diagnostico_eda.json')
    guardar_tabla(conteos.reset_index(), salida / 'conteos_fruta_calidad.csv')
    guardar_tabla(cobertura, salida / 'cobertura_grupos.csv')
    guardar_tabla(origen.reset_index(), salida / 'conteos_fuente_calidad.csv')
    guardar_tabla(base[columnas].describe().T.reset_index(names='caracteristica'), salida / 'estadisticos_caracteristicas.csv')

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    conteos.plot.bar(ax=axes[0, 0], color=COLORES, rot=0)
    axes[0, 0].set(title='Imágenes por fruta y calidad', xlabel='', ylabel='Imágenes')
    base.calidad_revisada.value_counts().reindex(CALIDADES).plot.bar(ax=axes[0, 1], color=COLORES, rot=0)
    axes[0, 1].set(title='Desbalance de calidad', xlabel='', ylabel='Imágenes')
    for i, n in enumerate(base.calidad_revisada.value_counts().reindex(CALIDADES)):
        axes[0, 1].text(i, n + 3, f'{n} ({n / len(base):.1%})', ha='center', fontsize=9)
    axes[0, 1].set_ylim(0, 230)
    origen.plot.bar(ax=axes[1, 0], color=COLORES, rot=0)
    axes[1, 0].set(title='Fuente y calidad', xlabel='', ylabel='Imágenes')
    cobertura.pivot(index='fruta', columns='calidad_revisada', values='grupos').reindex(
        index=FRUTAS, columns=CALIDADES).plot.bar(ax=axes[1, 1], color=COLORES, rot=0)
    axes[1, 1].set(title='Grupos presentes por fruta y calidad', xlabel='', ylabel='Grupos (pueden compartir calidades)')
    guardar_grafica(fig, salida / '01_distribucion_base.png')
    plt.close(fig)

    variables = ['color_s_media', 'color_v_media', 'textura_entropia', 'forma_circularidad']
    fig, axes = plt.subplots(3, 4, figsize=(13, 9), layout='constrained')
    for i, fruta in enumerate(FRUTAS):
        for j, variable in enumerate(variables):
            ax = axes[i, j]
            muestras = [base.loc[(base.fruta == fruta) & (base.calidad_revisada == c), variable].to_numpy() for c in CALIDADES]
            cajas = ax.boxplot(muestras, tick_labels=['B', 'I', 'M'], patch_artist=True)
            for caja, color in zip(cajas['boxes'], COLORES):
                caja.set_facecolor(color)
                caja.set_alpha(.6)
            ax.set_title(f'{fruta} · {variable.replace("color_", "").replace("textura_", "").replace("forma_", "")}', fontsize=9)
    fig.suptitle('Distribuciones dentro de cada fruta · B: buena, I: intermedia, M: mala')
    guardar_grafica(fig, salida / '02_distribuciones_caracteristicas.png')
    plt.close(fig)

    seleccion = ['color_r_media', 'color_g_media', 'color_b_media', 'color_s_media', 'color_v_media',
                 'textura_contraste', 'textura_homogeneidad', 'textura_energia', 'textura_entropia',
                 'forma_circularidad', 'forma_solidez', 'forma_aspecto']
    correlacion = base[seleccion].corr()
    guardar_tabla(correlacion.reset_index(names='caracteristica'), salida / 'correlaciones_seleccionadas.csv')
    fig, ax = plt.subplots(figsize=(10, 9), layout='constrained')
    im = ax.imshow(correlacion, vmin=-1, vmax=1, cmap='RdBu_r')
    etiquetas = [c.replace('color_', '').replace('textura_', '').replace('forma_', '') for c in seleccion]
    ax.set_xticks(range(len(seleccion)), etiquetas, rotation=60, ha='right')
    ax.set_yticks(range(len(seleccion)), etiquetas)
    ax.set_title('Correlación de características seleccionadas · EDA descriptivo')
    fig.colorbar(im, ax=ax, shrink=.8, label='Correlación de Pearson')
    guardar_grafica(fig, salida / '03_correlaciones.png')
    plt.close(fig)

    from src.data.preparacion_imagenes import leer_rgb

    fig, axes = plt.subplots(3, 3, figsize=(10, 10), layout='constrained')
    ejemplos = []
    for i, fruta in enumerate(FRUTAS):
        for j, calidad in enumerate(CALIDADES):
            fila = base.loc[(base.fruta == fruta) & (base.calidad_revisada == calidad)].sort_values('id_ejemplar').iloc[0]
            axes[i, j].imshow(leer_rgb(fila.ruta_cnn))
            axes[i, j].set_title(f'{fruta} · {calidad}\n{fila.codigo_revision}', fontsize=10)
            axes[i, j].axis('off')
            ejemplos.append(fila[['id_ejemplar', 'codigo_revision', 'fruta', 'calidad_revisada', 'ruta_cnn']].to_dict())
    fig.suptitle('Una imagen preparada por combinación · selección fija por identificador')
    guardar_grafica(fig, salida / '04_ejemplos_base.png')
    guardar_tabla(pd.DataFrame(ejemplos), salida / 'ejemplos_galeria.csv')
    plt.close(fig)
    return diagnostico, conteos, cobertura, origen


def crear_pliegues(base):
    """Partición fija por etiquetas/grupos, nunca elegida mediante resultados."""
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEMILLA)
    particiones = list(cv.split(base, base.calidad_revisada, base.grupo_particion))
    asignacion = np.full(len(base), -1, dtype=int)
    cobertura = []
    for pliegue, (entrenamiento, validacion) in enumerate(particiones, 1):
        grupos_train = set(base.iloc[entrenamiento].grupo_particion)
        grupos_val = set(base.iloc[validacion].grupo_particion)
        assert not grupos_train & grupos_val
        for indices, etapa in [(entrenamiento, 'entrenamiento'), (validacion, 'validacion')]:
            tabla = base.iloc[indices]
            if set(tabla.calidad_revisada) != set(CALIDADES):
                raise ValueError(f'El pliegue {pliegue} no tiene las tres calidades en {etapa}.')
            for fruta in FRUTAS:
                for calidad in CALIDADES:
                    sub = tabla.loc[(tabla.fruta == fruta) & (tabla.calidad_revisada == calidad)]
                    cobertura.append({'pliegue': pliegue, 'etapa': etapa, 'fruta': fruta,
                                      'calidad': calidad, 'imagenes': len(sub),
                                      'grupos': sub.grupo_particion.nunique()})
        assert (asignacion[validacion] == -1).all()
        asignacion[validacion] = pliegue
    assert (asignacion > 0).all()
    return particiones, asignacion, pd.DataFrame(cobertura)


def crear_modelos():
    return {
        'Referencia_mayoritaria': DummyClassifier(strategy='most_frequent'),
        'SVM': make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='scale', class_weight='balanced')),
        'Random_Forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                               class_weight='balanced', random_state=SEMILLA, n_jobs=1),
    }


def calcular_metricas(real, predicha, pesos=None):
    # Recall macro con etiquetas fijas equivale a exactitud balanceada si están todas presentes.
    return {
        'exactitud': float(accuracy_score(real, predicha, sample_weight=pesos)),
        'f1_macro': float(f1_score(real, predicha, labels=CALIDADES, average='macro', zero_division=0, sample_weight=pesos)),
        'recall_macro': float(recall_score(real, predicha, labels=CALIDADES, average='macro', zero_division=0, sample_weight=pesos)),
    }


def entrenar_y_comparar(base, columnas, estado, salida, checkpoints):
    salida, checkpoints = Path(salida), Path(checkpoints)
    salida.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    X, y = base[columnas], base.calidad_revisada
    particiones, asignacion, cobertura = crear_pliegues(base)
    guardar_tabla(cobertura, salida / 'cobertura_pliegues.csv')
    asignaciones = base[['id_ejemplar', 'grupo_particion', 'fruta', 'calidad_revisada', 'fuente']].copy()
    asignaciones['pliegue_validacion'] = asignacion
    guardar_tabla(asignaciones, salida / 'asignacion_pliegues.csv')
    resultados, por_pliegue, por_fruta, por_fuente, por_clase, predicciones = [], [], [], [], [], []
    pesos = 1 / base.groupby('grupo_particion').id_ejemplar.transform('size').to_numpy()
    modelos = crear_modelos()
    matrices = {}
    for nombre, prototipo in modelos.items():
        oof = np.full(len(base), '', dtype=object)
        for pliegue, (train, val) in enumerate(particiones, 1):
            modelo = clone(prototipo).fit(X.iloc[train], y.iloc[train])
            oof[val] = modelo.predict(X.iloc[val])
            por_pliegue.append({'modelo': nombre, 'pliegue': pliegue,
                                'n_train': len(train), 'n_validacion': len(val),
                                **calcular_metricas(y.iloc[val], oof[val])})
        assert (oof != '').all()
        resultados.append({'modelo': nombre, 'imagenes': len(base),
                           **calcular_metricas(y, oof),
                           **{f'{k}_peso_grupo': v for k, v in calcular_metricas(y, oof, pesos).items()}})
        tabla = asignaciones.copy()
        tabla['modelo'], tabla['prediccion'] = nombre, oof
        tabla['acierto'] = tabla.calidad_revisada == tabla.prediccion
        predicciones.append(tabla)
        for campo, destino in [('fruta', por_fruta), ('fuente', por_fuente)]:
            for categoria, sub in tabla.groupby(campo):
                destino.append({'modelo': nombre, campo: categoria, 'imagenes': len(sub),
                                **calcular_metricas(sub.calidad_revisada, sub.prediccion)})
        matriz = confusion_matrix(y, oof, labels=CALIDADES)
        matrices[nombre] = matriz
        guardar_tabla(pd.DataFrame(matriz, columns=CALIDADES).assign(real=CALIDADES)[['real', *CALIDADES]],
                      salida / f'matriz_confusion_{nombre}.csv')
        for clase in CALIDADES:
            por_clase.append({'modelo': nombre, 'calidad': clase, 'imagenes': int((y == clase).sum()),
                              'recall': float(recall_score(y, oof, labels=[clase], average='macro', zero_division=0)),
                              'f1': float(f1_score(y, oof, labels=[clase], average='macro', zero_division=0))})
        if nombre != 'Referencia_mayoritaria':
            # Prototipo para inferencia; las métricas anteriores proceden solo de OOF.
            ajustado = clone(prototipo).fit(X, y)
            paquete = {'modelo': ajustado, 'caracteristicas': columnas,
                       'calidades': CALIDADES, 'frutas': FRUTAS, 'sha_base': estado['sha_base'],
                       'alcance': 'Prototipo de desarrollo: calidad externa visible.'}
            contenido = io.BytesIO()
            joblib.dump(paquete, contenido, compress=3)
            ruta = checkpoints / f'{nombre.lower()}.joblib'
            guardar_bytes_si_cambio(ruta, contenido.getvalue())
            cargado = joblib.load(ruta)
            assert np.array_equal(cargado['modelo'].predict(X), ajustado.predict(X))

    tablas = {'metricas_globales': pd.DataFrame(resultados), 'metricas_pliegues': pd.DataFrame(por_pliegue),
              'metricas_por_fruta': pd.DataFrame(por_fruta), 'metricas_por_fuente': pd.DataFrame(por_fuente),
              'metricas_por_clase': pd.DataFrame(por_clase), 'predicciones_oof': pd.concat(predicciones, ignore_index=True)}
    for nombre, tabla in tablas.items():
        guardar_tabla(tabla, salida / f'{nombre}.csv')

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, (nombre, matriz) in zip(axes, matrices.items()):
        ax.imshow(matriz / matriz.sum(axis=1, keepdims=True), vmin=0, vmax=1, cmap='Blues')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CALIDADES, yticklabels=CALIDADES,
               xlabel='Predicción', ylabel='Calidad revisada', title=nombre.replace('_', ' '))
        for i in range(3):
            for j in range(3):
                proporcion = matriz[i, j] / matriz[i].sum()
                ax.text(j, i, f'{matriz[i, j]}\n{proporcion:.0%}', ha='center', va='center',
                        color='white' if proporcion > .55 else '#222222')
    fig.suptitle('Predicciones fuera del entrenamiento · 3 pliegues agrupados')
    guardar_grafica(fig, salida / '05_matrices_confusion.png')
    plt.close(fig)

    metadata = {
        'fecha_utc': datetime.now(timezone.utc).isoformat(), 'sha_base': estado['sha_base'],
        'imagenes': len(base), 'grupos': int(base.grupo_particion.nunique()),
        'caracteristicas': columnas, 'objetivo': 'calidad_revisada', 'calidades': CALIDADES, 'frutas': FRUTAS,
        'validacion': 'StratifiedGroupKFold, 3 pliegues, shuffle=True, semilla=42',
        'seleccion_pliegues': 'Fija antes de entrenar; no se seleccionó por puntuación.',
        'metricas': 'OOF: cada imagen predicha una vez por un modelo que no vio su grupo.',
        'modelo_conjunto': 'Un clasificador de calidad para las tres frutas; fruta no es una variable de entrada.',
        'parametros': {k: str(v) for k, v in modelos.items()},
        'versiones': {'python': platform.python_version(), 'scikit_learn': sklearn.__version__,
                     'numpy': np.__version__, 'pandas': pd.__version__, 'joblib': joblib.__version__},
        'evaluacion_independiente': False,
        'limitacion': 'Imágenes ya exploradas al preparar máscaras/etiquetas; pliegues desbalanceados y varias celdas fruta-calidad sin cobertura. Métricas exploratorias, sin prueba final ni ajuste de hiperparámetros.',
        'checkpoints': {n: str(checkpoints / f'{n.lower()}.joblib') for n in modelos if n != 'Referencia_mayoritaria'},
    }
    guardar_json(metadata, salida / 'configuracion_entrenamiento.json')
    guardar_json(metadata, checkpoints / 'metadata.json')
    return tablas


def espacios_hiperparametros(semilla: int = SEMILLA) -> dict[str, tuple[object, list[dict] | dict]]:
    """Definir estimadores y rejillas para búsqueda interna agrupada."""
    return {
        'SVM_ajustado': (
            Pipeline([
                ('escalar', StandardScaler()),
                ('svc', SVC(probability=True, class_weight='balanced', random_state=semilla)),
            ]),
            [
                {
                    'svc__kernel': ['rbf'],
                    'svc__C': [0.1, 1.0, 5.0, 10.0],
                    'svc__gamma': ['scale', 0.01, 0.1],
                },
                {
                    'svc__kernel': ['linear'],
                    'svc__C': [0.1, 1.0, 5.0, 10.0],
                },
            ],
        ),
        'Random_Forest_ajustado': (
            RandomForestClassifier(class_weight='balanced', random_state=semilla, n_jobs=1),
            {
                'n_estimators': [200, 400],
                'max_depth': [None, 8],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 0.3],
            },
        ),
    }


def evaluar_tradicionales_anidados(
    base: pd.DataFrame,
    columnas: list[str],
    particiones: list[tuple[np.ndarray, np.ndarray]],
    estado: dict,
    salida: str | Path,
    checkpoints: str | Path,
    semilla: int = SEMILLA,
    espacios: dict[str, tuple[object, list[dict] | dict]] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """Ejecutar validación cruzada anidada por grupos y guardar modelos ajustados."""
    salida = Path(salida)
    checkpoints = Path(checkpoints)
    salida.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)

    X = base[columnas]
    y = base.calidad_revisada
    grupos = base.grupo_particion
    scorer = make_scorer(f1_score, labels=CALIDADES, average='macro', zero_division=0, pos_label=None)

    predicciones_oof: dict[str, np.ndarray] = {}
    resumen_ajuste: dict[str, dict] = {}

    # Referencia mayoritaria sin fuga
    oof_ref = np.full(len(base), '', dtype=object)
    for train, val in particiones:
        oof_ref[val] = DummyClassifier(strategy='most_frequent').fit(X.iloc[train], y.iloc[train]).predict(X.iloc[val])
    predicciones_oof['Referencia_mayoritaria'] = oof_ref

    registros_busqueda = []
    catalogo_espacios = espacios or espacios_hiperparametros(semilla)
    for nombre, (estimador, rejilla) in catalogo_espacios.items():
        oof = np.full(len(base), '', dtype=object)
        mejores_por_pliegue = []

        for pliegue, (train, val) in enumerate(particiones, 1):
            cv_interno = crear_cv_interno(y.iloc[train], grupos.iloc[train], max_splits=3, semilla=semilla + pliegue)
            busqueda = GridSearchCV(
                estimator=clone(estimador),
                param_grid=rejilla,
                scoring=scorer,
                cv=cv_interno,
                refit=True,
                n_jobs=1,
            )
            busqueda.fit(X.iloc[train], y.iloc[train], groups=grupos.iloc[train])
            oof[val] = busqueda.predict(X.iloc[val])
            mejores_por_pliegue.append({
                'pliegue_externo': pliegue,
                'f1_macro_interno': float(busqueda.best_score_),
                'parametros': busqueda.best_params_,
            })

        predicciones_oof[nombre] = oof

        # Ajuste final sobre toda la base de desarrollo con CV interno por grupos.
        cv_global = crear_cv_interno(y, grupos, max_splits=3, semilla=semilla)
        busqueda_final = GridSearchCV(
            estimator=clone(estimador),
            param_grid=rejilla,
            scoring=scorer,
            cv=cv_global,
            refit=True,
            n_jobs=1,
        )
        busqueda_final.fit(X, y, groups=grupos)

        cv_df = pd.DataFrame(busqueda_final.cv_results_).sort_values('rank_test_score')
        cv_df.insert(0, 'modelo', nombre)
        registros_busqueda.append(cv_df)

        paquete = {
            'modelo': busqueda_final.best_estimator_,
            'mejores_parametros': busqueda_final.best_params_,
            'f1_macro_cv_interno': float(busqueda_final.best_score_),
            'caracteristicas': columnas,
            'calidades': CALIDADES,
            'frutas': FRUTAS,
            'sha_base': estado['sha_base'],
            'alcance': 'Prototipo ajustado mediante GridSearchCV agrupado sobre la base de desarrollo.',
        }
        contenido = io.BytesIO()
        joblib.dump(paquete, contenido, compress=3)
        ruta_ckpt = checkpoints / f'{nombre.lower()}.joblib'
        guardar_bytes_si_cambio(ruta_ckpt, contenido.getvalue())
        cargado = joblib.load(ruta_ckpt)
        assert np.array_equal(cargado['modelo'].predict(X), busqueda_final.best_estimator_.predict(X))

        resumen_ajuste[nombre] = {
            'mejores_por_pliegue': mejores_por_pliegue,
            'mejores_parametros_global': busqueda_final.best_params_,
            'f1_macro_cv_interno_global': float(busqueda_final.best_score_),
            'checkpoint': str(ruta_ckpt),
        }

    guardar_tabla(pd.concat(registros_busqueda, ignore_index=True), salida / 'busqueda_hiperparametros_ml.csv')
    guardar_json(resumen_ajuste, salida / 'resumen_ajuste_ml.json')
    return predicciones_oof, resumen_ajuste
