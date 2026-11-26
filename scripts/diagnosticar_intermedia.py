"""Diagnóstico reproducible de la clase intermedia y cribado agrupado de alternativas."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

PROYECTO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROYECTO))
from src.data.base import cargar_base, verificar_imagenes
from src.models.evaluacion import CALIDADES, crear_pliegues_externos, crear_cv_interno, calcular_metricas, guardar_json, guardar_grafica
from src.utils.archivos import guardar_tabla


def metricas_intermedia(real, pred):
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(real, pred, labels=['intermedia'], zero_division=0)
    return {'precision_intermedia': float(precision[0]), 'recall_intermedia': float(recall[0]), 'f1_intermedia': float(f1[0]),
            'aciertos_intermedia': int(((np.asarray(real) == 'intermedia') & (np.asarray(pred) == 'intermedia')).sum()),
            'predichas_intermedia': int((np.asarray(pred) == 'intermedia').sum())}


def especificaciones():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
    return {
        'Logistica_CNN': ('cnn', Pipeline([('escala', StandardScaler()), ('modelo', LogisticRegression(max_iter=3000, random_state=42))]),
                          {'modelo__C': [.01, .1, 1.], 'modelo__class_weight': [None, 'balanced']}),
        'SVM_CNN': ('cnn', Pipeline([('escala', StandardScaler()), ('modelo', SVC(random_state=42))]),
                    {'modelo__C': [.1, 1.], 'modelo__kernel': ['linear', 'rbf'], 'modelo__class_weight': [None, 'balanced']}),
        'Extra_Trees': ('manual', ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=1),
                        {'max_depth': [6, None], 'min_samples_leaf': [2, 5]}),
        'Gradient_Boosting': ('manual', HistGradientBoostingClassifier(max_iter=100, min_samples_leaf=5, class_weight='balanced',
                                                                     early_stopping=False, random_state=42),
                              {'max_leaf_nodes': [3, 7], 'l2_regularization': [1., 10.]}),
    }


def galeria(tabla, ruta, titulo):
    import matplotlib.pyplot as plt
    from PIL import Image
    columnas = 4
    filas = int(np.ceil(len(tabla) / columnas))
    fig, axes = plt.subplots(filas, columnas, figsize=(12, filas * 3.2), squeeze=False, layout='constrained')
    for ax, fila in zip(axes.flat, tabla.itertuples()):
        with Image.open(fila.ruta_cnn) as im:
            ax.imshow(im.convert('RGB'))
        ax.set_title(f'{fila.codigo_revision} | {fila.fruta}\npred: {fila.prediccion} | p(inter): {fila.prob_intermedia:.2f}', fontsize=9)
        ax.axis('off')
    for ax in axes.flat[len(tabla):]:
        ax.axis('off')
    fig.suptitle(titulo + '\nEtiqueta revisada: intermedia; la predicción no modifica la etiqueta', fontsize=12)
    guardar_grafica(fig, ruta)
    plt.close(fig)


def ejecutar(proyecto=PROYECTO, comparar_alternativas=False):
    from sklearn.metrics import f1_score
    proyecto = Path(proyecto).resolve()
    salida = proyecto / 'experiments/results/comparacion_modelos'
    base, columnas, estado = cargar_base(proyecto)
    verificar_imagenes(base)
    meta = json.loads((salida / 'configuracion_comparacion.json').read_text(encoding='utf-8'))
    if meta['sha_base'] != estado['sha_base']:
        raise ValueError('Los resultados no corresponden a la base actual. Ejecutar primero el cuaderno 08.')
    exteriores, asignacion, cobertura = crear_pliegues_externos(base)
    pred = pd.read_csv(salida / 'cnn_predicciones_oof.csv')
    if pred.id_ejemplar.duplicated().any() or set(pred.id_ejemplar) != set(base.id_ejemplar):
        raise ValueError('Las predicciones CNN no contienen exactamente la base.')
    pred = pred.set_index('id_ejemplar').loc[base.id_ejemplar].reset_index()
    for campo in ['grupo_particion', 'fruta', 'fuente', 'calidad_revisada']:
        if pred[campo].tolist() != base[campo].tolist():
            raise ValueError('Las predicciones difieren en ' + campo)
    if pred.pliegue_validacion.tolist() != asignacion.tolist():
        raise ValueError('Las particiones de CNN difieren de las actuales.')
    unido = pred.merge(base[['id_ejemplar', 'ruta_cnn', 'codigo_revision']], on='id_ejemplar', validate='1:1')
    inter = unido.loc[unido.calidad_revisada.eq('intermedia')].copy()
    inter['acierto'] = inter.prediccion.eq('intermedia')
    grupos = (inter.groupby(['grupo_particion', 'fruta', 'fuente', 'pliegue_validacion'], observed=True)
              .agg(imagenes=('id_ejemplar', 'size'), aciertos=('acierto', 'sum'), prob_intermedia_media=('prob_intermedia', 'mean'))
              .reset_index())
    grupos['recall_intermedia'] = grupos.aciertos / grupos.imagenes
    guardar_tabla(grupos, salida / 'diagnostico_intermedia_por_grupo.csv')
    guardar_tabla(inter.drop(columns='ruta_cnn'), salida / 'diagnostico_intermedia_errores.csv')
    guardar_tabla(cobertura, salida / 'diagnostico_cobertura.csv')
    guardar_tabla(pd.crosstab(base.fruta, base.calidad_revisada).reset_index(), salida / 'diagnostico_fruta_calidad.csv')
    dominante = grupos.sort_values('imagenes', ascending=False).iloc[0].grupo_particion
    publico = inter.loc[inter.grupo_particion.eq(dominante)].sort_values('prob_intermedia')
    muestra = publico.iloc[np.linspace(0, len(publico) - 1, min(12, len(publico)), dtype=int)]
    otras = inter.loc[~inter.grupo_particion.eq(dominante)]
    galeria(muestra, salida / 'diagnostico_intermedia_campana.png', 'Muestra de la campaña que concentra las intermedias')
    galeria(otras, salida / 'diagnostico_intermedia_otros_grupos.png', 'Todas las intermedias fuera de la campaña dominante')
    guardar_tabla(pd.concat([muestra, otras]).drop(columns='ruta_cnn'), salida / 'diagnostico_revision_visual.csv')
    auditoria = pd.read_csv(salida / 'cnn_particiones.csv')
    internos = auditoria.loc[auditoria.interior.ne(0)].groupby('exterior').interior.nunique()
    plan_cnn = json.loads((salida / 'cnn_plan_entrenamiento.json').read_text(encoding='utf-8'))
    resumen = {'sha_base': estado['sha_base'], 'imagenes': len(base), 'intermedias': len(inter),
               'intermedias_por_fruta': inter.fruta.value_counts().to_dict(), 'grupos_con_intermedia': len(grupos),
               'grupo_dominante': dominante, 'intermedias_grupo_dominante': int(grupos.imagenes.max()),
               'confusion_intermedia': inter.prediccion.value_counts().to_dict(),
               'mediana_prob_intermedia': float(inter.prob_intermedia.median()),
               'intermedias_predichas_buena_p_ge_09': int((inter.prediccion.eq('buena') & inter.prob_buena.ge(.9)).sum()),
               'pliegues_internos_por_exterior': {str(k): int(v) for k,v in internos.items()},
               'ajustes_internos_cnn': int(internos.sum() * len(plan_cnn['candidatos'])),
               'cambio_etiquetas': False, 'evaluacion_independiente': False}
    guardar_json(resumen, salida / 'diagnostico_intermedia.json')
    resultado = {'grupos_intermedia': grupos, 'cobertura': cobertura, 'resumen': resumen}
    if comparar_alternativas:
        resultado['metricas_alternativas'] = comparar(base, columnas, estado, proyecto, salida, exteriores, asignacion)
    print('Diagnóstico:', resumen, flush=True)
    return resultado


def comparar(base, columnas, estado, proyecto, salida, exteriores, asignacion):
    import torch
    import sklearn
    from sklearn.base import clone
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import f1_score
    from threadpoolctl import threadpool_limits
    cache = torch.load(salida / 'cnn_caracteristicas.pt', map_location='cpu', weights_only=True)
    if cache['firma']['sha_base'] != estado['sha_base'] or cache['ids'] != base.id_ejemplar.tolist():
        raise ValueError('El cache visual no corresponde a la base. Ejecutar el cuaderno 08.')
    if cache['firma']['pesos'] != 'MobileNet_V3_Small_Weights.IMAGENET1K_V1' or cache['firma']['vistas'][0] != 'original':
        raise ValueError('El extractor del cache no coincide con el plan.')
    x = {'cnn': cache['vectores'][:, 0].numpy(), 'manual': base[columnas].to_numpy(dtype=float)}
    if not all(np.isfinite(v).all() for v in x.values()):
        raise ValueError('Hay características no finitas.')
    y, g = base.calidad_revisada.to_numpy(), base.grupo_particion.to_numpy()
    specs = especificaciones()
    guardar_json({'sha_base': estado['sha_base'], 'semilla': 42, 'seleccion': 'F1 macro interno medio con tres etiquetas fijas',
                  'escalado': 'StandardScaler dentro de cada ajuste de logística/SVM; árboles sin escalado',
                  'extractor': cache['firma'], 'aumentos': 'Ninguno; se usa la vista original en este cribado.',
                  'espacios': {k: {'entrada': v[0], 'candidatos': list(ParameterGrid(v[2]))} for k,v in specs.items()},
                  'early_stopping_boosting': False, 'prueba_independiente': False,
                  'versiones': {'sklearn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'torch': str(torch.__version__)}},
                 salida / 'diagnostico_alternativas_plan.json')
    filas, selecciones, membresias, oof_todos = [], [], [], []
    pred_base = pd.read_csv(salida / 'predicciones_oof.csv')
    for nombre, sub in pred_base.groupby('modelo', sort=False):
        sub = sub.set_index('id_ejemplar').loc[base.id_ejemplar]
        if sub.pliegue_validacion.tolist() != asignacion.tolist() or sub.calidad_revisada.tolist() != y.tolist():
            raise ValueError('Las referencias usan otras etiquetas o bloques.')
        oof_todos.append((nombre, sub.prediccion.to_numpy()))
    with threadpool_limits(limits=4):
        for nombre, (entrada, modelo, espacio) in specs.items():
            oof = np.full(len(base), '', dtype=object)
            for exterior, (tr, va) in enumerate(exteriores, 1):
                cv = crear_cv_interno(y[tr], g[tr], semilla=42 + exterior * 100)
                interiores = list(cv.split(x[entrada][tr], y[tr], g[tr]))
                if set(g[tr]) & set(g[va]):
                    raise ValueError('Fuga externa de grupos.')
                for interior, (a, b) in enumerate(interiores, 1):
                    if set(g[tr[a]]) & set(g[tr[b]]) or set(y[tr[a]]) != set(CALIDADES):
                        raise ValueError('Fuga interna o falta de clases de entrenamiento.')
                    for etapa, ids in [('entrenamiento', tr[a]), ('validacion', tr[b])]:
                        membresias.extend({'modelo': nombre, 'exterior': exterior, 'interior': interior, 'etapa': etapa,
                                           'id_ejemplar': base.iloc[i].id_ejemplar, 'grupo': g[i]} for i in ids)
                candidatos = []
                for numero, params in enumerate(ParameterGrid(espacio), 1):
                    scores = []
                    for a, b in interiores:
                        ajuste = clone(modelo).set_params(**params).fit(x[entrada][tr[a]], y[tr[a]])
                        pred = ajuste.predict(x[entrada][tr[b]])
                        scores.append(f1_score(y[tr[b]], pred, labels=CALIDADES, average='macro', zero_division=0))
                    fila = {'modelo': nombre, 'exterior': exterior, 'candidato': numero,
                            'parametros': json.dumps(params, sort_keys=True), 'f1_macro_interno': float(np.mean(scores))}
                    filas.append(fila); candidatos.append(fila)
                mejor = max(candidatos, key=lambda f: f['f1_macro_interno'])
                ajustes = json.loads(mejor['parametros'])
                ajuste = clone(modelo).set_params(**ajustes).fit(x[entrada][tr], y[tr])
                oof[va] = ajuste.predict(x[entrada][va])
                selecciones.append(mejor)
                print(f'Alternativa {nombre}: bloque {exterior}/3 terminado.', flush=True)
            if (oof == '').any():
                raise ValueError('Predicciones externas incompletas.')
            oof_todos.append((nombre, oof))
    pesos = 1 / base.groupby('grupo_particion').id_ejemplar.transform('size').to_numpy()
    metricas, predicciones, bloques = [], [], []
    for nombre, pred in oof_todos:
        metricas.append({'modelo': nombre, **calcular_metricas(y, pred), **metricas_intermedia(y, pred),
                         'f1_macro_peso_grupo': calcular_metricas(y, pred, pesos)['f1_macro']})
        tabla = base[['id_ejemplar', 'grupo_particion', 'fruta', 'fuente', 'calidad_revisada']].copy()
        tabla['modelo'], tabla['prediccion'], tabla['pliegue_validacion'] = nombre, pred, asignacion
        predicciones.append(tabla)
        for exterior, (_, va) in enumerate(exteriores, 1):
            bloques.append({'modelo': nombre, 'exterior': exterior, **calcular_metricas(y[va], pred[va]), **metricas_intermedia(y[va], pred[va])})
    tablas = {'metricas': pd.DataFrame(metricas), 'predicciones': pd.concat(predicciones, ignore_index=True),
              'bloques': pd.DataFrame(bloques), 'busqueda': pd.DataFrame(filas),
              'seleccion': pd.DataFrame(selecciones), 'particiones': pd.DataFrame(membresias)}
    for nombre, tabla in tablas.items():
        guardar_tabla(tabla, salida / f'diagnostico_alternativas_{nombre}.csv')
    return tablas['metricas']



def verificar(proyecto=PROYECTO):
    """Recalcular métricas y reproducir las predicciones de los doce ajustes externos."""
    import torch
    from sklearn.base import clone
    from sklearn.model_selection import ParameterGrid
    from threadpoolctl import threadpool_limits
    proyecto = Path(proyecto).resolve()
    salida = proyecto / 'experiments/results/comparacion_modelos'
    base, columnas, estado = cargar_base(proyecto)
    plan = json.loads((salida / 'diagnostico_alternativas_plan.json').read_text(encoding='utf-8'))
    assert plan['sha_base'] == estado['sha_base']
    cache = torch.load(salida / 'cnn_caracteristicas.pt', map_location='cpu', weights_only=True)
    assert cache['ids'] == base.id_ejemplar.tolist() and cache['firma']['sha_base'] == estado['sha_base']
    x = {'cnn': cache['vectores'][:,0].numpy(), 'manual': base[columnas].to_numpy(dtype=float)}
    y = base.calidad_revisada.to_numpy()
    exteriores, asignacion, _ = crear_pliegues_externos(base)
    preds = pd.read_csv(salida / 'diagnostico_alternativas_predicciones.csv')
    metricas = pd.read_csv(salida / 'diagnostico_alternativas_metricas.csv').set_index('modelo')
    busqueda = pd.read_csv(salida / 'diagnostico_alternativas_busqueda.csv')
    seleccion = pd.read_csv(salida / 'diagnostico_alternativas_seleccion.csv')
    auditoria = pd.read_csv(salida / 'diagnostico_alternativas_particiones.csv')
    assert not preds[['modelo','id_ejemplar']].duplicated().any()
    pesos = 1 / base.groupby('grupo_particion').id_ejemplar.transform('size').to_numpy()
    for nombre, sub in preds.groupby('modelo'):
        assert sub.id_ejemplar.tolist() == base.id_ejemplar.tolist()
        assert sub.calidad_revisada.tolist() == y.tolist()
        assert sub.pliegue_validacion.tolist() == asignacion.tolist()
        valores = {**calcular_metricas(y, sub.prediccion), **metricas_intermedia(y, sub.prediccion),
                   'f1_macro_peso_grupo': calcular_metricas(y, sub.prediccion,pesos)['f1_macro']}
        for campo, valor in valores.items():
            assert np.isclose(valor,metricas.loc[nombre,campo])
    assert len(preds) == len(base) * len(metricas)
    anteriores = pd.read_csv(salida / 'predicciones_oof.csv')
    for nombre, sub in anteriores.groupby('modelo'):
        assert sub.prediccion.tolist() == preds.loc[preds.modelo.eq(nombre),'prediccion'].tolist()
    with threadpool_limits(limits=4):
        for nombre, (entrada,modelo,espacio) in especificaciones().items():
            for exterior, (tr,va) in enumerate(exteriores,1):
                assert not set(base.iloc[tr].grupo_particion) & set(base.iloc[va].grupo_particion)
                tabla = busqueda[(busqueda.modelo==nombre)&(busqueda.exterior==exterior)]
                assert len(tabla) == len(ParameterGrid(espacio))
                mejor = tabla.loc[tabla.f1_macro_interno.idxmax()]
                fila = seleccion[(seleccion.modelo==nombre)&(seleccion.exterior==exterior)].iloc[0]
                assert fila.candidato == mejor.candidato and fila.parametros == mejor.parametros
                bloque = auditoria[(auditoria.modelo==nombre)&(auditoria.exterior==exterior)]
                for _, dentro in bloque.groupby('interior'):
                    a = dentro[dentro.etapa=='entrenamiento']; b = dentro[dentro.etapa=='validacion']
                    assert not set(a.grupo) & set(b.grupo)
                    assert set(a.id_ejemplar) | set(b.id_ejemplar) == set(base.iloc[tr].id_ejemplar)
                    assert len(dentro) == len(tr)
                ajuste = clone(modelo).set_params(**json.loads(fila.parametros)).fit(x[entrada][tr],y[tr])
                esperado = preds[preds.modelo.eq(nombre)].iloc[va].prediccion.to_numpy()
                assert np.array_equal(ajuste.predict(x[entrada][va]),esperado)
    print(f'Verificado diagnóstico: {len(preds)} predicciones de {len(metricas)} modelos; doce ajustes externos reproducidos.')
    print('Métricas, selección interna, referencias y separación de grupos correctas.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparar-alternativas', action='store_true')
    parser.add_argument('--verificar', action='store_true')
    args = parser.parse_args()
    if args.verificar:
        verificar()
    else:
        resultado = ejecutar(comparar_alternativas=args.comparar_alternativas)
        if 'metricas_alternativas' in resultado:
            print(resultado['metricas_alternativas'].to_string(index=False))
