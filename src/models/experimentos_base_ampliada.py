"""Experimentos v2 sobre la base ampliada: desarrollo/prueba por grupos y comparación justificada.

Protocolo (cada decisión se justifica en DECISIONES y se guarda en decisiones.json):

1. Prueba independiente por grupos, congelada antes de entrenar ningún modelo.
2. Desarrollo: validación cruzada externa agrupada (5 bloques) con ajuste de
   hiperparámetros anidado (validación interna agrupada). Las predicciones externas
   (OOF) del desarrollo sirven para elegir el modelo final.
3. Dos condiciones de datos con las mismas filas de evaluación:
   - ``anterior``: se entrena sin el lote drive_20261126 (datos previos);
   - ``ampliada``: se entrena con todos los datos de desarrollo.
   Así se aísla el efecto de los datos nuevos del efecto del modelo.
4. La prueba se predice una sola vez por modelo, después de fijar la selección.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, make_scorer,
                             precision_recall_fscore_support, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data.base import cargar_base
from src.models.evaluacion import crear_cv_interno
from src.utils.archivos import guardar_bytes_si_cambio, guardar_tabla

CALIDADES = ['buena', 'intermedia', 'mala']
SEMILLA = 42
FUENTE_LOTE = 'propia_lote_20261126'
UMBRAL_CAMPANA = 0.10
PESOS_DECISION = np.geomspace(0.25, 8, 21)

DECISIONES = {
    'particion_prueba': (
        'Se prueban StratifiedGroupKFold con 4, 5 y 6 bloques (semilla 42) y se elige como prueba el bloque '
        'que (a) contiene las tres calidades, (b) no incluye ninguna campaña con más del 10 % de la base y '
        '(c) tiene más grupos independientes; desempate por distribución de clases más cercana a la global. '
        'Por qué: con grupos, la unidad independiente es el grupo, no la foto. Una campaña de 70-103 fotos en '
        'prueba mediría un solo grupo y retiraría del entrenamiento casi toda una combinación fruta-calidad. '
        'La regla usa solo etiquetas y grupos; se fija antes de entrenar y no depende de ningún resultado.'),
    'validacion_desarrollo': (
        'StratifiedGroupKFold de 5 bloques sobre desarrollo con búsqueda interna agrupada (hasta 3 bloques). '
        'Por qué: los grupos impiden que vistas del mismo fruto o campaña queden a ambos lados; la anidación '
        'evita elegir hiperparámetros con los mismos datos que se usan para estimar el desempeño.'),
    'metrica_seleccion': (
        'F1 macro de las predicciones externas de desarrollo; desempate por recall macro. '
        'Por qué: con 63 % de buenas, la exactitud premia predecir la mayoritaria; F1 macro da el mismo peso '
        'a buena, intermedia y mala y penaliza tanto falsos negativos (recall) como falsas alarmas.'),
    'balanceo': (
        'Pesos de clase inversamente proporcionales a la frecuencia (class_weight="balanced" o pesos en la '
        'entropía cruzada) en lugar de sobremuestreo. Por qué: duplicar o sintetizar imágenes de pocas campañas '
        'no aporta variabilidad independiente y, con grupos, el sobremuestreo puede inflar la validación.'),
    'ajuste_decision': (
        'Variante "+umbral": se multiplican las probabilidades por pesos (1, a, b) elegidos en las predicciones '
        'internas del entrenamiento para maximizar F1 macro; a, b en una rejilla logarítmica de 0,25 a 8. '
        'Por qué: es la regla de Bayes con costes por clase; corrige el sesgo hacia la clase mayoritaria y '
        'actúa directamente sobre el recall de intermedia y mala. Se ajusta solo con datos de entrenamiento.'),
    'caracteristicas_cnn': (
        'MobileNetV3 Small preentrenada en ImageNet y congelada produce 576 rasgos por imagen. Por qué: con '
        '~300 imágenes no es viable aprender filtros profundos; la red congelada no ve etiquetas (sin fuga) '
        'y aporta texturas y bordes que las 43 características manuales no capturan.'),
    'modelo_profundo': (
        'Cabezal propio (lineal o MLP de 64 unidades) sobre la red congelada, entrenado con AdamW, pesos de '
        'clase y aumentos (espejo, giros de 10 grados, brillo) solo en entrenamiento. Cumple el requisito de '
        'transferencia con capas congeladas y capas propias.'),
    'modelos_nuevos': (
        'Regresión logística: referencia lineal regularizada, estable con pocos datos y probabilidades útiles '
        'para el ajuste de decisión. HistGradientBoosting: interacciones no lineales entre color y textura '
        'con regularización L2; sin parada temprana aleatoria para no partir grupos al azar. '
        'Fusión: concatena rasgos manuales (forma y color dentro de la máscara) y rasgos CNN (textura global).'),
    'comparacion_antes_despues': (
        'Las cifras del informe anterior usaban otro protocolo (3 bloques sobre 291 imágenes) y no son '
        'comparables. La comparación válida reentrena las configuraciones anteriores con este protocolo en '
        'las condiciones "anterior" y "ampliada" y las evalúa en las mismas filas.'),
    'pesos_fruta_calidad': (
        'Variante "_pesos_fruta": cada imagen pesa N / (C·n_celda), donde n_celda es el número de imágenes de su '
        'combinación fruta-calidad en el entrenamiento y C el número de combinaciones presentes. Por qué: el '
        'diagnóstico mostró que la CNN aprendió la fruta como atajo (77 de 101 bananos son intermedios; casi '
        'todos los limones y manzanas, buenos). Con pesos por clase el modelo aún gana prediciendo "banano = '
        'intermedia"; con pesos por combinación, cada celda fruta-calidad pesa igual y ese atajo deja de rendir.'),
    'intervalos': (
        'Intervalos al 95 % por bootstrap de grupos (2 000 remuestreos) en la prueba. Por qué: la prueba tiene '
        'pocos grupos independientes; remuestrear fotos sueltas subestimaría la incertidumbre.'),
}


# --------------------------------------------------------------------------- partición
def particion_desarrollo_prueba(base):
    """Regla fija para elegir la prueba independiente (ver DECISIONES['particion_prueba'])."""
    y, g = base.calidad_revisada, base.grupo_particion
    fraccion = g.map(g.value_counts()) / len(base)
    global_ = y.value_counts(normalize=True).reindex(CALIDADES)
    candidatos = []
    for n in (4, 5, 6):
        cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEMILLA)
        for k, (tr, te) in enumerate(cv.split(base, y, g)):
            prueba = base.iloc[te]
            dist = prueba.calidad_revisada.value_counts(normalize=True).reindex(CALIDADES, fill_value=0)
            candidatos.append({
                'n_bloques': n, 'bloque': k, 'imagenes': len(te), 'grupos': prueba.grupo_particion.nunique(),
                'tres_clases': set(prueba.calidad_revisada) == set(CALIDADES)
                               and set(base.iloc[tr].calidad_revisada) == set(CALIDADES),
                'sin_campana_grande': bool((fraccion.iloc[te] <= UMBRAL_CAMPANA).all()),
                'distancia_l1': float((dist - global_).abs().sum()),
                **{f'n_{c}': int((prueba.calidad_revisada == c).sum()) for c in CALIDADES}})
    tabla = pd.DataFrame(candidatos)
    validos = tabla[tabla.tres_clases & tabla.sin_campana_grande]
    if validos.empty:
        raise ValueError('Ningún bloque cumple la regla de prueba.')
    elegido = validos.sort_values(['grupos', 'distancia_l1', 'imagenes'], ascending=[False, True, False]).iloc[0]
    cv = StratifiedGroupKFold(n_splits=int(elegido.n_bloques), shuffle=True, random_state=SEMILLA)
    te = list(cv.split(base, y, g))[int(elegido.bloque)][1]
    etapa = np.array(['desarrollo'] * len(base), dtype=object)
    etapa[te] = 'prueba'
    return etapa, tabla, elegido.to_dict()


def cargar_o_crear_particion(proyecto, base, estado):
    """Congelar la partición en data/base/; si la base cambia, exigir regenerarla explícitamente."""
    ruta = Path(proyecto) / 'data/base/particion_desarrollo_prueba.csv'
    etapa, tabla, elegido = particion_desarrollo_prueba(base)
    if ruta.is_file():
        guardada = pd.read_csv(ruta)
        if guardada.sha_base.iloc[0] != estado['sha_base'] or guardada.id_ejemplar.tolist() != base.id_ejemplar.tolist():
            raise ValueError('La base cambió desde que se congeló la partición. Borrar el archivo para regenerarla '
                             'y documentar el cambio.')
        if not (guardada.etapa.to_numpy() == etapa).all():
            raise AssertionError('La regla de partición no reproduce la partición congelada.')
    else:
        guardar_tabla(pd.DataFrame({'id_ejemplar': base.id_ejemplar, 'grupo_particion': base.grupo_particion,
                                    'etapa': etapa, 'sha_base': estado['sha_base']}), ruta)
    dev_g = set(base.grupo_particion[etapa == 'desarrollo'])
    test_g = set(base.grupo_particion[etapa == 'prueba'])
    if dev_g & test_g:
        raise AssertionError('Grupos compartidos entre desarrollo y prueba.')
    return etapa, tabla, elegido


# --------------------------------------------------------------------------- métricas
def metricas(real, pred):
    p, r, f, s = precision_recall_fscore_support(real, pred, labels=CALIDADES, zero_division=0)
    out = {'exactitud': accuracy_score(real, pred),
           'precision_macro': float(p.mean()), 'recall_macro': float(r.mean()),
           'f1_macro': f1_score(real, pred, labels=CALIDADES, average='macro', zero_division=0),
           'f1_ponderado': f1_score(real, pred, labels=CALIDADES, average='weighted', zero_division=0)}
    for i, c in enumerate(CALIDADES):
        out.update({f'precision_{c}': p[i], f'recall_{c}': r[i], f'f1_{c}': f[i], f'soporte_{c}': int(s[i])})
    return {k: float(v) for k, v in out.items()}


def bootstrap_grupos(real, pred, grupos, n=2000, semilla=SEMILLA):
    """IC95 % remuestreando grupos completos. Las métricas se calculan desde matrices de
    confusión por grupo (suma de matrices = matriz del remuestreo), equivalente a sklearn."""
    rng = np.random.default_rng(semilla)
    real, pred, grupos = map(np.asarray, (real, pred, grupos))
    unicos = np.unique(grupos)
    ri = np.array([CALIDADES.index(v) for v in real])
    pi = np.array([CALIDADES.index(v) for v in pred])
    cms = np.stack([np.bincount(3 * ri[grupos == u] + pi[grupos == u], minlength=9).reshape(3, 3) for u in unicos])
    valores = {k: [] for k in ['f1_macro', 'recall_macro', 'recall_intermedia', 'recall_mala']}
    for _ in range(n):
        cm = cms[rng.integers(0, len(unicos), len(unicos))].sum(0)
        tp = np.diag(cm).astype(float)
        soporte, predichos = cm.sum(1), cm.sum(0)
        recall = np.divide(tp, soporte, out=np.zeros(3), where=soporte > 0)
        precision = np.divide(tp, predichos, out=np.zeros(3), where=predichos > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(3), where=(precision + recall) > 0)
        valores['f1_macro'].append(f1.mean())
        valores['recall_macro'].append(recall.mean())
        valores['recall_intermedia'].append(recall[1])
        valores['recall_mala'].append(recall[2])
    return {f'{k}_ic95': [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in valores.items()}


def ajustar_pesos_decision(prob, y):
    """Pesos (1, a, b) que maximizan F1 macro en predicciones internas; desempate: pesos más cercanos a 1."""
    mejor = (-1, 0, (1., 1., 1.))
    for a in PESOS_DECISION:
        for b in PESOS_DECISION:
            w = np.array([1., a, b])
            pred = np.array(CALIDADES)[(prob * w).argmax(1)]
            f1 = f1_score(y, pred, labels=CALIDADES, average='macro', zero_division=0)
            cercania = -abs(np.log(a)) - abs(np.log(b))
            if (f1, cercania) > mejor[:2]:
                mejor = (f1, cercania, (1., float(a), float(b)))
    return np.array(mejor[2]), mejor[0]


# --------------------------------------------------------------------------- modelos
def modelos_sklearn():
    """Nombre -> (conjunto de rasgos, estimador, rejilla). Las rejillas se fijan antes de ver resultados."""
    from src.models.modelos_tradicionales import espacios_hiperparametros
    previos = espacios_hiperparametros(SEMILLA)
    escalar_lr = lambda: Pipeline([('escalar', StandardScaler()),
                                   ('lr', LogisticRegression(max_iter=5000, class_weight='balanced'))])
    return {
        'Referencia_mayoritaria': ('manual', DummyClassifier(strategy='prior'), {}),
        'RF_inicial': ('manual', RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight='balanced',
                                                        random_state=SEMILLA, n_jobs=1), {}),
        'SVM_ajustado': ('manual', previos['SVM_ajustado'][0], previos['SVM_ajustado'][1]),
        'RF_ajustado': ('manual', previos['Random_Forest_ajustado'][0], previos['Random_Forest_ajustado'][1]),
        'Logistica': ('manual', escalar_lr(), {'lr__C': [0.01, 0.1, 1., 10.]}),
        'HistGradientBoosting': ('manual', HistGradientBoostingClassifier(
            max_iter=200, early_stopping=False, class_weight='balanced', random_state=SEMILLA),
            {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [3, 7, 15], 'l2_regularization': [1., 10.],
             'min_samples_leaf': [5, 10]}),
        'Logistica_CNN': ('cnn', escalar_lr(), {'lr__C': [0.001, 0.01, 0.1, 1.]}),
        'Logistica_fusion': ('fusion', escalar_lr(), {'lr__C': [0.001, 0.01, 0.1, 1.]}),
    }


def modelos_pesos_fruta():
    """Mismos estimadores y rejillas, con pesos por combinación fruta-calidad en lugar de por clase."""
    lr = lambda: Pipeline([('escalar', StandardScaler()), ('lr', LogisticRegression(max_iter=5000))])
    return {
        'HistGB_pesos_fruta': ('manual', HistGradientBoostingClassifier(
            max_iter=200, early_stopping=False, random_state=SEMILLA),
            {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [3, 7, 15], 'l2_regularization': [1., 10.],
             'min_samples_leaf': [5, 10]}, 'sample_weight'),
        'Logistica_CNN_pesos_fruta': ('cnn', lr(), {'lr__C': [0.001, 0.01, 0.1, 1.]}, 'lr__sample_weight'),
        'Logistica_fusion_pesos_fruta': ('fusion', lr(), {'lr__C': [0.001, 0.01, 0.1, 1.]}, 'lr__sample_weight'),
    }


def pesos_fruta_calidad(fruta, y):
    """w_i = N / (C · n_celda(i)), calculado solo con las filas de entrenamiento recibidas."""
    celda = pd.Series(np.asarray(fruta).astype(str)) + '|' + pd.Series(np.asarray(y).astype(str))
    conteo = celda.map(celda.value_counts()).to_numpy(float)
    return len(celda) / (celda.nunique() * conteo)


ESCORER = make_scorer(f1_score, labels=CALIDADES, average='macro', zero_division=0)


def ajustar_sklearn(estimador, rejilla, X, y, g, semilla, fruta=None, nombre_peso=None):
    """Búsqueda interna agrupada + predicciones internas (para el ajuste de decisión)."""
    cv = crear_cv_interno(y, g, max_splits=3, semilla=semilla)
    rejilla = rejilla or {}
    parametros = {nombre_peso: pesos_fruta_calidad(fruta, y)} if nombre_peso else {}
    busqueda = GridSearchCV(clone(estimador), rejilla or [{}], scoring=ESCORER, cv=cv, n_jobs=-1, refit=True)
    busqueda.fit(X, y, groups=g, **parametros)
    prob_int = cross_val_predict(clone(estimador).set_params(**busqueda.best_params_), X, y, groups=g, cv=cv,
                                 method='predict_proba', n_jobs=-1, params=parametros)
    return busqueda.best_estimator_, busqueda.best_params_, float(busqueda.best_score_), prob_int


def ajustar_cabezal_cnn(z, y_idx, g, semilla):
    """Selección interna de la configuración del cabezal (mismo criterio que el cuaderno 08)."""
    from src.models.cnn_calidad import candidatos_transferencia, entrenar_cabezal
    cv = crear_cv_interno(y_idx, g, max_splits=3, semilla=semilla)
    internos = list(cv.split(np.zeros(len(y_idx)), y_idx, g))
    mejor = None
    for numero, cfg in enumerate(candidatos_transferencia(max_epocas=120)):
        f1s, epocas, prob_int = [], [], np.zeros((len(y_idx), 3))
        for i, (a, b) in enumerate(internos, 1):
            cabeza, _, info = entrenar_cabezal(z[a], y_idx[a], cfg, semilla + i, z[b], y_idx[b])
            cabeza.eval()
            with torch.no_grad():
                prob_int[b] = cabeza(z[b][:, 0]).softmax(1).numpy()
            f1s.append(info['mejor_f1_val'])
            epocas.append(info['mejor_epoca'])
        clave = (float(np.mean(f1s)), -numero)
        if mejor is None or clave > mejor[0]:
            elegido = replace(cfg, max_epocas=max(cfg.min_epocas, int(round(np.mean(epocas)))))
            mejor = (clave, elegido, prob_int)
    (f1_int, _), cfg, prob_int = mejor
    cabeza, _, _ = entrenar_cabezal(z, y_idx, cfg, semilla)
    cabeza.eval()
    return cabeza, asdict(cfg), f1_int, prob_int


def rasgos_cnn(proyecto, base, estado, salida):
    from src.models.cnn_calidad import CNNTransferencia, ConfigTransferencia, extraer_vistas_fijas
    torch.hub.set_dir(str(Path(proyecto) / 'data/external/torch'))
    red = CNNTransferencia(ConfigTransferencia(), pesos_preentrenados=True).eval()
    z, firma = extraer_vistas_fijas(red, base, estado, Path(salida))
    return z, firma


# --------------------------------------------------------------------------- ejecución
def crear_entrenador(z, y, y_idx, g, fruta, rasgos, catalogo):
    """Procedimiento común de ajuste: búsqueda interna agrupada + pesos de decisión internos."""
    def entrenar(nombre, idx, semilla):
        """Devuelve un predictor de probabilidades, el modelo y un resumen del ajuste interno."""
        if nombre == 'CNN_MobileNetV3_cabezal':
            cabeza, params, f1_int, prob_int = ajustar_cabezal_cnn(z[idx], y_idx[idx], g[idx], semilla)
            def predecir(ids):
                with torch.no_grad():
                    return cabeza(z[ids][:, 0]).softmax(1).numpy()
            modelo = cabeza
        else:
            tipo, est, rejilla, nombre_peso = catalogo[nombre]
            X = rasgos[tipo]
            modelo, params, f1_int, prob_int = ajustar_sklearn(est, rejilla, X[idx], y[idx], g[idx], semilla,
                                                               fruta[idx], nombre_peso)
            predecir = lambda ids: modelo.predict_proba(X[ids])
        pesos, f1_umbral = ajustar_pesos_decision(prob_int, y[idx])
        return predecir, modelo, {'parametros': params, 'f1_interno': f1_int, 'pesos_decision': pesos.tolist(),
                                  'f1_interno_umbral': f1_umbral}
    return entrenar


def validacion_anidada_completa(proyecto='.', semillas=(42, 7, 2026), salida=None):
    """Validación cruzada anidada y agrupada sobre TODAS las imágenes (complemento de la prueba).

    Cada imagen recibe una predicción externa por repetición, como en la evaluación de
    3 bloques anterior, pero los hiperparámetros y pesos de decisión se eligen siempre
    dentro del entrenamiento de cada bloque. Se repite con varias semillas de partición
    para medir la estabilidad. No se usa para elegir el modelo final (ya fijado).
    """
    proyecto = Path(proyecto).resolve()
    salida = Path(salida or proyecto / 'experiments/results/experimentos_v2')
    torch.set_num_threads(4)
    base, columnas, estado = cargar_base(proyecto)
    y = base.calidad_revisada.to_numpy()
    g = base.grupo_particion.to_numpy()
    fruta = base.fruta.to_numpy()
    Xm = base[columnas].to_numpy(float)
    z, _ = rasgos_cnn(proyecto, base, estado, salida)
    Zn = z.numpy()
    rasgos = {'manual': Xm, 'cnn': Zn[:, 0], 'fusion': np.hstack([Xm, Zn[:, 0]])}
    y_idx = np.array([CALIDADES.index(v) for v in y])
    catalogo = {k: (*v, None) for k, v in modelos_sklearn().items()}
    catalogo.update(modelos_pesos_fruta())
    nombres = ['Referencia_mayoritaria', 'RF_inicial', 'SVM_ajustado', 'CNN_MobileNetV3_cabezal',
               'Logistica_CNN', 'Logistica_CNN_pesos_fruta', 'Logistica_fusion_pesos_fruta']
    entrenar = crear_entrenador(z, y, y_idx, g, fruta, rasgos, catalogo)
    todas = np.arange(len(base))
    filas, registros = [], []
    inicio = time.time()
    for semilla in semillas:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla)
        pliegues = list(cv.split(todas, y, g))
        for tr, va in pliegues:
            assert not set(g[tr]) & set(g[va])
            assert set(y[tr]) == set(CALIDADES), 'Falta una clase en un entrenamiento externo.'
        for nombre in nombres:
            prob = np.full((len(base), 3), np.nan)
            pesos = np.full((len(base), 3), np.nan)
            for k, (tr, va) in enumerate(pliegues, 1):
                predecir, _, info = entrenar(nombre, tr, semilla + 100 * k)
                prob[va] = predecir(va)
                pesos[va] = info['pesos_decision']
            for variante in ('base', 'umbral'):
                w = 1. if variante == 'base' else pesos
                pred = np.array(CALIDADES)[(prob * w).argmax(1)]
                filas.append(pd.DataFrame({'semilla': semilla, 'modelo': nombre, 'variante': variante,
                                           'id_ejemplar': base.id_ejemplar, 'fruta': fruta,
                                           'fuente': base.fuente, 'real': y, 'prediccion': pred}))
                registros.append({'semilla': semilla, 'modelo': nombre, 'variante': variante, **metricas(y, pred)})
            print(f'[{time.time() - inicio:6.0f}s] anidada semilla {semilla} {nombre}', flush=True)
    por_semilla = pd.DataFrame(registros)
    claves = ['f1_macro', 'f1_ponderado', 'exactitud', 'recall_macro', 'recall_buena', 'recall_intermedia',
              'recall_mala', 'precision_intermedia', 'precision_mala']
    resumen = por_semilla.groupby(['modelo', 'variante'], sort=False)[claves].agg(['mean', 'std']).round(4)
    resumen.columns = [f'{a}_{"media" if b == "mean" else "desv"}' for a, b in resumen.columns]
    resumen = resumen.reset_index()
    guardar_tabla(por_semilla.round(4), salida / 'anidada_332_por_semilla.csv')
    guardar_tabla(resumen, salida / 'anidada_332_resumen.csv')
    guardar_tabla(pd.concat(filas, ignore_index=True), salida / 'anidada_332_predicciones.csv')
    guardar_bytes_si_cambio(salida / 'anidada_332.json', json.dumps(
        {'sha_base': estado['sha_base'], 'semillas': list(semillas), 'bloques': 5, 'imagenes': len(base),
         'grupos': int(len(set(g))), 'modelos': nombres, 'duracion_s': round(time.time() - inicio)},
        ensure_ascii=False, indent=2).encode('utf-8'))
    return resumen, por_semilla


def ejecutar(proyecto='.', salida=None, rapido=False):
    proyecto = Path(proyecto).resolve()
    salida = Path(salida or proyecto / 'experiments/results/experimentos_v2')
    salida.mkdir(parents=True, exist_ok=True)
    checkpoints = proyecto / 'experiments/checkpoints/experimentos_v2'
    checkpoints.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    base, columnas, estado = cargar_base(proyecto)
    etapa, candidatos, elegido = cargar_o_crear_particion(proyecto, base, estado)
    guardar_tabla(candidatos, salida / 'candidatos_particion.csv')
    dev = np.where(etapa == 'desarrollo')[0]
    test = np.where(etapa == 'prueba')[0]
    y = base.calidad_revisada.to_numpy()
    g = base.grupo_particion.to_numpy()
    lote = base.fuente.eq(FUENTE_LOTE).to_numpy()

    Xm = base[columnas].to_numpy(float)
    z, firma = rasgos_cnn(proyecto, base, estado, salida)
    Zn = z.numpy()
    rasgos = {'manual': Xm, 'cnn': Zn[:, 0], 'fusion': np.hstack([Xm, Zn[:, 0]])}
    y_idx = np.array([CALIDADES.index(v) for v in y])

    cv_dev = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
    pliegues = [(dev[a], dev[b]) for a, b in cv_dev.split(dev, y[dev], g[dev])]
    for tr, va in pliegues:
        assert not set(g[tr]) & set(g[va])
        for cond_tr in (tr, tr[~lote[tr]]):
            assert set(y[cond_tr]) == set(CALIDADES), 'Falta una clase en un entrenamiento.'

    condiciones = {'anterior': lambda idx: idx[~lote[idx]], 'ampliada': lambda idx: idx}
    catalogo = {k: (*v, None) for k, v in modelos_sklearn().items()}
    catalogo.update(modelos_pesos_fruta())
    fruta = base.fruta.to_numpy()
    if rapido:
        catalogo = {k: catalogo[k] for k in ['Referencia_mayoritaria', 'Logistica', 'Logistica_CNN_pesos_fruta']}
    nombres = list(catalogo) + ([] if rapido else ['CNN_MobileNetV3_cabezal'])

    entrenar = crear_entrenador(z, y, y_idx, g, fruta, rasgos, catalogo)
    registros, filas_pred, ajustes = [], [], []
    inicio = time.time()
    for cond, filtro in condiciones.items():
        for nombre in nombres:
            prob_dev = np.full((len(base), 3), np.nan)
            pesos_dev = np.full((len(base), 3), np.nan)
            for k, (tr, va) in enumerate(pliegues, 1):
                predecir, _, info = entrenar(nombre, filtro(tr), SEMILLA + 100 * k)
                prob_dev[va] = predecir(va)
                pesos_dev[va] = info['pesos_decision']
                ajustes.append({'condicion': cond, 'modelo': nombre, 'pliegue': k, **info,
                                'parametros': json.dumps(info['parametros'], default=str)})
            for variante in ('base', 'umbral'):
                w = 1. if variante == 'base' else pesos_dev[dev]
                pred = np.array(CALIDADES)[(prob_dev[dev] * w).argmax(1)]
                filas_pred.append(pd.DataFrame({'condicion': cond, 'modelo': nombre, 'variante': variante,
                                                'conjunto': 'desarrollo', 'id_ejemplar': base.id_ejemplar.iloc[dev],
                                                'real': y[dev], 'prediccion': pred,
                                                'pliegue': [next(k for k, (_, va) in enumerate(pliegues, 1) if i in va) for i in dev]}))
            print(f'[{time.time() - inicio:6.0f}s] desarrollo {cond:9s} {nombre}', flush=True)

    pred_dev = pd.concat(filas_pred, ignore_index=True).merge(
        base[['id_ejemplar', 'fruta', 'fuente', 'grupo_particion']], on='id_ejemplar')
    for (cond, nombre, variante), d in pred_dev.groupby(['condicion', 'modelo', 'variante'], sort=False):
        m = metricas(d.real, d.prediccion)
        por_pliegue = [f1_score(s.real, s.prediccion, labels=CALIDADES, average='macro', zero_division=0)
                       for _, s in d.groupby('pliegue')]
        registros.append({'conjunto': 'desarrollo', 'condicion': cond, 'modelo': nombre, 'variante': variante, **m,
                          'f1_macro_pliegues_media': float(np.mean(por_pliegue)),
                          'f1_macro_pliegues_desv': float(np.std(por_pliegue))})
    tabla_dev = pd.DataFrame(registros)

    # Selección fijada ANTES de evaluar la prueba (ver DECISIONES['metrica_seleccion']).
    amp = tabla_dev[(tabla_dev.condicion == 'ampliada') & (tabla_dev.modelo != 'Referencia_mayoritaria')]
    orden = amp.sort_values(['f1_macro', 'recall_macro'], ascending=False)
    final = orden.iloc[0]
    # Profundo: clasificadores sobre la CNN congelada (cabezal propio entrenado por gradiente o capa softmax
    # lineal regularizada, equivalente a un cabezal lineal). Tradicional: solo las 43 características manuales.
    tipo_rasgos = {k: v[0] for k, v in catalogo.items()} | {'CNN_MobileNetV3_cabezal': 'cnn'}
    profundos = orden[orden.modelo.map(tipo_rasgos).eq('cnn')]
    tradicionales = orden[orden.modelo.map(tipo_rasgos).eq('manual')]
    seleccion = {'modelo_final': {'modelo': final.modelo, 'variante': final.variante},
                 'mejor_profundo': (profundos.iloc[0][['modelo', 'variante']].to_dict() if len(profundos) else None),
                 'mejor_tradicional': tradicionales.iloc[0][['modelo', 'variante']].to_dict(),
                 'regla': DECISIONES['metrica_seleccion'],
                 'fijada_antes_de_prueba': True}
    guardar_bytes_si_cambio(salida / 'seleccion.json', json.dumps(seleccion, ensure_ascii=False, indent=2).encode())

    # Prueba: un ajuste final por modelo y condición sobre todo el desarrollo; una sola predicción.
    filas_test = []
    for cond, filtro in condiciones.items():
        for nombre in nombres:
            predecir, modelo, info = entrenar(nombre, filtro(dev), SEMILLA)
            prob = predecir(test)
            for variante in ('base', 'umbral'):
                w = 1. if variante == 'base' else np.array(info['pesos_decision'])
                filas_test.append(pd.DataFrame({'condicion': cond, 'modelo': nombre, 'variante': variante,
                                                'conjunto': 'prueba', 'id_ejemplar': base.id_ejemplar.iloc[test],
                                                'real': y[test], 'prediccion': np.array(CALIDADES)[(prob * w).argmax(1)],
                                                **{f'p_{c}': prob[:, i] for i, c in enumerate(CALIDADES)}}))
            ajustes.append({'condicion': cond, 'modelo': nombre, 'pliegue': 'final', **info,
                            'parametros': json.dumps(info['parametros'], default=str)})
            if cond == 'ampliada' and nombre != 'Referencia_mayoritaria':
                paquete = {'modelo': modelo, 'rasgos': (catalogo[nombre][0] if nombre in catalogo else 'cnn'),
                           'caracteristicas': columnas, 'pesos_decision': info['pesos_decision'],
                           'parametros': info['parametros'], 'calidades': CALIDADES, 'sha_base': estado['sha_base'],
                           'entrenado_con': 'desarrollo (sin prueba)'}
                if nombre == 'CNN_MobileNetV3_cabezal':
                    torch.save({'state_dict': modelo.state_dict(), **{k: v for k, v in paquete.items() if k != 'modelo'}},
                               checkpoints / f'{nombre.lower()}.pt')
                else:
                    joblib.dump(paquete, checkpoints / f'{nombre.lower()}.joblib', compress=3)
            print(f'[{time.time() - inicio:6.0f}s] prueba     {cond:9s} {nombre}', flush=True)
    pred_test = pd.concat(filas_test, ignore_index=True).merge(
        base[['id_ejemplar', 'fruta', 'fuente', 'grupo_particion']], on='id_ejemplar')
    for (cond, nombre, variante), d in pred_test.groupby(['condicion', 'modelo', 'variante'], sort=False):
        registros.append({'conjunto': 'prueba', 'condicion': cond, 'modelo': nombre, 'variante': variante,
                          **metricas(d.real, d.prediccion),
                          **bootstrap_grupos(d.real, d.prediccion, d.grupo_particion)})
    resultados = pd.DataFrame(registros)

    # Desgloses por fuente y fruta, y matrices de confusión.
    todas = pd.concat([pred_dev, pred_test], ignore_index=True)
    desglose = []
    for campo in ['fuente', 'fruta']:
        for claves, d in todas.groupby(['conjunto', 'condicion', 'modelo', 'variante', campo], sort=False):
            desglose.append({'conjunto': claves[0], 'condicion': claves[1], 'modelo': claves[2], 'variante': claves[3],
                             'campo': campo, 'valor': claves[4], 'imagenes': len(d), **metricas(d.real, d.prediccion)})
    matrices = []
    for claves, d in todas.groupby(['conjunto', 'condicion', 'modelo', 'variante'], sort=False):
        cm = confusion_matrix(d.real, d.prediccion, labels=CALIDADES)
        for i, r in enumerate(CALIDADES):
            matrices.append({'conjunto': claves[0], 'condicion': claves[1], 'modelo': claves[2],
                             'variante': claves[3], 'real': r, **{f'pred_{c}': int(cm[i, j]) for j, c in enumerate(CALIDADES)}})

    guardar_tabla(resultados.round(4).astype({c: str for c in resultados.columns if c.endswith('_ic95')}),
                  salida / 'metricas.csv')
    guardar_tabla(pd.DataFrame(desglose).round(4), salida / 'metricas_por_fuente_y_fruta.csv')
    guardar_tabla(pd.DataFrame(matrices), salida / 'matrices_confusion.csv')
    guardar_tabla(todas, salida / 'predicciones.csv')
    guardar_tabla(pd.DataFrame(ajustes), salida / 'ajustes.csv')
    resumen = {'sha_base': estado['sha_base'], 'semilla': SEMILLA, 'decisiones': DECISIONES,
               'particion': {'elegida': {k: (v.item() if hasattr(v, 'item') else v) for k, v in elegido.items()},
                             'desarrollo': {'imagenes': len(dev), 'grupos': int(len(set(g[dev]))),
                                            'clases': pd.Series(y[dev]).value_counts().to_dict()},
                             'prueba': {'imagenes': len(test), 'grupos': int(len(set(g[test]))),
                                        'clases': pd.Series(y[test]).value_counts().to_dict(),
                                        'fuentes': base.fuente.iloc[test].value_counts().to_dict()}},
               'extractor_cnn': firma, 'seleccion': seleccion, 'duracion_s': round(time.time() - inicio)}
    guardar_bytes_si_cambio(salida / 'decisiones.json',
                            json.dumps(resumen, ensure_ascii=False, indent=2, default=str).encode('utf-8'))
    return resultados, seleccion


def escribir_manifiesto(proyecto='.'):
    """Manifiesto para la interfaz: archivo de cada modelo v2 y variante elegida por F1 macro de desarrollo.

    Los modelos se entrenaron solo con desarrollo, de modo que las métricas de prueba
    corresponden exactamente a los pesos que usa la aplicación.
    """
    proyecto = Path(proyecto).resolve()
    res = proyecto / 'experiments/results/experimentos_v2'
    ck = proyecto / 'experiments/checkpoints/experimentos_v2'
    m = pd.read_csv(res / 'metricas.csv')
    sel = json.loads((res / 'seleccion.json').read_text(encoding='utf-8'))
    dev = m[(m.conjunto == 'desarrollo') & (m.condicion == 'ampliada') & (m.modelo != 'Referencia_mayoritaria')]
    prueba = m[(m.conjunto == 'prueba') & (m.condicion == 'ampliada')].set_index(['modelo', 'variante'])
    manifiesto = {'sha_base': json.loads((res / 'decisiones.json').read_text(encoding='utf-8'))['sha_base'],
                  'modelo_final': sel['modelo_final'], 'modelos': {}}
    for modelo, d in dev.groupby('modelo'):
        mejor = d.sort_values(['f1_macro', 'recall_macro'], ascending=False).iloc[0]
        archivo = ck / (modelo.lower() + ('.pt' if modelo == 'CNN_MobileNetV3_cabezal' else '.joblib'))
        if archivo.is_file():
            manifiesto['modelos'][modelo] = {
                'archivo': archivo.relative_to(proyecto).as_posix(), 'variante': mejor.variante,
                'f1_macro_desarrollo': round(float(mejor.f1_macro), 4),
                'f1_macro_prueba': round(float(prueba.loc[(modelo, mejor.variante), 'f1_macro']), 4)}
    guardar_bytes_si_cambio(ck / 'manifiesto.json', json.dumps(manifiesto, ensure_ascii=False, indent=2).encode())
    return manifiesto


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proyecto', default='.')
    parser.add_argument('--rapido', action='store_true', help='Prueba de humo con dos modelos.')
    parser.add_argument('--anidada', action='store_true',
                        help='Solo la validación anidada sobre todas las imágenes (3 semillas).')
    args = parser.parse_args()
    if args.anidada:
        resumen, _ = validacion_anidada_completa(args.proyecto)
        pd.set_option('display.width', 220)
        print(resumen.to_string(index=False))
        raise SystemExit(0)
    resultados, seleccion = ejecutar(args.proyecto, rapido=args.rapido)
    if not args.rapido:
        escribir_manifiesto(args.proyecto)
    cols = ['conjunto', 'condicion', 'modelo', 'variante', 'exactitud', 'f1_macro', 'f1_ponderado', 'recall_macro',
            'recall_intermedia', 'recall_mala']
    pd.set_option('display.width', 220)
    print(resultados[cols].round(3).to_string(index=False))
    print(json.dumps(seleccion, ensure_ascii=False, indent=2))
