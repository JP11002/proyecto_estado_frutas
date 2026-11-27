"""Figuras vectoriales (SVG y PDF) de los experimentos v2 para el informe."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / 'experiments/results/experimentos_v2'
FIG = RAIZ / 'docs/figuras'
CALIDADES = ['buena', 'intermedia', 'mala']
COLORES = {'buena': '#36856c', 'intermedia': '#dea33c', 'mala': '#ba5260'}
NOMBRES = {'RF_inicial': 'RF inicial', 'SVM_ajustado': 'SVM ajustado', 'RF_ajustado': 'RF ajustado',
           'Logistica': 'Logística', 'HistGradientBoosting': 'HistGB', 'Logistica_CNN': 'Logística CNN',
           'Logistica_fusion': 'Logística fusión', 'CNN_MobileNetV3_cabezal': 'CNN (MobileNetV3)',
           'Referencia_mayoritaria': 'Mayoritaria', 'HistGB_pesos_fruta': 'HistGB pesos fruta',
           'Logistica_CNN_pesos_fruta': 'Log. CNN pesos fruta', 'Logistica_fusion_pesos_fruta': 'Log. fusión pesos fruta'}


def guardar(fig, nombre):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ['svg', 'pdf', 'png']:
        fig.savefig(FIG / f'{nombre}.{ext}', bbox_inches='tight', dpi=160)
    plt.close(fig)


def etiqueta(r):
    return NOMBRES.get(r.modelo, r.modelo) + (' +umbral' if r.variante == 'umbral' else '')


def main():
    m = pd.read_csv(RES / 'metricas.csv')
    m = m[~((m.modelo == 'Referencia_mayoritaria') & (m.variante == 'umbral'))]
    sel = json.loads((RES / 'seleccion.json').read_text(encoding='utf-8'))

    # 1. Comparación de modelos (condición ampliada): F1 macro y recall por clase, desarrollo y prueba.
    fig, ejes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    for ax, conjunto in zip(ejes, ['desarrollo', 'prueba']):
        d = m[(m.conjunto == conjunto) & (m.condicion == 'ampliada')].copy()
        d['nombre'] = [etiqueta(r) for r in d.itertuples()]
        d = d.sort_values('f1_macro')
        yy = np.arange(len(d))
        ax.barh(yy, d.f1_macro, color='#4c6a92', height=.55, label='F1 macro')
        for i, c in enumerate(CALIDADES):
            ax.scatter(d[f'recall_{c}'], yy, color=COLORES[c], s=22, zorder=3, label=f'Recall {c}')
        if conjunto == 'prueba':
            ic = d.f1_macro_ic95.apply(lambda s: json.loads(s) if isinstance(s, str) and s.startswith('[') else [np.nan] * 2)
            lo, hi = np.array(ic.tolist()).T
            ax.errorbar(d.f1_macro, yy, xerr=[d.f1_macro - lo, hi - d.f1_macro], fmt='none', ecolor='black', lw=.8)
        ax.set_yticks(yy, d.nombre)
        ax.set_xlim(0, 1)
        ax.set_title(f'{conjunto.capitalize()} ({"OOF 5 bloques" if conjunto == "desarrollo" else "IC95 % por grupos"})')
        ax.grid(axis='x', alpha=.3)
    ejes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle('Base ampliada: F1 macro (barras) y recall por clase (puntos)')
    guardar(fig, 'v2_comparacion_modelos')

    # 2. Efecto de los datos nuevos: misma evaluación, entrenamiento anterior vs ampliado.
    modelos = [k for k in ['RF_inicial', 'SVM_ajustado', 'RF_ajustado', 'Logistica', 'HistGradientBoosting',
                           'Logistica_CNN', 'Logistica_fusion', 'CNN_MobileNetV3_cabezal'] if k in set(m.modelo)]
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, conjunto in zip(ejes, ['desarrollo', 'prueba']):
        d = m[(m.conjunto == conjunto) & (m.variante == 'base')].set_index(['modelo', 'condicion'])
        x = np.arange(len(modelos))
        for j, cond in enumerate(['anterior', 'ampliada']):
            ax.bar(x + (j - .5) * .38, [d.loc[(k, cond), 'f1_macro'] for k in modelos], .38,
                   label=f'Entrenado con base {cond}', color=['#9aa7b8', '#2f4b7c'][j])
        ax.set_xticks(x, [NOMBRES[k] for k in modelos], rotation=35, ha='right')
        ax.set_title(conjunto.capitalize())
        ax.grid(axis='y', alpha=.3)
    ejes[0].set_ylabel('F1 macro')
    ejes[0].legend(fontsize=8)
    fig.suptitle('Efecto del lote nuevo con las mismas filas de evaluación')
    guardar(fig, 'v2_efecto_datos')

    # 3. Matrices de confusión del modelo final y del mejor profundo (desarrollo y prueba).
    cm = pd.read_csv(RES / 'matrices_confusion.csv')
    elegidos = [('Modelo final', sel['modelo_final'])]
    if sel.get('mejor_profundo'):
        elegidos.append(('Mejor profundo', sel['mejor_profundo']))
    fig, ejes = plt.subplots(len(elegidos), 2, figsize=(9, 4.2 * len(elegidos)), squeeze=False)
    for fila, (titulo, s) in enumerate(elegidos):
        for col, conjunto in enumerate(['desarrollo', 'prueba']):
            d = cm[(cm.conjunto == conjunto) & (cm.condicion == 'ampliada') & (cm.modelo == s['modelo'])
                   & (cm.variante == s['variante'])].set_index('real').loc[CALIDADES]
            mat = d[[f'pred_{c}' for c in CALIDADES]].to_numpy()
            ax = ejes[fila, col]
            norm = mat / mat.sum(1, keepdims=True)
            ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{mat[i, j]}\n({norm[i, j]:.0%})', ha='center', va='center',
                            color='white' if norm[i, j] > .55 else 'black', fontsize=9)
            ax.set_xticks(range(3), CALIDADES)
            ax.set_yticks(range(3), CALIDADES)
            ax.set_xlabel('Predicción')
            ax.set_ylabel('Real')
            ax.set_title(f'{titulo}: {NOMBRES.get(s["modelo"], s["modelo"])}'
                         f'{" +umbral" if s["variante"] == "umbral" else ""} — {conjunto}', fontsize=9)
    guardar(fig, 'v2_matrices_confusion')
    figuras_informe(m, sel, cm)
    print('Figuras en', FIG)


def figuras_informe(m, sel, cm):
    """Versiones de una columna (3,5 pulgadas) para el informe IEEE."""
    plt.rcParams.update({'font.size': 7, 'axes.titlesize': 7.5})
    claves = [('Referencia_mayoritaria', 'base'), ('RF_inicial', 'base'), ('SVM_ajustado', 'base'),
              ('CNN_MobileNetV3_cabezal', 'base'), ('Logistica_CNN', 'base'),
              (sel['mejor_tradicional']['modelo'], sel['mejor_tradicional']['variante']),
              (sel['mejor_profundo']['modelo'], sel['mejor_profundo']['variante']),
              (sel['modelo_final']['modelo'], sel['modelo_final']['variante'])]
    claves = list(dict.fromkeys(claves))
    t = m[(m.conjunto == 'prueba') & (m.condicion == 'ampliada')].set_index(['modelo', 'variante'])
    d = t.loc[[k for k in claves if k in t.index]].reset_index()
    d['nombre'] = [etiqueta(r) for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    yy = np.arange(len(d))[::-1]
    ic = np.array([json.loads(s) for s in d.f1_macro_ic95])
    ax.barh(yy, d.f1_macro, color='#4c6a92', height=.55, label='F1 macro (IC95 %)')
    ax.errorbar(d.f1_macro, yy, xerr=[d.f1_macro - ic[:, 0], ic[:, 1] - d.f1_macro], fmt='none', ecolor='black', lw=.7)
    for c in ['intermedia', 'mala']:
        ax.scatter(d[f'recall_{c}'], yy, color=COLORES[c], s=14, zorder=3, label=f'Recall {c}')
    ax.set_yticks(yy, d.nombre)
    ax.set_xlim(0, 1)
    ax.grid(axis='x', alpha=.3)
    ax.legend(fontsize=6, loc='lower right')
    ax.set_title('Prueba independiente (43 imágenes, 11 grupos)')
    guardar(fig, 'v2_informe_prueba')

    # Atajo por fruta: recall por fruta y clase en desarrollo, antes y después de pesos fruta-calidad.
    des = pd.read_csv(RES / 'metricas_por_fuente_y_fruta.csv')
    # Ablación limpia: mismo modelo y rejilla; solo cambia la ponderación.
    pares = [('Logistica_CNN', 'base', 'Logística CNN: pesos por clase'),
             ('Logistica_CNN_pesos_fruta', 'base', 'Logística CNN: pesos fruta-calidad')]
    fig, ejes = plt.subplots(1, 2, figsize=(3.5, 2.5), sharey=True)
    for ax, (mod, var, titulo) in zip(ejes, pares):
        s = des[(des.conjunto == 'desarrollo') & (des.condicion == 'ampliada') & (des.modelo == mod)
                & (des.variante == var) & (des.campo == 'fruta')].set_index('valor').reindex(['manzana', 'banano', 'limon'])
        x = np.arange(3)
        for j, c in enumerate(CALIDADES):
            ax.bar(x + (j - 1) * .27, s[f'recall_{c}'], .27, color=COLORES[c], label=c)
        ax.set_xticks(x, ['manz.', 'banano', 'limón'], fontsize=6)
        ax.set_title(titulo, fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=.3)
    ejes[0].set_ylabel('Recall (desarrollo)')
    manejadores, nombres = ejes[0].get_legend_handles_labels()
    fig.legend(manejadores, nombres, loc='lower center', ncol=3, fontsize=6, frameon=False, bbox_to_anchor=(.5, -.04))
    fig.tight_layout(rect=(0, .07, 1, 1))
    guardar(fig, 'v2_informe_fruta')

    # Matriz del modelo final en prueba.
    s = sel['modelo_final']
    d = cm[(cm.conjunto == 'prueba') & (cm.condicion == 'ampliada') & (cm.modelo == s['modelo'])
           & (cm.variante == s['variante'])].set_index('real').loc[CALIDADES]
    mat = d[[f'pred_{c}' for c in CALIDADES]].to_numpy()
    norm = mat / mat.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f'{mat[i, j]}\n({norm[i, j]:.0%})', ha='center', va='center',
                    color='white' if norm[i, j] > .55 else 'black', fontsize=7)
    ax.set_xticks(range(3), CALIDADES)
    ax.set_yticks(range(3), CALIDADES)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Real')
    ax.set_title('Modelo final — prueba independiente')
    guardar(fig, 'v2_informe_confusion')


if __name__ == '__main__':
    sys.exit(main())
