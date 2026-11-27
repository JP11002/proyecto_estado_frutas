"""Incorporar a data/base/ un lote revisado de Drive sin modificar los registros previos.

Flujo (lote drive_20261126):
1. data/annotations/drive_20261126/inventario_nuevas.csv clasifica cada archivo
   descargado (original, variante aumentada, duplicado exacto o stock).
2. revision_visual.json contiene la decisión visual, el grupo conservador y la
   receta de máscara de cada original revisado.
3. Este módulo reconstruye las máscaras desde las recetas, las guarda como
   anotación versionada, calcula las 43 características y la entrada CNN de
   fotografía completa, y añade las filas a data/base/datos.csv.

Las filas existentes se conservan byte a byte en sus campos, y sus grupos no
cambian. Si el lote ya estaba incorporado, sus filas se reemplazan (idempotente).

Uso:  python -m src.data.incorporar_lote --proyecto . [--galeria]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.data.base import cargar_base, guardar_base
from src.data.entrada_cnn import ENTRADA_CNN, firma_comparable, preparar_imagen_completa
from src.data.preparacion_imagenes import (
    _componente_principal, construir_grupos, extraer_caracteristicas, leer_rgb, mascara_grabcut,
    validar_mascara,
)
from src.data.preparar_drive import proponer_mascara
from src.utils.archivos import guardar_bytes_si_cambio, guardar_tabla

LOTE = 'drive_20261126'
METODO = 'lote_20261126_receta_v1'


def _refinar_con_grabcut(rgb, aproximada):
    """Refinar un contorno manual aproximado: núcleo seguro, banda probable y fondo seguro."""
    lado = min(rgb.shape[:2])
    k = max(3, int(round(0.06 * lado)) | 1)
    nucleo = cv2.erode(aproximada, np.ones((k, k), np.uint8))
    exterior = cv2.dilate(aproximada, np.ones((k, k), np.uint8))
    etiquetas = np.full(aproximada.shape, cv2.GC_BGD, np.uint8)
    etiquetas[exterior > 0] = cv2.GC_PR_BGD
    etiquetas[aproximada > 0] = cv2.GC_PR_FGD
    etiquetas[nucleo > 0] = cv2.GC_FGD
    cv2.setRNGSeed(42)
    try:
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), etiquetas, None,
                    np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return aproximada
    refinada = _componente_principal(np.isin(etiquetas, [cv2.GC_FGD, cv2.GC_PR_FGD]))
    # Si GrabCut se aleja demasiado de la anotación, conservar el contorno manual.
    union = np.logical_or(refinada, aproximada).sum()
    iou = np.logical_and(refinada, aproximada).sum() / max(1, union)
    return refinada if iou >= 0.7 else aproximada


def mascara_desde_receta(rgb, fruta, receta):
    """Máscara determinista a resolución original a partir de la receta revisada."""
    h, w = rgb.shape[:2]
    tipo = receta['tipo']
    if tipo == 'grabcut_rectangulo':
        return validar_mascara(rgb, mascara_grabcut(rgb, receta['rectangulo'], lado_maximo=700))
    if tipo == 'color_hsv':
        return validar_mascara(rgb, proponer_mascara(rgb, fruta, 'publica', {}))
    aproximada = np.zeros((h, w), np.uint8)
    if tipo == 'elipse_grabcut':
        cx, cy, rx, ry = receta['elipse']
        cv2.ellipse(aproximada, (int(round(cx * (w - 1))), int(round(cy * (h - 1)))),
                    (max(1, int(round(rx * w))), max(1, int(round(ry * h)))), 0, 0, 360, 1, cv2.FILLED)
    elif tipo == 'poligono_grabcut':
        puntos = np.rint(np.asarray(receta['puntos'], float) * [w - 1, h - 1]).astype(np.int32)
        cv2.fillPoly(aproximada, [puntos], 1)
    else:
        raise ValueError('Receta de máscara desconocida: ' + tipo)
    return validar_mascara(rgb, _refinar_con_grabcut(rgb, aproximada))


def _png(arreglo):
    buffer = io.BytesIO()
    Image.fromarray(arreglo).save(buffer, format='PNG')
    return buffer.getvalue()


def preparar_lote(proyecto):
    """Construir las filas nuevas con máscara, entrada CNN y características."""
    proyecto = Path(proyecto).resolve()
    anotaciones = proyecto / 'data/annotations' / LOTE
    revision = json.loads((anotaciones / 'revision_visual.json').read_text(encoding='utf-8'))
    inventario = pd.read_csv(anotaciones / 'inventario_nuevas.csv').set_index('drive_id')
    temporal = proyecto / 'data/processed' / LOTE / 'entrada_cnn'
    filas = []
    for r in revision['revisiones']:
        if r['decision_imagen'] != 'incluir':
            continue
        fuente = inventario.loc[r['drive_id']]
        if fuente.categoria != 'original':
            raise ValueError('Solo se incorporan originales: ' + r['codigo_revision'])
        original = proyecto / fuente.ruta_local
        sha_original = hashlib.sha256(original.read_bytes()).hexdigest()
        if sha_original != fuente.sha256:
            raise ValueError('Cambió el original respecto al inventario: ' + r['codigo_revision'])
        if r['calidad_revisada'] not in {'buena', 'intermedia', 'mala'}:
            raise ValueError('Calidad inválida en ' + r['codigo_revision'])
        if r['fruta'] == 'manzana' and r['manzana_roja'] != 'si':
            raise ValueError('Manzana fuera del alcance rojo: ' + r['codigo_revision'])
        rgb = leer_rgb(original)
        mascara = mascara_desde_receta(rgb, r['fruta'], r['receta_mascara'])
        id_ejemplar = 'drive_' + r['drive_id']
        contenido_mascara = _png(mascara * 255)
        sha_mascara = hashlib.sha256(contenido_mascara).hexdigest()
        carpeta = anotaciones / 'mascaras_revisadas' / id_ejemplar
        guardar_bytes_si_cambio(carpeta / 'mascara.png', contenido_mascara)
        receta = {'metodo': METODO, 'receta': r['receta_mascara'], 'sha_original': sha_original,
                  'sha_mascara': sha_mascara, 'opencv': cv2.__version__}
        guardar_bytes_si_cambio(carpeta / 'receta.json',
                                json.dumps(receta, ensure_ascii=False, indent=2).encode('utf-8'))
        contenido_cnn = _png(preparar_imagen_completa(rgb))
        ruta_cnn = temporal / (id_ejemplar + '.png')
        guardar_bytes_si_cambio(ruta_cnn, contenido_cnn)
        rasgos = extraer_caracteristicas(rgb, mascara)
        firma = hashlib.sha256(json.dumps(receta, sort_keys=True).encode()).hexdigest()
        filas.append({
            'id_ejemplar': id_ejemplar, 'id_origen': id_ejemplar, 'codigo_revision': r['codigo_revision'],
            'fuente': revision['fuente_base'], 'fruta': r['fruta'], 'calidad_revisada': r['calidad_revisada'],
            'grupo_ejemplar': r['grupo_ejemplar'], 'sha_original': sha_original, 'sha_mascara': sha_mascara,
            'sha_cnn': hashlib.sha256(contenido_cnn).hexdigest(), 'firma_preparacion': firma,
            'revisor': revision['revisor'], 'ruta_original': fuente.ruta_local,
            'ruta_mascara': (carpeta / 'mascara.png').relative_to(proyecto).as_posix(),
            'ruta_cnn': ruta_cnn.relative_to(proyecto).as_posix(),
            'id_fruto_propuesto': np.nan, 'certeza_identidad': 'sin_revision_fisica',
            'revisor_identidad': np.nan, **rasgos, 'preparacion_cnn': ENTRADA_CNN})
    return pd.DataFrame(filas), revision


def incorporar(proyecto):
    proyecto = Path(proyecto).resolve()
    base, columnas, estado = cargar_base(proyecto)
    anterior = pd.read_csv(proyecto / 'data/base/datos.csv', dtype=str, keep_default_na=False)
    nuevas, revision = preparar_lote(proyecto)
    previa = anterior.loc[~anterior.fuente.eq(revision['fuente_base'])].copy()
    if nuevas.id_ejemplar.isin(previa.id_ejemplar).any():
        raise ValueError('Un ejemplar del lote ya existe con otra fuente.')
    if nuevas.sha_original.isin(previa.sha_original).any():
        raise ValueError('Un original del lote coincide byte a byte con la base previa.')
    orden = list(anterior.columns)
    nuevas['grupo_particion'] = ''  # se calcula después con construir_grupos
    nuevas = nuevas[orden]
    tabla = pd.concat([previa.astype(object), nuevas.astype(object)], ignore_index=True)
    grupos = construir_grupos(tabla.fillna(''))
    # Los grupos previos no pueden cambiar al añadir el lote.
    if not (grupos[:len(previa)] == previa.grupo_particion.to_numpy()).all():
        raise AssertionError('La incorporación alteró grupos existentes.')
    tabla['grupo_particion'] = grupos
    if set(tabla.grupo_particion[len(previa):]) & set(previa.grupo_particion):
        raise AssertionError('Un grupo nuevo coincide con uno previo.')
    for c in columnas:
        tabla[c] = tabla[c].astype(float)
    estado = dict(estado)
    estado.update(ejemplares=len(tabla), grupos_conservadores=int(tabla.grupo_particion.nunique()),
                  lotes_incorporados={LOTE: {
                      'ejemplares': len(nuevas), 'fuente': revision['fuente_base'],
                      'revision': f'data/annotations/{LOTE}/revision_visual.json',
                      'inventario': f'data/annotations/{LOTE}/inventario_nuevas.csv'}},
                  firma_etiquetas_grupos=firma_comparable(tabla))
    base, _, estado = guardar_base(proyecto, tabla, estado)
    # Comprobar que las filas previas no cambiaron en ningún campo.
    actual = pd.read_csv(proyecto / 'data/base/datos.csv', dtype=str, keep_default_na=False)
    if not actual.iloc[:len(previa)].reset_index(drop=True).equals(previa.reset_index(drop=True)):
        raise AssertionError('Cambiaron registros previos de la base.')
    resumen = base.groupby(['fuente', 'fruta', 'calidad_revisada']).agg(
        ejemplares=('id_ejemplar', 'size'), grupos_conservadores=('grupo_particion', 'nunique')).reset_index()
    guardar_tabla(resumen, proyecto / 'data/base/resumen_clases.csv')
    esperado = pd.MultiIndex.from_product([['manzana', 'banano', 'limon'], ['buena', 'intermedia', 'mala']],
                                          names=['fruta', 'calidad_revisada'])
    cobertura = base.groupby(['fruta', 'calidad_revisada']).size().reindex(esperado, fill_value=0)
    guardar_tabla(cobertura.rename('ejemplares').reset_index(), proyecto / 'data/base/cobertura.csv')
    return base, nuevas, estado


def galeria(proyecto, nuevas, destino):
    """Original con contorno de la máscara, para la revisión visual de cada ejemplar."""
    proyecto = Path(proyecto)
    celda, columnas = 220, 6
    filas = (len(nuevas) + columnas - 1) // columnas
    lienzo = Image.new('RGB', (columnas * celda, filas * (celda + 16)), 'white')
    dibujo = ImageDraw.Draw(lienzo)
    for i, r in enumerate(nuevas.itertuples()):
        rgb = leer_rgb(proyecto / r.ruta_original)
        mascara = np.asarray(Image.open(proyecto / r.ruta_mascara)) > 0
        contornos, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contornos, -1, (255, 0, 60), max(1, round(max(rgb.shape[:2]) / 150)))
        im = Image.fromarray(rgb)
        f = celda / max(im.size)
        im = im.resize((max(1, int(im.width * f)), max(1, int(im.height * f))))
        x, y = (i % columnas) * celda, (i // columnas) * (celda + 16)
        lienzo.paste(im, (x, y))
        dibujo.text((x + 3, y + celda + 2), f'{r.codigo_revision} {r.calidad_revisada}', fill='black')
    Path(destino).parent.mkdir(parents=True, exist_ok=True)
    lienzo.save(destino, quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proyecto', default='.')
    parser.add_argument('--galeria', action='store_true')
    args = parser.parse_args()
    base, nuevas, estado = incorporar(args.proyecto)
    print(pd.crosstab([base.fuente, base.fruta], base.calidad_revisada, margins=True).to_string())
    print('Registros:', len(base), '| grupos:', estado['grupos_conservadores'], '| huella:', estado['sha_base'])
    if args.galeria:
        destino = Path(args.proyecto) / 'experiments/results' / LOTE / 'galeria_mascaras.jpg'
        galeria(args.proyecto, base.loc[base.fuente.eq('propia_lote_20261126')].assign(
            ruta_original=lambda d: [Path(p).relative_to(Path(args.proyecto).resolve()) for p in d.ruta_original],
            ruta_mascara=lambda d: [Path(p).relative_to(Path(args.proyecto).resolve()) for p in d.ruta_mascara]),
            destino)
        print('Galería:', destino)


if __name__ == '__main__':
    main()
