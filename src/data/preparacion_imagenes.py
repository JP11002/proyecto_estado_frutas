"""Operaciones de imagen usadas por el cuaderno 02, sin inferir etiquetas de calidad.

Las máscaras automáticas son propuestas. El equipo debe revisarlas antes de usar
sus características o su ocupación como mediciones de la fruta.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from src.utils.archivos import guardar_bytes_si_cambio, guardar_tabla

VERSION = 'preparacion_v1'


def guardar_figura(figura, ruta, **opciones):
    contenido = io.BytesIO()
    figura.savefig(contenido, format='jpeg', **opciones)
    guardar_bytes_si_cambio(ruta, contenido.getvalue())


def leer_rgb(ruta):
    """Leer con orientación EXIF y RGB, independientemente del sistema operativo."""
    if Path(ruta).suffix.lower() in {'.heic', '.heif'}:
        try:
            from pillow_heif import register_heif_opener
        except ImportError as error:
            raise ImportError('Instalar pillow-heif para leer las fotografías HEIC del equipo.') from error
        register_heif_opener()
    with Image.open(ruta) as imagen:
        imagen.load()
        return np.asarray(ImageOps.exif_transpose(imagen).convert('RGB')).copy()


def validar_mascara(rgb, mascara):
    if mascara.shape != rgb.shape[:2]:
        raise ValueError('La máscara debe tener el alto y ancho de la imagen original.')
    if not np.isin(mascara, [0, 1, False, True]).all():
        raise ValueError('La máscara debe ser binaria: 0 para fondo y 1 para fruta.')
    mascara = mascara.astype(np.uint8)
    if mascara.sum() < 25:
        raise ValueError('La máscara contiene menos de 25 píxeles; revisar la segmentación.')
    return mascara


def _componente_principal(mascara):
    """Conservar el componente mayor y rellenar huecos, incluidos defectos oscuros."""
    mascara = (mascara > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resultado = np.zeros_like(mascara)
    if contornos:
        cv2.drawContours(resultado, [max(contornos, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    return resultado


def mascara_otsu(rgb):
    """Otsu en gris. Elegir la polaridad que menos ocupa el borde de la imagen.

    Es una hipótesis de fondo, no una identificación semántica de fruta. Un fondo
    con textura o una mano puede quedar dentro de la máscara.
    """
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gris = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(gris, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    borde = np.concatenate([binaria[0], binaria[-1], binaria[:, 0], binaria[:, -1]])
    if borde.mean() > 0.5:
        binaria = 1 - binaria
    return _componente_principal(binaria)


def mascara_grabcut(rgb, rectangulo=None, semilla=42, lado_maximo=320):
    """GrabCut con rectángulo normalizado (x0, y0, x1, y1), en la imagen original.

    Reducir solo para calcular la propuesta; restaurar la máscara con vecino más
    cercano. El rectángulo debe encerrar la fruta completa y dejar algo de fondo.
    """
    if rectangulo is None:
        rectangulo = (0.02, 0.02, 0.98, 0.98)
    x0, y0, x1, y1 = map(float, rectangulo)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError('Rectángulo inválido; usar valores normalizados entre 0 y 1.')
    h, w = rgb.shape[:2]
    escala = min(1, lado_maximo / max(h, w))
    pequeno = cv2.resize(rgb, (max(8, round(w * escala)), max(8, round(h * escala))))
    ph, pw = pequeno.shape[:2]
    izquierda, arriba = int(x0 * pw), int(y0 * ph)
    derecha, abajo = min(pw, int(np.ceil(x1 * pw))), min(ph, int(np.ceil(y1 * ph)))
    if derecha - izquierda < 3 or abajo - arriba < 3:
        raise ValueError('El rectángulo es demasiado pequeño.')
    if izquierda == arriba == 0 and derecha == pw and abajo == ph:
        raise ValueError('GrabCut necesita algo de fondo fuera del rectángulo.')
    mascara = np.zeros((ph, pw), dtype=np.uint8)
    cv2.setRNGSeed(int(semilla))
    cv2.grabCut(cv2.cvtColor(pequeno, cv2.COLOR_RGB2BGR), mascara,
                (izquierda, arriba, derecha - izquierda, abajo - arriba),
                np.zeros((1, 65)), np.zeros((1, 65)), 4, cv2.GC_INIT_WITH_RECT)
    mascara = np.isin(mascara, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    mascara = _componente_principal(mascara)
    return cv2.resize(mascara, (w, h), interpolation=cv2.INTER_NEAREST)


def mascara_poligono(rgb, puntos):
    """Crear una máscara manual con puntos (x, y) normalizados entre 0 y 1."""
    puntos = np.asarray(puntos, dtype=float)
    if puntos.ndim != 2 or puntos.shape[1] != 2 or len(puntos) < 3:
        raise ValueError('Se necesitan al menos tres puntos [x, y].')
    if not np.isfinite(puntos).all() or (puntos < 0).any() or (puntos > 1).any():
        raise ValueError('Las coordenadas deben estar entre 0 y 1.')
    h, w = rgb.shape[:2]
    pixeles = np.rint(puntos * [w - 1, h - 1]).astype(np.int32)
    if abs(cv2.contourArea(pixeles)) < 25:
        raise ValueError('El polígono tiene un área demasiado pequeña.')
    mascara = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mascara, [pixeles], 1)
    return validar_mascara(rgb, mascara)


def diagnosticar_mascara(rgb, mascara):
    """Alertas geométricas; no sustituyen una revisión visual de la máscara."""
    mascara = validar_mascara(rgb, mascara)
    ocupacion = float(mascara.mean())
    toca = bool(mascara[0].any() or mascara[-1].any() or
                mascara[:, 0].any() or mascara[:, -1].any())
    alertas = []
    if ocupacion < 0.03:
        alertas.append('ocupacion_menor_3_por_ciento')
    if ocupacion > 0.85:
        alertas.append('ocupacion_mayor_85_por_ciento')
    if toca:
        alertas.append('mascara_toca_borde')
    return {'ocupacion': ocupacion, 'toca_borde': toca,
            'alertas_mascara': ';'.join(alertas)}


def preparar_cnn(rgb, mascara, lado=224):
    """Recortar con margen, preservar proporciones y rellenar a un cuadrado RGB."""
    mascara = validar_mascara(rgb, mascara)
    ys, xs = np.where(mascara > 0)
    margen = max(2, round(0.04 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)))
    x0, y0 = max(0, xs.min() - margen), max(0, ys.min() - margen)
    x1, y1 = min(rgb.shape[1], xs.max() + margen + 1), min(rgb.shape[0], ys.max() + margen + 1)
    recorte = rgb[y0:y1, x0:x1].copy()
    recorte[mascara[y0:y1, x0:x1] == 0] = 128
    h, w = recorte.shape[:2]
    escala = lado / max(h, w)
    ancho, alto = max(1, round(w * escala)), max(1, round(h * escala))
    interpolacion = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    reducido = cv2.resize(recorte, (ancho, alto), interpolation=interpolacion)
    salida = np.full((lado, lado, 3), 128, dtype=np.uint8)
    x, y = (lado - ancho) // 2, (lado - alto) // 2
    salida[y:y + alto, x:x + ancho] = reducido
    return salida, [int(x0), int(y0), int(x1), int(y1)]


def _textura(gris, mascara):
    """Coocurrencia de 16 niveles, solo con vecinos que pertenecen a la fruta."""
    cuantizada = np.minimum(gris.astype(np.int32) // 16, 15)
    matriz = np.zeros((16, 16), dtype=float)
    for eje in [0, 1]:
        if eje == 0:
            a, b = cuantizada[:-1], cuantizada[1:]
            dentro = mascara[:-1].astype(bool) & mascara[1:].astype(bool)
        else:
            a, b = cuantizada[:, :-1], cuantizada[:, 1:]
            dentro = mascara[:, :-1].astype(bool) & mascara[:, 1:].astype(bool)
        if dentro.any():
            matriz += np.bincount(16 * a[dentro] + b[dentro], minlength=256).reshape(16, 16)
    matriz += matriz.T
    if matriz.sum() == 0:
        raise ValueError('No hay suficientes pares de píxeles para calcular textura.')
    p = matriz / matriz.sum()
    i, j = np.indices(p.shape)
    positivos = p[p > 0]
    return {'textura_contraste': float((p * ((i-j) / 15)**2).sum()),
            'textura_homogeneidad': float((p / (1 + (i-j)**2)).sum()),
            'textura_energia': float((p**2).sum()),
            'textura_entropia': float(-(positivos * np.log2(positivos)).sum())}


def extraer_caracteristicas(rgb, mascara):
    """Color, textura y forma. No incluir fondo ni convertir oscuridad en calidad."""
    mascara = validar_mascara(rgb, mascara)
    seleccion = mascara.astype(bool)
    pixeles = rgb[seleccion].astype(float) / 255
    resultado = {}
    for indice, canal in enumerate(['r', 'g', 'b']):
        resultado[f'color_{canal}_media'] = float(pixeles[:, indice].mean())
        resultado[f'color_{canal}_desv'] = float(pixeles[:, indice].std())
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[seleccion].astype(float)
    angulo = 2 * np.pi * hsv[:, 0] / 180
    resultado.update(color_h_seno=float(np.sin(angulo).mean()),
                     color_h_coseno=float(np.cos(angulo).mean()))
    for indice, canal in enumerate(['h', 's', 'v']):
        maximo = 180 if canal == 'h' else 256
        hist, _ = np.histogram(hsv[:, indice], bins=8, range=(0, maximo))
        for k, valor in enumerate(hist / hist.sum()):
            resultado[f'color_hist_{canal}_{k}'] = float(valor)
    for indice, canal in [(1, 's'), (2, 'v')]:
        resultado[f'color_{canal}_media'] = float(hsv[:, indice].mean() / 255)
        resultado[f'color_{canal}_desv'] = float(hsv[:, indice].std() / 255)
    gris = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resultado.update(_textura(gris, mascara))
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contorno = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(contorno)
    perimetro = cv2.arcLength(contorno, True)
    casco = cv2.contourArea(cv2.convexHull(contorno))
    _, _, w, h = cv2.boundingRect(contorno)
    resultado.update(forma_circularidad=float(4 * np.pi * area / perimetro**2) if perimetro else 0.,
                     forma_solidez=float(area / casco) if casco else 0.,
                     forma_aspecto=float(max(w, h) / max(1, min(w, h))))
    diagnostico = diagnosticar_mascara(rgb, mascara)
    resultado['tam_ocupacion_imagen'] = diagnostico['ocupacion']
    resultado['tam_diametro_equivalente_normalizado'] = float(
        2 * np.sqrt(mascara.sum() / np.pi) / min(rgb.shape[:2]))
    if not np.isfinite(list(resultado.values())).all():
        raise ValueError('Las características contienen valores no finitos.')
    return resultado


def guardar_preparacion(rgb, mascara, carpeta, id_ejemplar, sha_original, metodo, parametros=None):
    """Guardar máscara a resolución original, entrada CNN y metadatos trazables."""
    mascara = validar_mascara(rgb, mascara)
    carpeta = Path(carpeta) / id_ejemplar / metodo
    carpeta.mkdir(parents=True, exist_ok=True)
    entrada, caja = preparar_cnn(rgb, mascara)
    ruta_mascara = carpeta / 'mascara.png'
    ruta_cnn = carpeta / 'entrada_cnn.png'
    for imagen, ruta in [(mascara * 255, ruta_mascara), (entrada, ruta_cnn)]:
        contenido = io.BytesIO()
        Image.fromarray(imagen).save(contenido, format='PNG')
        guardar_bytes_si_cambio(ruta, contenido.getvalue())
    sha_mascara = hashlib.sha256(ruta_mascara.read_bytes()).hexdigest()
    sha_cnn = hashlib.sha256(ruta_cnn.read_bytes()).hexdigest()
    config = {'version': VERSION, 'opencv': cv2.__version__, 'lado_cnn': 224,
              'metodo': metodo, 'parametros': parametros or {}, 'sha_original': sha_original,
              'sha_mascara': sha_mascara, 'sha_cnn': sha_cnn}
    firma = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    datos = {'id_ejemplar': id_ejemplar, 'metodo': metodo,
             'ruta_mascara': str(ruta_mascara.resolve()), 'ruta_cnn': str(ruta_cnn.resolve()),
             'sha_mascara': sha_mascara, 'firma_preparacion': firma,
             'caja_recorte': caja, **config, **diagnosticar_mascara(rgb, mascara),
             **extraer_caracteristicas(rgb, mascara)}
    guardar_bytes_si_cambio(carpeta / 'preparacion.json', json.dumps(datos, indent=2).encode('utf-8'))
    return datos


def construir_grupos(tabla):
    """Unir vínculos transitivos de ejemplar, foto de origen y archivos idénticos.

    Cada fila debe tener un grupo de ejemplar revisado. No reemplazar los grupos
    desconocidos por identificadores únicos: eso ocultaría la dependencia.
    """
    if tabla['grupo_ejemplar'].fillna('').str.strip().eq('').any():
        raise ValueError('Faltan grupos de ejemplar; completarlos antes de dividir los datos.')
    if not tabla['id_ejemplar'].is_unique:
        raise ValueError('Cada ejemplar debe aparecer una sola vez antes de construir grupos.')
    padres = list(range(len(tabla)))

    def raiz(i):
        while padres[i] != i:
            padres[i] = padres[padres[i]]
            i = padres[i]
        return i

    vistos = {}
    for i, fila in enumerate(tabla.to_dict('records')):
        for campo in ['grupo_ejemplar', 'id_origen', 'sha_original']:
            valor = str(fila.get(campo, '')).strip()
            if not valor:
                continue
            clave = (campo, valor)
            if clave in vistos:
                padres[raiz(i)] = raiz(vistos[clave])
            else:
                vistos[clave] = i
    miembros = {}
    for i, identificador in enumerate(tabla['id_ejemplar']):
        miembros.setdefault(raiz(i), []).append(identificador)
    claves = {r: hashlib.sha256('|'.join(sorted(ids)).encode()).hexdigest()[:16]
              for r, ids in miembros.items()}
    return np.array([claves[raiz(i)] for i in range(len(tabla))])


def particionar_por_grupos(tabla, origenes_explorados=(), semilla=42,
                           frutas=('manzana', 'banano', 'naranja')):
    """Plan 60/20/20 por grupos, con cobertura de fruta y calidad en las tres partes.

    Los grupos explorados quedan en entrenamiento. Para grupos que contienen más
    de un estrato se buscan particiones conjuntas; si no se logra cobertura, se
    informa la falta de una partición viable en lugar de separar fotos relacionadas.
    """
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit

    salida = tabla.copy()
    if salida.empty:
        raise ValueError('Todavía no hay ejemplares con etiquetas y máscaras confirmadas.')
    salida['grupo_particion'] = construir_grupos(salida)
    salida['estrato'] = salida['fruta'] + '|' + salida['calidad_revisada']
    esperados = {f'{f}|{c}' for f in frutas
                 for c in ['buena', 'intermedia', 'mala']}
    if set(salida['estrato']) != esperados:
        raise ValueError('Se necesitan las tres calidades de cada fruta antes de dividir.')
    conteos = salida.groupby('estrato')['grupo_particion'].nunique()
    if conteos.min() < 5:
        raise ValueError('Se requieren al menos cinco grupos por fruta y calidad para este plan. '
                         'Es un mínimo operativo, no garantía de suficiencia estadística.')
    fijos = set(salida.loc[salida['id_origen'].isin(set(origenes_explorados)), 'grupo_particion'])
    grupos = salida.groupby('grupo_particion')['estrato'].agg(lambda x: sorted(set(x)))
    rng = np.random.default_rng(semilla)
    asignaciones = {g: 'entrenamiento' for g in fijos}
    if grupos.map(len).eq(1).all():
        estrato_grupo = grupos.map(lambda x: x[0])
        for estrato in sorted(esperados):
            todos = estrato_grupo.index[estrato_grupo.eq(estrato)].tolist()
            disponibles = [g for g in todos if g not in fijos]
            rng.shuffle(disponibles)
            n = max(1, round(0.20 * len(todos)))
            if len(disponibles) < 2 * n:
                raise ValueError('Quedan pocos grupos no explorados para validación y prueba.')
            for i, g in enumerate(disponibles):
                asignaciones[g] = 'validacion' if i < n else 'prueba' if i < 2*n else 'entrenamiento'
    else:
        libres = salida.loc[~salida['grupo_particion'].isin(fijos)].copy()
        n_total = salida['grupo_particion'].nunique()
        n_holdout = max(2, round(0.40 * n_total))
        if libres['grupo_particion'].nunique() <= n_holdout:
            raise ValueError('No quedan grupos suficientes fuera de la exploración.')
        candidatos = GroupShuffleSplit(n_splits=2000, test_size=n_holdout, random_state=semilla)
        mejor = None
        mejor_error = float('inf')
        for intento, (_, indices) in enumerate(candidatos.split(libres, groups=libres['grupo_particion'])):
            holdout = libres.iloc[indices]
            divisor = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=semilla + intento)
            iv, ip = next(divisor.split(holdout, groups=holdout['grupo_particion']))
            val = set(holdout.iloc[iv]['grupo_particion'])
            prueba = set(holdout.iloc[ip]['grupo_particion'])
            etiquetas = salida['grupo_particion'].map(
                lambda g: 'validacion' if g in val else 'prueba' if g in prueba else 'entrenamiento')
            cobertura = pd.crosstab(etiquetas, salida['estrato']).reindex(
                index=['entrenamiento', 'validacion', 'prueba'], columns=sorted(esperados), fill_value=0)
            if not cobertura.gt(0).all().all():
                continue
            proporciones = etiquetas.value_counts(normalize=True).reindex(
                ['entrenamiento', 'validacion', 'prueba'], fill_value=0)
            error = np.abs(proporciones.to_numpy() - [0.6, 0.2, 0.2]).sum()
            if error < mejor_error:
                mejor = etiquetas
                mejor_error = error
                if error < 0.01:
                    break
        if mejor is None:
            raise ValueError('No se encontró una partición agrupada con cobertura completa; '
                             'revisar grupos y reunir más ejemplares independientes.')
        salida['particion'] = mejor
    if 'particion' not in salida:
        salida['particion'] = salida['grupo_particion'].map(asignaciones)
    assert salida.groupby('grupo_particion')['particion'].nunique().eq(1).all()
    assert salida.loc[salida['grupo_particion'].isin(fijos), 'particion'].eq('entrenamiento').all()
    return salida
