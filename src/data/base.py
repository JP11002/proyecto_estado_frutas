"""Lectura, validación y actualización de la única base del proyecto: data/base/."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath

import pandas as pd
import numpy as np

from src.utils.archivos import guardar_bytes_si_cambio, guardar_tabla


def resolver_ruta(proyecto: str | Path, valor: str) -> Path:
    """Resolver rutas relativas y reubicar rutas históricas de Windows/POSIX.

    La reubicación solo permite las carpetas data/ y experiments/ del proyecto.
    No depende del usuario, la letra del disco o el nombre de la carpeta clonada.
    """
    proyecto = Path(proyecto).resolve()
    texto = str(valor).replace('\\', '/')
    ruta = Path(texto)
    absoluta = texto.startswith('/') or ruta.is_absolute() or PureWindowsPath(texto).is_absolute()
    if absoluta:
        if ruta.is_absolute() and ruta.is_relative_to(proyecto):
            candidata = ruta
        else:
            partes = texto.split('/')
            posiciones = [i for i, parte in enumerate(partes) if parte in {'data', 'experiments'}]
            if not posiciones:
                raise ValueError(f'Ruta externa sin una subcarpeta del proyecto: {valor}')
            candidata = proyecto.joinpath(*partes[posiciones[-1]:])
    else:
        candidata = proyecto / ruta
    candidata = candidata.resolve()
    if not candidata.is_relative_to(proyecto):
        raise ValueError(f'La ruta sale del proyecto: {valor}')
    return candidata


def cargar_archivos_base(proyecto: str | Path):
    """Cargar exclusivamente data/base/datos.csv; no hay fuentes alternativas."""
    proyecto = Path(proyecto).resolve()
    ruta = proyecto / 'data/base/datos.csv'
    if not ruta.is_file():
        raise FileNotFoundError('Falta data/base/datos.csv. Recuperar la base incluida en el repositorio.')
    estado = json.loads(ruta.with_name('estado.json').read_text(encoding='utf-8'))
    if hashlib.sha256(ruta.read_bytes()).hexdigest() != estado['sha_base']:
        raise ValueError('La base no coincide con su huella. No editar el CSV manualmente; reexportar la base revisada.')
    base = pd.read_csv(ruta)
    for columna in ['ruta_original', 'ruta_mascara', 'ruta_cnn']:
        if columna in base:
            base[columna] = [str(resolver_ruta(proyecto, valor)) for valor in base[columna]]
    return base, estado['caracteristicas'], estado


def cargar_base(proyecto: str | Path):
    """Entrada compartida por EDA, modelos tradicionales y CNN."""
    base, columnas, estado = cargar_archivos_base(proyecto)
    if len(columnas) != 43 or not all(c.startswith(('color_', 'textura_', 'forma_')) for c in columnas):
        raise ValueError('Se esperan las 43 características de imagen; no incluir identificadores.')
    if not np.isfinite(base[columnas].to_numpy(dtype=float)).all():
        raise ValueError('Hay características faltantes o no finitas.')
    if base[['id_ejemplar', 'grupo_particion', 'calidad_revisada', 'fruta']].isna().any().any():
        raise ValueError('Hay etiquetas o grupos faltantes.')
    if base.id_ejemplar.duplicated().any() or set(base.calidad_revisada) != {'buena', 'intermedia', 'mala'}:
        raise ValueError('Revisar identificadores únicos y las tres calidades.')
    if set(base.fruta) != {'manzana', 'banano', 'limon'}:
        raise ValueError('Revisar el alcance de frutas.')
    return base, columnas, estado


def guardar_base(proyecto, tabla, estado):
    """Actualizar la misma base con entradas ya revisadas, sin crear otra copia."""
    proyecto = Path(proyecto).resolve()
    destino = proyecto / 'data/base'
    registros = tabla.copy()
    for indice, fila in registros.iterrows():
        for campo, huella, carpeta in [('ruta_cnn', 'sha_cnn', 'imagenes'),
                                      ('ruta_mascara', 'sha_mascara', 'mascaras')]:
            origen = resolver_ruta(proyecto, fila[campo])
            contenido = origen.read_bytes()
            if hashlib.sha256(contenido).hexdigest() != fila[huella]:
                raise ValueError(f'Cambió {campo} de {fila.id_ejemplar}.')
            archivo = destino / carpeta / (fila.id_ejemplar + '.png')
            guardar_bytes_si_cambio(archivo, contenido)
            registros.at[indice, campo] = archivo.relative_to(proyecto).as_posix()
        registros.at[indice, 'ruta_original'] = resolver_ruta(proyecto, fila.ruta_original).relative_to(proyecto).as_posix()
    guardar_tabla(registros, destino / 'datos.csv')
    estado = {k: v for k, v in estado.items() if k not in {'version', 'portabilidad', 'origen_sha_base'}}
    estado['sha_base'] = hashlib.sha256((destino / 'datos.csv').read_bytes()).hexdigest()
    guardar_bytes_si_cambio(destino / 'estado.json', json.dumps(estado, ensure_ascii=False, indent=2).encode('utf-8'))
    return cargar_base(proyecto)


def verificar_imagenes(base):
    """Comprobar los recortes necesarios para CNN y las máscaras revisadas."""
    for fila in base.itertuples():
        for campo, huella in [('ruta_cnn', 'sha_cnn'), ('ruta_mascara', 'sha_mascara')]:
            ruta = Path(getattr(fila, campo))
            if not ruta.is_file():
                raise FileNotFoundError(f'Falta {campo} de {fila.id_ejemplar}: {ruta}')
            if hashlib.sha256(ruta.read_bytes()).hexdigest() != getattr(fila, huella):
                raise ValueError(f'Cambió {campo} de {fila.id_ejemplar}.')
