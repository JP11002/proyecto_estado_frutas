"""Inventario de originales de Drive y galerías de revisión, sin aprobar etiquetas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from .preparacion_imagenes import guardar_tabla


def abrir_foto(path):
    if path.suffix.lower() in {'.heic','.heif'}:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    return Image.open(path)


def inventario_drive(proyecto):
    proyecto=Path(proyecto).resolve();folder=proyecto/'data/annotations/drive_20261124'
    items=json.loads((folder/'listado_drive.json').read_text(encoding='utf-8'))['archivos']
    catalog=pd.DataFrame(items).sort_values(['fuente','fruta','ruta_drive','drive_id']).reset_index(drop=True)
    catalog['numero_revision']=catalog.groupby(['fuente','fruta']).cumcount()+1
    catalog['codigo_revision']=[f'{"pub" if f=="publica" else "eq"}_{t}_{n:03}' for f,t,n in
        zip(catalog.fuente,catalog.fruta,catalog.numero_revision)]
    cache=proyecto/'data/processed/drive_20261124/miniaturas';cache.mkdir(parents=True,exist_ok=True)
    previous_path=folder/'inventario_drive.csv'
    previous=pd.read_csv(previous_path,keep_default_na=False).set_index('id_imagen') if previous_path.is_file() else pd.DataFrame()
    rows=[]
    for item in catalog.to_dict('records'):
        path=proyecto/item['ruta_local'];thumb=cache/(item['id_imagen']+'.jpg')
        result={**item,'descargado':path.is_file(),'archivo_legible':False,'sha256':'',
                'ancho':0,'alto':0,'fecha_exif':'','sha_pixeles':'','ruta_miniatura':'','error':''}
        if path.is_file():
            sha=hashlib.sha256(path.read_bytes()).hexdigest();result['sha256']=sha
            if not previous.empty and item['id_imagen'] in previous.index:
                old=previous.loc[item['id_imagen']]
                if old.sha256==sha and str(old.archivo_legible).lower()=='true' and thumb.is_file():
                    rows.append({**item,**{k:old[k] for k in result if k not in item}});continue
            try:
                with abrir_foto(path) as image:
                    exif=image.getexif()
                    date=exif.get(36867) or exif.get(306) or ''
                    # DateTimeOriginal suele estar dentro del sub-IFD EXIF.
                    if not date:
                        try:date=exif.get_ifd(34665).get(36867,'')
                        except (KeyError,ValueError):pass
                    image=ImageOps.exif_transpose(image).convert('RGB');image.load()
                    result.update(ancho=image.width,alto=image.height,fecha_exif=str(date),
                        sha_pixeles=hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),archivo_legible=True)
                    image.thumbnail((420,420),Image.Resampling.LANCZOS)
                    if not thumb.is_file():image.save(thumb,quality=90)
                    result['ruta_miniatura']=str(thumb.resolve())
            except Exception as error:result['error']=type(error).__name__+': '+str(error)
        rows.append(result)
    data=pd.DataFrame(rows)
    available=data.loc[data.archivo_legible].copy()
    counts=available.groupby('sha256').size()
    pixels=available.groupby('sha_pixeles').size()
    data['repeticiones_bytes']=data.sha256.map(counts).fillna(0).astype(int)
    data['repeticiones_pixeles']=data.sha_pixeles.map(pixels).fillna(0).astype(int)
    original=proyecto/'experiments/results/revision_datos/inventario_imagenes.csv'
    old_sha=set(pd.read_csv(original,usecols=['sha256']).sha256) if original.is_file() else set()
    data['duplicado_exacto_kaggle']=data.sha256.isin(old_sha)
    guardar_tabla(data,previous_path)
    return data


def galerias(proyecto,data,por_pagina=20):
    destination=Path(proyecto)/'experiments/results/integracion_drive/galerias'
    destination.mkdir(parents=True,exist_ok=True)
    pages=[]
    for (source,fruit),group in data.loc[data.archivo_legible].groupby(['fuente','fruta']):
        group=group.sort_values('numero_revision')
        for offset in range(0,len(group),por_pagina):
            subset=group.iloc[offset:offset+por_pagina]
            canvas=Image.new('RGB',(1500,5*310+55),'#fafafa');draw=ImageDraw.Draw(canvas)
            draw.text((15,10),f'{source.upper()} / {fruit.upper()} / etiquetas de carpeta sin confirmar',fill='black')
            for i,row in enumerate(subset.itertuples()):
                with Image.open(row.ruta_miniatura) as image:
                    image=ImageOps.contain(image,(285,260))
                    x=(i%5)*300+(300-image.width)//2;y=(i//5)*310+35
                    canvas.paste(image,(x,y))
                    draw.text(((i%5)*300+8,(i//5)*310+298),row.codigo_revision+' | '+row.calidad_declarada,fill='black')
            path=destination/f'{source}_{fruit}_{offset//por_pagina+1:02}.jpg'
            # Cada página nueva se guarda una sola vez; la galería se ejecuta tras completar la descarga.
            canvas.save(path,quality=91)
            pages.append({'ruta':str(path.resolve()),'imagenes':subset.id_imagen.tolist()})
    (destination/'indice.json').write_text(json.dumps(pages,indent=2),encoding='utf-8')
    return pages


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proyecto',default='.')
    parser.add_argument('--galerias',action='store_true');args=parser.parse_args()
    data=inventario_drive(args.proyecto)
    print(data.groupby(['fuente','fruta']).agg(descubiertas=('id_imagen','size'),
        descargadas=('descargado','sum'),legibles=('archivo_legible','sum')).to_string(),flush=True)
    print('Duplicados por bytes:',int(data.loc[data.archivo_legible,'sha256'].duplicated().sum()),flush=True)
    if args.galerias:print('Galerías:',len(galerias(args.proyecto,data)),flush=True)


if __name__=='__main__':main()
