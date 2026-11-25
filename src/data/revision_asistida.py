"""Reconstruir la revisión visual registrada en revision_asistente_v1.json.

La calidad procede de las anotaciones visuales, no de un umbral automático. Una
región no se exporta como máscara aceptada hasta que el registro lo indique.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from .preparacion_imagenes import (
    VERSION, guardar_bytes_si_cambio, guardar_preparacion, guardar_tabla, leer_rgb,
)

REVISOR = 'Asistente - revision visual'
METODO = 'revision_asistente_v5'


def contorno_anotado(rgb, region):
    """Convertir el contorno aproximado anotado a una semilla binaria."""
    h, w = rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if region['forma'] == 'elipse':
        x0, y0, x1, y1 = region['caja']
        if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
            raise ValueError('Caja fuera de la imagen.')
        centro = (round((x0+x1)*w/2), round((y0+y1)*h/2))
        radios = (max(1, round((x1-x0)*w/2)), max(1, round((y1-y0)*h/2)))
        cv2.ellipse(mask, centro, radios, 0, 0, 360, 1, -1)
    elif region['forma'] == 'trazo':
        puntos = np.asarray(region['puntos'], dtype=float)
        radios = np.asarray(region['radios'], dtype=float)
        if puntos.shape != (len(radios), 2) or len(radios)<2 or not np.isfinite(puntos).all():
            raise ValueError('Trazo inválido.')
        if (puntos<0).any() or (puntos>1).any() or (radios<=0).any():
            raise ValueError('Coordenadas o radios inválidos.')
        distancias = np.r_[0, np.cumsum(np.linalg.norm(np.diff(puntos,axis=0),axis=1))]
        if (np.diff(distancias)<=0).any():
            raise ValueError('El trazo contiene puntos consecutivos idénticos.')
        t = np.linspace(0, distancias[-1], 160)
        xy = PchipInterpolator(distancias, puntos)(t)
        r = PchipInterpolator(distancias, radios)(t)
        for (x,y),radius in zip(xy,r):
            cv2.circle(mask,(round(x*(w-1)),round(y*(h-1))),max(1,round(radius*min(h,w))),1,-1)
    elif region['forma'] == 'poligono':
        points=np.asarray(region['puntos'],dtype=float)
        cv2.fillPoly(mask,[np.rint(points*[w-1,h-1]).astype(np.int32)],1)
    else:
        raise ValueError('Tipo de contorno desconocido.')
    if mask.sum()<25:
        raise ValueError('Región demasiado pequeña.')
    return mask


def refinar_contorno(rgb, region):
    """Refinar con GrabCut la semilla manual, manteniendo sus huecos rellenos."""
    seed = contorno_anotado(rgb,region)
    if region.get('usar_contorno_directo',False):
        if region.get('limpiar_fondo_neutro',False):
            hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
            color=(hsv[:,:,1]>45)&(hsv[:,:,0]>=8)&(hsv[:,:,0]<=85)
            propuesta=(seed.astype(bool)&color).astype(np.uint8)
            contours,_=cv2.findContours(propuesta,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                raise ValueError('La limpieza dejó una región vacía.')
            seed=np.zeros_like(seed)
            cv2.drawContours(seed,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
        return seed
    if region.get('fondo_cian',False):
        hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
        fondo=(hsv[:,:,0]>=75)&(hsv[:,:,0]<=125)&(hsv[:,:,1]>60)
        mask=(~fondo).astype(np.uint8)
        n,labels,stats,centros=cv2.connectedComponentsWithStats(mask)
        anchor=region.get('centro_componente')
        if anchor is None:
            ys,xs=np.where(seed>0);anchor=[float(xs.mean()/rgb.shape[1]),float(ys.mean()/rgb.shape[0])]
        candidatos=[k for k in range(1,n) if stats[k,cv2.CC_STAT_AREA]>=40
                    and not (stats[k,cv2.CC_STAT_WIDTH]>.9*rgb.shape[1])]
        if not candidatos:
            raise ValueError('No hay componente de fruta con el fondo cian propuesto.')
        chosen=min(candidatos,key=lambda k:np.linalg.norm(centros[k]/[rgb.shape[1],rgb.shape[0]]-anchor))
        contours,_=cv2.findContours((labels==chosen).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        result=np.zeros_like(seed)
        cv2.drawContours(result,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
        return result
    margen = max(3,round(min(seed.shape)*0.025))
    expanded=cv2.dilate(seed,np.ones((2*margen+1,2*margen+1),np.uint8))
    labels=np.zeros(seed.shape,np.uint8)
    labels[expanded>0]=cv2.GC_PR_BGD
    labels[seed>0]=cv2.GC_PR_FGD
    distance=cv2.distanceTransform(seed,cv2.DIST_L2,3)
    labels[cv2.erode(seed,np.ones((5,5),np.uint8))>0]=cv2.GC_FGD
    cv2.setRNGSeed(42)
    cv2.grabCut(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),labels,None,
                np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_MASK)
    mask=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('El refinamiento no produjo un contorno.')
    result=np.zeros_like(mask)
    cv2.drawContours(result,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
    return result


def preparar_revision(proyecto):
    proyecto=Path(proyecto).resolve()
    annotations=proyecto/'data/annotations'
    manifest=json.loads((annotations/'revision_asistente_v1.json').read_text(encoding='utf-8'))
    inventory=pd.read_csv(proyecto/'experiments/results/revision_datos/inventario_imagenes.csv').set_index('id_imagen')
    records=[]
    cv2.setNumThreads(1)
    for region in manifest['ejemplares_propuestos']:
        source=inventory.loc[region['id_origen']]
        path=proyecto/'data/raw/fruit-quality-classification'/source.ruta_relativa
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source.sha256:
            raise ValueError('La fotografía cambió respecto al inventario.')
        rgb=leer_rgb(path)
        example_id=region['id_origen']+('_'+region['id_revision'] if region['es_hijo'] else '')
        params={k:v for k,v in region.items() if k in ['forma','caja','puntos','radios','usar_contorno_directo','fondo_cian','centro_componente','limpiar_fondo_neutro']}
        record_path=proyecto/'data/processed'/VERSION/example_id/METODO/'preparacion.json'
        if record_path.is_file():
            previous=json.loads(record_path.read_text(encoding='utf-8'))
            if previous['parametros']==params and previous['sha_original']==source.sha256:
                for field,key in [('ruta_mascara','sha_mascara'),('ruta_cnn','sha_cnn')]:
                    if hashlib.sha256(Path(previous[field]).read_bytes()).hexdigest()!=previous[key]:
                        raise ValueError('Cambió un archivo preparado; revisar antes de reutilizarlo.')
                records.append({**region,**previous,'ruta_relativa':source.ruta_relativa})
                continue
        mask=refinar_contorno(rgb,region)
        data=guardar_preparacion(rgb,mask,proyecto/'data/processed'/VERSION,example_id,
                                 source.sha256,METODO,params)
        records.append({**region,**data,'ruta_relativa':source.ruta_relativa})
    result=pd.DataFrame(records)
    out=proyecto/'experiments/results/revision_asistida'
    out.mkdir(parents=True,exist_ok=True)
    guardar_tabla(result.drop(columns=['parametros','caja_recorte'],errors='ignore'),out/'candidatos.csv')
    print('Ejemplares preparados para comprobar visualmente:',len(result))
    return manifest,result


def _combinar_revisiones(ruta,nuevas,clave):
    """Preservar revisiones ajenas al asistente y actualizar sus propias filas."""
    nuevas=pd.DataFrame(nuevas)
    if nuevas.empty:
        return
    if ruta.is_file():
        anterior=pd.read_csv(ruta,dtype=str,keep_default_na=False)
        propias=anterior['revisor'].eq(REVISOR)
        protegidas=set(anterior.loc[~propias,clave])
        nuevas=nuevas.loc[~nuevas[clave].isin(protegidas)]
        anterior=anterior.loc[~anterior[clave].isin(nuevas[clave])]
        nuevas=pd.concat([anterior,nuevas],ignore_index=True)
    assert nuevas[clave].is_unique
    guardar_tabla(nuevas,ruta)


def guardar_revisiones(proyecto,manifest,candidatos):
    proyecto=Path(proyecto).resolve(); annotations=proyecto/'data/annotations'
    stamp=datetime.now(timezone.utc).isoformat()
    quality=[]
    for review in manifest['originales_revisados']:
        quality.append({k:review.get(k,'') for k in ['id_imagen','sha256','calidad_revisada',
            'decision','manzana_roja','ejemplar_individual','grupo_ejemplar','observaciones','revisor']}
            | {'fecha_revision_utc':stamp})
    masks=[]; children=[]
    for row in candidatos.to_dict('records'):
        decision=row['decision_mascara']
        if decision not in ['aceptada','rechazada']:
            continue
        note=row.get('nota_mascara','Contorno comprobado visualmente sobre la imagen original y el recorte.')
        masks.append({k:row[k] for k in ['id_ejemplar','metodo','sha_original','sha_mascara','firma_preparacion']}
            | {'decision_mascara':decision,'observaciones':note,'revisor':REVISOR,'fecha_revision_utc':stamp})
        if decision!='aceptada':
            continue
        folder=annotations/'mascaras_revisadas'/row['id_ejemplar']/row['sha_mascara']
        guardar_bytes_si_cambio(folder/'mascara.png',Path(row['ruta_mascara']).read_bytes())
        metadata=Path(row['ruta_mascara']).parent/'preparacion.json'
        guardar_bytes_si_cambio(folder/'receta.json',metadata.read_bytes())
        if row['es_hijo']:
            children.append({'id_ejemplar':row['id_ejemplar'],'id_origen':row['id_origen'],
                'sha_original':row['sha_original'],'fruta':row['fruta'],'ruta_relativa':row['ruta_relativa'],
                'calidad_revisada':row['calidad'],'manzana_roja':row['roja'],
                'ejemplar_individual':'si','grupo_ejemplar':row['grupo_ejemplar'],
                'revisor':REVISOR,'observaciones':row['observaciones'],'fecha_revision_utc':stamp})
    _combinar_revisiones(annotations/'revision_equipo.csv',quality,'id_imagen')
    _combinar_revisiones(annotations/'revision_mascaras.csv',masks,'id_ejemplar')
    _combinar_revisiones(annotations/'ejemplares_segmentados.csv',children,'id_ejemplar')
    print('Decisiones visuales guardadas:',len(quality),'| Máscaras aceptadas:',sum(r['decision_mascara']=='aceptada' for r in masks))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proyecto',default='.')
    parser.add_argument('--guardar-revisiones',action='store_true')
    args=parser.parse_args()
    manifest,result=preparar_revision(args.proyecto)
    if args.guardar_revisiones:
        guardar_revisiones(args.proyecto,manifest,result)


if __name__=='__main__':
    main()
