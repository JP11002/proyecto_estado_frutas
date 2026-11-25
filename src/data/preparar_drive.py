"""Preparar propuestas de máscara de un lote explícito; no aprobarlas al calcularlas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageOps

from .preparacion_imagenes import (
    leer_rgb,mascara_grabcut,mascara_poligono,guardar_preparacion,guardar_tabla,
)

METODO='drive_color_grabcut_v1'


def proponer_mascara(rgb,fruta,fuente,parametros):
    if 'puntos' in parametros:
        return mascara_poligono(rgb,parametros['puntos'])
    if parametros.get('semilla_color_v2'):
        h,w=rgb.shape[:2];scale=min(1.,800/max(h,w))
        small=cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
        hsv=cv2.cvtColor(small,cv2.COLOR_RGB2HSV)
        if fruta=='banano' and fuente=='propia':
            seed=((hsv[:,:,2]<100)&(hsv[:,:,1]>40)).astype(np.uint8)
        else:
            seed=((hsv[:,:,1]>100)&(hsv[:,:,2]>120)).astype(np.uint8)
            if fruta=='banano':seed&=((hsv[:,:,0]>=10)&(hsv[:,:,0]<=95)).astype(np.uint8)
            if fuente=='propia' and fruta=='manzana':
                seed&=((small[:,:,0].astype(float)/(small[:,:,2]+1.)<2.6)&(hsv[:,:,2]>145)).astype(np.uint8)
        if 'rectangulo' in parametros:
            x0,y0,x1,y1=parametros['rectangulo'];inside=np.zeros_like(seed)
            inside[round(y0*seed.shape[0]):round(y1*seed.shape[0]),round(x0*seed.shape[1]):round(x1*seed.shape[1])]=1
            seed&=inside
        contours,_=cv2.findContours(seed,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:raise ValueError('No se encontró una semilla de color.')
        seed=np.zeros_like(seed);cv2.drawContours(seed,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
        ys,xs=np.where(seed);margin=max(8,round(.3*max(xs.max()-xs.min(),ys.max()-ys.min())))
        labels=np.zeros(seed.shape,np.uint8)
        labels[max(0,ys.min()-margin):min(seed.shape[0],ys.max()+margin+1),
               max(0,xs.min()-margin):min(seed.shape[1],xs.max()+margin+1)]=cv2.GC_PR_BGD
        labels[seed>0]=cv2.GC_PR_FGD
        eroded=cv2.erode(seed,np.ones((5,5),np.uint8));labels[eroded>0]=cv2.GC_FGD
        cv2.setRNGSeed(42)
        cv2.grabCut(cv2.cvtColor(small,cv2.COLOR_RGB2BGR),labels,None,
                    np.zeros((1,65)),np.zeros((1,65)),4,cv2.GC_INIT_WITH_MASK)
        result=np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
        contours,_=cv2.findContours(result,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        result[:]=0;cv2.drawContours(result,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
        return cv2.resize(result,(w,h),interpolation=cv2.INTER_NEAREST)
    if fuente=='propia' and fruta!='limon':
        return mascara_grabcut(rgb,parametros['rectangulo'],lado_maximo=700)
    h,w=rgb.shape[:2];scale=min(1.,800/max(h,w))
    small=cv2.resize(rgb,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(small,cv2.COLOR_RGB2HSV)
    mask=((hsv[:,:,1]>65)&(hsv[:,:,2]>35)).astype(np.uint8)
    if fruta=='limon':mask&=((hsv[:,:,0]>=25)&(hsv[:,:,0]<=100)).astype(np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:raise ValueError('La propuesta de color no encontró una región.')
    result=np.zeros_like(mask)
    cv2.drawContours(result,[max(contours,key=cv2.contourArea)],-1,1,cv2.FILLED)
    return cv2.resize(result,(w,h),interpolation=cv2.INTER_NEAREST)


def preparar(proyecto):
    proyecto=Path(proyecto).resolve();annotations=proyecto/'data/annotations/drive_20261124'
    catalog=pd.read_csv(annotations/'inventario_drive.csv',keep_default_na=False).set_index('codigo_revision')
    manifest=json.loads((annotations/'revision_visual_v2.json').read_text(encoding='utf-8'))
    cv2.setNumThreads(1);records=[]
    for review in manifest['lote_detallado']:
        if review.get('decision_imagen')=='excluir_foto_multiple':
            continue
        source=catalog.loc[review['codigo_revision']]
        params=review.get('parametros',{})
        method=params.get('metodo',METODO)
        cached=proyecto/'data/processed/drive_20261124/preparacion'/source.id_imagen/method/'preparacion.json'
        if hashlib.sha256((proyecto/source.ruta_local).read_bytes()).hexdigest()!=source.sha256:
            raise ValueError('Cambió un original respecto al inventario.')
        if cached.is_file():
            prepared=json.loads(cached.read_text(encoding='utf-8'))
            if prepared['sha_original']!=source.sha256 or prepared['parametros']!=params:
                raise ValueError('La receta cambió; usar una versión de método nueva.')
            for path,sha in [(prepared['ruta_mascara'],prepared['sha_mascara']),
                             (prepared['ruta_cnn'],prepared['sha_cnn'])]:
                if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('Cambió una preparación.')
        else:
            rgb=leer_rgb(proyecto/source.ruta_local)
            mask=proponer_mascara(rgb,source.fruta,source.fuente,params)
            prepared=guardar_preparacion(rgb,mask,proyecto/'data/processed/drive_20261124/preparacion',
                source.id_imagen,source.sha256,method,params)
        records.append({**source.to_dict(),**review,**prepared,'id_origen':source.id_imagen})
        if len(records)%10==0:print('Propuestas preparadas:',len(records),flush=True)
    data=pd.DataFrame(records)
    guardar_tabla(data.drop(columns=['parametros','caja_recorte'],errors='ignore'),
                   proyecto/'experiments/results/integracion_drive/propuestas_mascaras.csv')
    return data


def galerias_mascaras(proyecto,data):
    folder=Path(proyecto)/'experiments/results/integracion_drive/mascaras';folder.mkdir(parents=True,exist_ok=True)
    for (source,fruit),group in data.groupby(['fuente','fruta']):
        for offset in range(0,len(group),8):
            sample=group.iloc[offset:offset+8]
            canvas=Image.new('RGB',(1500,4*370+45),'white');draw=ImageDraw.Draw(canvas)
            draw.text((12,10),f'{source} / {fruit}: original con contorno y recorte — propuestas sin aprobar',fill='black')
            for i,row in enumerate(sample.itertuples()):
                with Image.open(row.ruta_miniatura) as im:original=im.convert('RGB')
                with Image.open(row.ruta_mascara) as im:
                    mask=np.asarray(im.resize(original.size,Image.Resampling.NEAREST))>0
                overlay=np.asarray(original).copy()
                contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay,contours,-1,(255,20,70),2)
                panel=ImageOps.contain(Image.fromarray(overlay),(350,310))
                with Image.open(row.ruta_cnn) as im:crop=im.resize((310,310))
                x=(i%2)*750;y=(i//2)*370+35
                canvas.paste(panel,(x+(350-panel.width)//2,y))
                canvas.paste(crop,(x+370,y))
                draw.text((x+10,y+320),row.codigo_revision+' | propuesta calidad: '+row.calidad_propuesta,fill='black')
            canvas.save(folder/f'{source}_{fruit}_{offset//8+1:02}.jpg',quality=92)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proyecto',default='.')
    args=parser.parse_args();data=preparar(args.proyecto);galerias_mascaras(args.proyecto,data)
    print('Propuestas disponibles para inspección:',len(data),flush=True)


if __name__=='__main__':main()
