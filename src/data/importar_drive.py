"""Importación reproducible de las dos carpetas compartidas, sin credenciales.

Conserva originales, ruta en Drive, identidad del archivo y etiqueta declarada.
La etiqueta de una carpeta nunca se considera una revisión visual por sí sola.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen

FUENTES = [
    {'id':'12Wcfr_mRLxJ98EpHR9WwbMd7jp49kZxK','tipo':'publica',
     'carpetas':{'Apple':'manzana','Banana':'banano','Lime':'limon'}},
    {'id':'1AyydHBltsfnj9YC4-F9xi9AeogtiK6Fx','tipo':'propia',
     'carpetas':{'Manzanas':'manzana','Bananos':'banano','Limones':'limon'}},
]
CALIDADES={'Buenos':'buena','Buenas':'buena','Intermedio':'intermedia',
           'Intermedios':'intermedia','Malos':'mala','Malas':'mala'}


class EnlacesCarpeta(HTMLParser):
    def __init__(self):
        super().__init__();self.enlaces=[];self.actual=None

    def handle_starttag(self,tag,attrs):
        if tag=='a':
            href=dict(attrs).get('href','')
            match=re.match(r'https://drive\.google\.com/(file/d/|drive/folders/)([-\w]+)',href)
            if match:self.actual={'id':match[2],'es_carpeta':match[1]=='drive/folders/','nombre':''}

    def handle_data(self,data):
        if self.actual is not None:self.actual['nombre']+=data

    def handle_endtag(self,tag):
        if tag=='a' and self.actual is not None:
            self.actual['nombre']=self.actual['nombre'].strip()
            self.enlaces.append(self.actual);self.actual=None


def listar_carpeta(identificador,cache):
    path=cache/(identificador+'.html')
    if not path.is_file():
        request=Request('https://drive.google.com/embeddedfolderview?'+urlencode({'id':identificador}),
                        headers={'User-Agent':'Mozilla/5.0'})
        with urlopen(request,timeout=60) as response:payload=response.read()
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    parser=EnlacesCarpeta();parser.feed(path.read_text(encoding='utf-8'))
    if 'accounts.google.com/ServiceLogin' in path.read_text(encoding='utf-8') and not parser.enlaces:
        raise ValueError('La carpeta requiere iniciar sesión: '+identificador)
    return parser.enlaces


def inventariar(proyecto):
    cache=proyecto/'data/external/drive_20261124/listados'
    archivos=[];carpetas=[]
    def recorrer(item,fuente,ruta,fruta):
        hijos=listar_carpeta(item['id'],cache)
        carpetas.append({'fuente':fuente['tipo'],'id':item['id'],'ruta':ruta,'elementos':len(hijos)})
        for hijo in hijos:
            nombre=hijo['nombre']
            if hijo['es_carpeta']:
                recorrer(hijo,fuente,ruta+'/'+nombre,fruta)
            elif Path(nombre).suffix.lower() in {'.jpg','.jpeg','.png','.heic','.heif','.webp','.bmp'}:
                calidad=next((CALIDADES[p] for p in ruta.split('/') if p in CALIDADES),'sin_etiqueta')
                seguro=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',nombre)
                relative=Path('data/raw/drive_20261124')/fuente['tipo']/fruta/(hijo['id']+'__'+seguro)
                archivos.append({'id_imagen':'drive_'+hijo['id'],'drive_id':hijo['id'],
                    'fuente':fuente['tipo'],'carpeta_raiz_id':fuente['id'],
                    'url_fuente':'https://drive.google.com/file/d/'+hijo['id']+'/view',
                    'ruta_drive':ruta+'/'+nombre,'nombre_original':nombre,'fruta':fruta,
                    'calidad_declarada':calidad,'ruta_local':relative.as_posix()})
    for fuente in FUENTES:
        items=listar_carpeta(fuente['id'],cache)
        for item in items:
            if item['es_carpeta'] and item['nombre'] in fuente['carpetas']:
                recorrer(item,fuente,item['nombre'],fuente['carpetas'][item['nombre']])
    return archivos,carpetas


def descargar_archivo(proyecto,item):
    path=(proyecto/item['ruta_local']).resolve()
    if not path.is_relative_to(proyecto/'data/raw/drive_20261124'):
        raise ValueError('Ruta de descarga fuera del destino autorizado.')
    reused=path.is_file()
    if not reused:
        url='https://drive.usercontent.google.com/download?'+urlencode(
            {'id':item['drive_id'],'export':'download','confirm':'t'})
        request=Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urlopen(request,timeout=90) as response:
            content_type=response.headers.get('Content-Type','')
            payload=response.read()
        if 'text/html' in content_type or not payload:
            raise ValueError('Drive no entregó el archivo original (confirmación o cuota).')
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.part');temporary.write_bytes(payload);temporary.replace(path)
    return {**item,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'descargado':True,'reutilizado':reused,'error':''}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proyecto',default='.')
    parser.add_argument('--solo-listar',action='store_true')
    args=parser.parse_args();project=Path(args.proyecto).resolve()
    files,folders=inventariar(project)
    destination=project/'data/annotations/drive_20261124';destination.mkdir(parents=True,exist_ok=True)
    (destination/'listado_drive.json').write_text(json.dumps({'fuentes':FUENTES,'carpetas':folders,
        'archivos':files},ensure_ascii=False,indent=2),encoding='utf-8')
    for folder in folders:print(folder,flush=True)
    print('Imágenes descubiertas:',len(files),flush=True)
    if args.solo_listar:return
    results=[]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures={executor.submit(descargar_archivo,project,item):item for item in files}
        for future in as_completed(futures):
            item=futures[future]
            try:results.append(future.result())
            except Exception as error:
                results.append({**item,'descargado':False,'error':type(error).__name__+': '+str(error)})
            if len(results)%20==0:print('Procesadas:',len(results),'/',len(files),flush=True)
    results.sort(key=lambda r:(r['fuente'],r['ruta_drive'],r['drive_id']))
    (destination/'descargas.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Descargadas:',sum(r['descargado'] for r in results),'| Errores:',sum(not r['descargado'] for r in results),flush=True)


if __name__=='__main__':main()
