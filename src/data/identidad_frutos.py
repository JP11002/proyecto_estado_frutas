"""Identidades visuales propuestas de fotos propias y grupos seguros de evaluación.

La identidad propuesta y el grupo de evaluación son datos diferentes: un vínculo
incierto mantiene juntas las vistas aunque se muestren con identificadores distintos.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image,ImageDraw,ImageOps,ImageFont

from .preparacion_imagenes import guardar_tabla,guardar_bytes_si_cambio


def cargar_identidades(proyecto):
    project=Path(proyecto).resolve()
    path=project/'data/annotations/identidad_propias_v1.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    inventory=pd.read_csv(project/'data/annotations/drive_20261124/inventario_drive.csv',keep_default_na=False)
    own=inventory.loc[inventory.fuente.eq('propia')].set_index('codigo_revision')
    catalog=pd.DataFrame(manifest['catalogo_frutos'])
    occurrences=pd.DataFrame(manifest['ocurrencias'])
    if not catalog.id_fruto.is_unique or not occurrences.id_ocurrencia.is_unique:
        raise ValueError('Identificadores de frutos u ocurrencias repetidos.')
    if set(occurrences.codigo_revision)!=set(own.index):
        raise ValueError('La anotación debe cubrir exactamente las 51 fotografías propias.')
    profile=catalog.set_index('id_fruto')
    images=[]
    for code,group in occurrences.groupby('codigo_revision',sort=True):
        source=own.loc[code]
        if set(group.sha_original)!={source.sha256} or set(group.id_imagen)!={source.id_imagen}:
            raise ValueError('Identidad desactualizada respecto al original: '+code)
        if hashlib.sha256((project/source.ruta_local).read_bytes()).hexdigest()!=source.sha256:
            raise ValueError('Cambió una fotografía propia: '+code)
        ids=[value for value in group.id_fruto_propuesto if value]
        if len(ids)!=len(set(ids)):
            raise ValueError('Un mismo fruto no puede aparecer dos veces simultáneamente: '+code)
        for record in group.to_dict('records'):
            references=[record['id_fruto_propuesto'],*record['ids_alternativos']]
            for ident in references:
                if ident and (ident not in profile.index or profile.loc[ident,'fruta']!=source.fruta):
                    raise ValueError('Identidad desconocida o de otra fruta: '+code)
            if record['certeza_visual'] not in {'alta','media','no_resuelta'}:
                raise ValueError('Certeza visual no reconocida.')
        single=len(group)==1 and bool(ids)
        main=ids[0] if single else ''
        certainty=group.iloc[0].certeza_visual if single else 'varias_regiones'
        safe=profile.loc[main,'grupo_evaluacion_seguro'] if single else ''
        alternatives=sorted({ident for values in group.ids_alternativos for ident in values})
        images.append({'codigo_revision':code,'id_imagen':source.id_imagen,'nombre_original':source.nombre_original,
            'fruta':source.fruta,'fecha_exif':source.fecha_exif,'sha_original':source.sha256,
            'id_fruto_propuesto':main,'ids_frutos_visibles':'|'.join(ids),
            'ids_alternativos':'|'.join(alternatives),'ocurrencias_anotadas':len(group),
            'certeza_identidad':certainty,'grupo_evaluacion_seguro':safe,
            'estado_identidad':'propuesta_visual' if single else 'varios_frutos' if ids else 'sin_resolver',
            'revisor_identidad':'|'.join(sorted(set(group.revisor_identidad)))})
    photos=pd.DataFrame(images)
    # Todas las alternativas de una foto individual deben permanecer en el mismo bloque seguro.
    for row in photos.loc[photos.id_fruto_propuesto.ne('')].itertuples():
        for ident in row.ids_alternativos.split('|'):
            if ident and profile.loc[ident,'grupo_evaluacion_seguro']!=row.grupo_evaluacion_seguro:
                raise ValueError('Una alternativa incierta quedó separada: '+row.codigo_revision)
    return manifest,catalog,occurrences,photos


def aplicar_identidades(proyecto,base):
    """Añadir metadatos; sustituir grupos propios solo por bloques visualmente seguros."""
    result=base.copy()
    result['id_fruto_propuesto']='';result['certeza_identidad']='sin_revision_fisica'
    result['revisor_identidad']=''
    path=Path(proyecto)/'data/annotations/identidad_propias_v1.json'
    if not path.is_file():return result
    _,_,_,photos=cargar_identidades(proyecto)
    lookup=photos.set_index('codigo_revision')
    own=result.fuente.eq('propia')
    for index in result.index[own]:
        code=result.at[index,'codigo_revision'];row=lookup.loc[code]
        if not row.id_fruto_propuesto or not row.grupo_evaluacion_seguro:
            raise ValueError('Una foto de entrenamiento no tiene una identidad individual utilizable: '+code)
        if row.sha_original!=result.at[index,'sha_original']:
            raise ValueError('La identidad no corresponde al original preparado.')
        result.at[index,'id_fruto_propuesto']=row.id_fruto_propuesto
        result.at[index,'certeza_identidad']=row.certeza_identidad
        result.at[index,'revisor_identidad']=row.revisor_identidad
        result.at[index,'grupo_ejemplar']=row.grupo_evaluacion_seguro
    return result


def exportar_identidades(proyecto):
    project=Path(proyecto).resolve();manifest,catalog,occurrences,photos=cargar_identidades(project)
    folder=project/'data/annotations/identidad_propias'
    guardar_tabla(photos,folder/'fotos_propias_con_identidad.csv')
    printable=occurrences.copy()
    for col in ['ids_alternativos','centro_normalizado']:
        printable[col]=printable[col].map(lambda value:json.dumps(value,ensure_ascii=False))
    guardar_tabla(printable,folder/'ocurrencias_por_fruto.csv')
    counts=occurrences.loc[occurrences.id_fruto_propuesto.ne('')].groupby('id_fruto_propuesto').codigo_revision.nunique()
    catalog['fotografias_con_asociacion_propuesta']=catalog.id_fruto.map(counts).fillna(0).astype(int)
    singles=photos.loc[photos.id_fruto_propuesto.ne('')].groupby('id_fruto_propuesto').size()
    catalog['fotografias_individuales']=catalog.id_fruto.map(singles).fillna(0).astype(int)
    guardar_tabla(catalog,folder/'catalogo_frutos.csv')
    unresolved=photos.loc[photos.estado_identidad.eq('sin_resolver')|photos.certeza_identidad.eq('media')|
                           photos.ids_alternativos.ne('')].copy()
    guardar_tabla(unresolved,folder/'correspondencias_por_confirmar.csv')
    return catalog,occurrences,photos


def guardar_identidad_individual(proyecto,codigo,ident,alternativas,certeza,notas,revisor):
    """Corregir una propuesta individual con autoría e historial, sin separar bloques inciertos."""
    project=Path(proyecto).resolve();manifest,catalog,occurrences,photos=cargar_identidades(project)
    if not revisor.strip() or not notas.strip():raise ValueError('Registrar el nombre del revisor y la evidencia.')
    if certeza not in {'alta','media'}:raise ValueError('Elegir certeza alta o media.')
    rows=occurrences.loc[occurrences.codigo_revision.eq(codigo)]
    if len(rows)!=1:raise ValueError('Este editor corresponde a fotos de un solo fruto.')
    references=[ident,*alternativas];profile=catalog.set_index('id_fruto')
    if any(value not in profile.index or profile.loc[value,'fruta']!=rows.iloc[0].fruta for value in references):
        raise ValueError('Seleccionar identificadores de la fruta correspondiente.')
    record=next(r for r in manifest['ocurrencias'] if r['codigo_revision']==codigo)
    before={k:record[k] for k in ['id_fruto_propuesto','ids_alternativos','certeza_visual','evidencia','revisor_identidad']}
    record.update(id_fruto_propuesto=ident,ids_alternativos=sorted(set(alternativas)-{ident}),
                  certeza_visual=certeza,evidencia=notas.strip(),revisor_identidad=revisor.strip())
    # Una nueva alternativa incierta une los bloques; nunca habilita una separación automáticamente.
    groups=set(profile.loc[references,'grupo_evaluacion_seguro'])
    if len(groups)>1:
        merged='propia_union_'+hashlib.sha256('|'.join(sorted(groups)).encode()).hexdigest()[:12]
        for fruit in manifest['catalogo_frutos']:
            if fruit['grupo_evaluacion_seguro'] in groups:fruit['grupo_evaluacion_seguro']=merged
    manifest.setdefault('historial_identidades',[]).append({'codigo_revision':codigo,
        'fecha_utc':datetime.now(timezone.utc).isoformat(),'anterior':before,
        'posterior':{k:record[k] for k in before}})
    guardar_bytes_si_cambio(project/'data/annotations/identidad_propias_v1.json',
        json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8'))
    exportar_identidades(project)


def crear_evidencia(proyecto):
    project=Path(proyecto).resolve();manifest,catalog,occurrences,photos=cargar_identidades(project)
    inventory=pd.read_csv(project/'data/annotations/drive_20261124/inventario_drive.csv',keep_default_na=False).set_index('codigo_revision')
    destination=project/'experiments/results/identidad_propias';destination.mkdir(parents=True,exist_ok=True)
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    except OSError:font=ImageFont.load_default(size=17)
    for code in ['eq_manzana_016','eq_limon_008','eq_banano_001']:
        with Image.open(inventory.loc[code,'ruta_miniatura']) as image:
            picture=ImageOps.contain(image.convert('RGB'),(945,1260))
        draw=ImageDraw.Draw(picture)
        for record in occurrences.loc[occurrences.codigo_revision.eq(code)].to_dict('records'):
            if record['centro_normalizado'] is None:continue
            x,y=record['centro_normalizado'];x*=picture.width;y*=picture.height
            label=record['id_fruto_propuesto']+(' ?' if record['certeza_visual']=='media' else '')
            draw.ellipse((x-7,y-7,x+7,y+7),fill='#e03131',outline='white',width=2)
            box=draw.textbbox((x+10,y-12),label,font=font)
            if box[2]>picture.width-3:
                x=picture.width-(box[2]-box[0])-16;box=draw.textbbox((x+10,y-12),label,font=font)
            draw.rectangle((box[0]-3,box[1]-3,box[2]+3,box[3]+3),fill='white')
            draw.text((x+10,y-12),label,fill='#142b3b',font=font)
        buffer=io.BytesIO();picture.save(buffer,format='JPEG',quality=94)
        guardar_bytes_si_cambio(destination/(code+'_identificado.jpg'),buffer.getvalue())
    # Panel de repeticiones individuales para comprobar las correspondencias más claras.
    panels=[('EQ_BANANO_01',['eq_banano_006','eq_banano_007']),
            ('EQ_LIMON_01',['eq_limon_001','eq_limon_003']),
            ('EQ_MANZANA_03',['eq_manzana_007','eq_manzana_008'])]
    preparations=pd.read_csv(project/'experiments/results/integracion_drive/propuestas_mascaras.csv').set_index('codigo_revision')
    canvas=Image.new('RGB',(1100,1050),'white');draw=ImageDraw.Draw(canvas)
    for i,(ident,codes) in enumerate(panels):
        draw.text((12,i*350+8),ident+' — dos vistas del mismo perfil visual',fill='black',font=font)
        for j,code in enumerate(codes):
            with Image.open(preparations.loc[code,'ruta_cnn']) as image:crop=image.convert('RGB').resize((280,280))
            canvas.paste(crop,(j*550+125,i*350+40))
            draw.text((j*550+90,i*350+322),inventory.loc[code,'nombre_original']+' / '+code,fill='black',font=font)
    buffer=io.BytesIO();canvas.save(buffer,format='JPEG',quality=94)
    guardar_bytes_si_cambio(destination/'vistas_repetidas.jpg',buffer.getvalue())


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proyecto',default='.')
    args=parser.parse_args();catalog,occurrences,photos=exportar_identidades(args.proyecto)
    crear_evidencia(args.proyecto)
    print('Fotografías:',len(photos),'| Perfiles de referencia:',len(catalog),'| Regiones:',len(occurrences))
    print(photos.groupby(['fruta','estado_identidad']).size().to_string())


if __name__=='__main__':main()
