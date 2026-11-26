"""Entrada CNN de fotografía completa, sin recorte ni eliminación del fondo."""
from pathlib import Path
import io,hashlib,json,tempfile
import numpy as np
import cv2
from PIL import Image
from src.data.preparacion_imagenes import leer_rgb

ENTRADA_CNN='fotografia_completa_proporcion_conservada'


def preparar_imagen_completa(rgb,lado=224):
    """Conservar todo el encuadre, escalar y centrar con relleno gris."""
    rgb=np.asarray(rgb)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
        raise ValueError('Se necesita una imagen RGB no vacía.')
    h,w=rgb.shape[:2];escala=lado/max(h,w)
    nw,nh=max(1,round(w*escala)),max(1,round(h*escala))
    imagen=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_AREA if escala<1 else cv2.INTER_LINEAR)
    salida=np.full((lado,lado,3),128,dtype=np.uint8)
    x,y=(lado-nw)//2,(lado-nh)//2;salida[y:y+nh,x:x+nw]=imagen
    return salida


def firma_comparable(base):
    """Identidad, etiquetas, grupos y máscaras; independiente del encuadre CNN."""
    campos=['id_ejemplar','grupo_particion','fruta','fuente','calidad_revisada','sha_original','sha_mascara']
    return hashlib.sha256(base[campos].to_csv(index=False).encode('utf-8')).hexdigest()


def guardar_con_fotografias_completas(proyecto,tabla,estado):
    """Actualizar la misma base; mantener máscaras, etiquetas y rasgos tradicionales."""
    from src.data.base import guardar_base,resolver_ruta,cargar_base
    proyecto=Path(proyecto).resolve();tabla=tabla.copy()
    if estado.get('entrada_cnn')==ENTRADA_CNN and 'preparacion_cnn' in tabla and tabla.preparacion_cnn.eq(ENTRADA_CNN).all():
        return guardar_base(proyecto,tabla,estado)
    temporal=proyecto/'tmp';temporal.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='preparar_fotos_',dir=temporal) as carpeta:
        for n,(idx,f) in enumerate(tabla.iterrows(),1):
            original=resolver_ruta(proyecto,f.ruta_original)
            if hashlib.sha256(original.read_bytes()).hexdigest()!=f.sha_original:
                raise ValueError('Cambió el original de '+f.id_ejemplar)
            entrada=preparar_imagen_completa(leer_rgb(original));buf=io.BytesIO();Image.fromarray(entrada).save(buf,format='PNG')
            contenido=buf.getvalue();destino=Path(carpeta)/(f.id_ejemplar+'.png');destino.write_bytes(contenido)
            tabla.at[idx,'ruta_cnn']=str(destino)
            tabla.at[idx,'sha_cnn']=hashlib.sha256(contenido).hexdigest()
            tabla.at[idx,'preparacion_cnn']=ENTRADA_CNN
            if n%50==0:print('Fotografías completas preparadas:',n,'/',len(tabla),flush=True)
        nuevo=dict(estado)
        nuevo.update(entrada_cnn=ENTRADA_CNN,lado_cnn=224,
                     descripcion_entrada_cnn='Fotografía RGB completa con fondo original; proporción conservada y relleno gris. Sin recorte.',
                     firma_etiquetas_grupos=firma_comparable(tabla))
        # Escribir sin reemplazar/renombrar archivos: algunos sistemas Windows
        # interrumpen las sustituciones masivas. Las fuentes ya están preparadas
        # y verificadas antes de tocar la base; etiquetas y máscaras no se editan.
        destino=proyecto/'data/base'
        for idx,f in tabla.iterrows():
            contenido=Path(f.ruta_cnn).read_bytes()
            archivo=destino/'imagenes'/(f.id_ejemplar+'.png')
            archivo.write_bytes(contenido)
            tabla.at[idx,'ruta_cnn']=archivo.relative_to(proyecto).as_posix()
            for campo in ['ruta_original','ruta_mascara']:
                tabla.at[idx,campo]=resolver_ruta(proyecto,f[campo]).relative_to(proyecto).as_posix()
        contenido=tabla.to_csv(index=False).encode('utf-8-sig')
        nuevo['sha_base']=hashlib.sha256(contenido).hexdigest()
        (destino/'datos.csv').write_bytes(contenido)
        (destino/'estado.json').write_text(json.dumps(nuevo,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Manifiesto de fotografías completas escrito:',nuevo['sha_base'],flush=True)
        return cargar_base(proyecto)
