"""Medidas de tamaño de la base única a partir de sus máscaras revisadas."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import numpy as np
from PIL import Image
from src.data.base import cargar_base
from src.inference.motor import medir_tamano
from src.utils.archivos import guardar_tabla

def exportar(proyecto=None):
    proyecto=Path(proyecto or Path(__file__).resolve().parents[1])
    base,_,estado=cargar_base(proyecto);filas=[]
    for f in base.itertuples():
        with Image.open(f.ruta_mascara) as im:medidas=medir_tamano(np.asarray(im)>0)
        medidas['encuadre_ancho_px'],medidas['encuadre_alto_px']=medidas.pop('encuadre_px')
        filas.append({'id_ejemplar':f.id_ejemplar,'fruta':f.fruta,'fuente':f.fuente,'sha_base':estado['sha_base'],
                      'metodo':'medición desde máscara revisada; sin referencia física',**medidas})
    tabla=pd.DataFrame(filas);guardar_tabla(tabla,proyecto/'data/base/medidas_tamano.csv')
    print(f'{len(tabla)} medidas enlazadas a la base única; etiquetas y huella de datos.csv intactas.')
    return tabla
if __name__=='__main__':exportar()
