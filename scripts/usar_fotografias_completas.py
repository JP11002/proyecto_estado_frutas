"""Adoptar fotografías completas en la base única por instrucción del equipo."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd,numpy as np
from src.data.base import cargar_base
from src.data.entrada_cnn import guardar_con_fotografias_completas,ENTRADA_CNN,firma_comparable
from src.utils.archivos import guardar_tabla
from src.models.evaluacion import guardar_json


def main():
    p=Path(__file__).resolve().parents[1];base,columnas,estado=cargar_base(p)
    if estado.get('entrada_cnn')==ENTRADA_CNN:
        print('La base única ya contiene las fotografías completas.');return
    salida=p/'experiments/results/comparacion_modelos'
    pred=pd.read_csv(salida/'predicciones_oof.csv')
    anteriores=pred[pred.modelo.isin(['CNN_simple','CNN_transferencia'])].copy()
    anteriores['modelo']=anteriores.modelo.replace({'CNN_transferencia':'CNN_recortada'})
    if len(anteriores):
        guardar_tabla(anteriores,salida/'cnn_referencias_historicas.csv')
        guardar_json({'sha_base_origen':estado['sha_base'],'firma_comparable':firma_comparable(base),
                      'entrada':'Recorte con margen 4% y fondo enmascarado, antes de adoptar fotografías completas.'},
                     salida/'cnn_referencias_historicas.json')
    nueva,_,nuevo_estado=guardar_con_fotografias_completas(p,base,estado)
    assert base.id_ejemplar.tolist()==nueva.id_ejemplar.tolist()
    for c in ['fruta','fuente','calidad_revisada','grupo_particion','sha_original','sha_mascara']:
        assert base[c].tolist()==nueva[c].tolist()
    assert np.allclose(base[columnas],nueva[columnas],rtol=0,atol=1e-12)
    cfg=json.loads((p/'config/alcance_actual.json').read_text(encoding='utf-8'));cfg['entrada_cnn']=ENTRADA_CNN
    guardar_json(cfg,p/'config/alcance_actual.json')
    print('Base única actualizada:',nuevo_estado['sha_base'],'| 291 fotografías completas, etiquetas y grupos conservados.')
if __name__=='__main__':main()
