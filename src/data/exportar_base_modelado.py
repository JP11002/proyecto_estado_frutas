"""Validar las revisiones históricas usadas como insumo de la base única."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .preparacion_imagenes import construir_grupos, guardar_bytes_si_cambio, guardar_tabla, particionar_por_grupos


def exportar_base_modelado(proyecto):
    proyecto=Path(proyecto).resolve()
    salidas=proyecto/'experiments/results/preparacion_imagenes'
    annotations=proyecto/'data/annotations'
    path=salidas/'base_preparada.csv'
    base=pd.read_csv(path,keep_default_na=False)
    if base.empty:
        raise ValueError('Ejecutar la exportación del cuaderno 02 después de revisar etiquetas y máscaras.')
    if not base.id_ejemplar.is_unique:
        raise ValueError('La base tiene ejemplares repetidos.')
    originales=pd.read_csv(annotations/'revision_equipo.csv',dtype=str,keep_default_na=False).set_index('id_imagen')
    hijos=(pd.read_csv(annotations/'ejemplares_segmentados.csv',dtype=str,keep_default_na=False).set_index('id_ejemplar')
           if (annotations/'ejemplares_segmentados.csv').is_file() else pd.DataFrame())
    masks=pd.read_csv(annotations/'revision_mascaras.csv',dtype=str,keep_default_na=False).set_index('id_ejemplar')
    for row in base.itertuples():
        child=str(row.es_recorte_individual).lower()=='true'
        labels=hijos.loc[row.id_ejemplar] if child else originales.loc[row.id_ejemplar]
        if labels.calidad_revisada!=row.calidad_revisada or labels.grupo_ejemplar!=row.grupo_ejemplar:
            raise ValueError('Cambió una etiqueta o un grupo; volver a exportar desde el cuaderno 02.')
        sha_etiquetada=labels.sha_original if child else labels.sha256
        if sha_etiquetada!=row.sha_original or labels.ejemplar_individual!='si' or not labels.revisor.strip():
            raise ValueError('La etiqueta no confirma un ejemplar individual vigente y revisado.')
        if child and (labels.id_origen!=row.id_origen or labels.fruta!=row.fruta):
            raise ValueError('Cambió el origen o el tipo de fruta de un recorte; actualizar la base.')
        if not child and labels.decision!='incluir':
            raise ValueError('Un original dejó de estar incluido; actualizar la base.')
        if row.fruta=='manzana' and labels.manzana_roja!='si':
            raise ValueError('La base contiene una manzana sin confirmar el alcance rojo.')
        mask=masks.loc[row.id_ejemplar]
        if (mask.decision_mascara!='aceptada' or mask.firma_preparacion!=row.firma_preparacion
                or mask.sha_original!=row.sha_original or mask.sha_mascara!=row.sha_mascara
                or not mask.revisor.strip()):
            raise ValueError('Cambió una revisión de máscara; actualizar la base.')
        for file,expected in [(proyecto/'data/raw/fruit-quality-classification'/row.ruta_relativa,row.sha_original),
                              (Path(row.ruta_mascara),row.sha_mascara),(Path(row.ruta_cnn),row.sha_cnn)]:
            if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Cambió un archivo del ejemplar {row.id_ejemplar}.')
    columnas=[c for c in base if c.startswith(('color_','textura_','forma_'))]
    if not columnas or not np.isfinite(base[columnas].to_numpy(dtype=float)).all():
        raise ValueError('Las características contienen datos ausentes o no finitos.')
    if set(base.calidad_revisada)!={'buena','intermedia','mala'}:
        raise ValueError('La base inicial debe contener las tres calidades.')
    esperadas={(fruta,calidad) for fruta in ['manzana','banano','naranja']
               for calidad in ['buena','intermedia','mala']}
    if set(zip(base.fruta,base.calidad_revisada))!=esperadas:
        raise ValueError('Faltan combinaciones de fruta y calidad o hay categorías fuera del alcance.')
    base['grupo_particion']=construir_grupos(base)
    registros=base[['id_ejemplar','id_origen','fruta','calidad_revisada','grupo_particion',
                    'grupo_ejemplar','sha_original','firma_preparacion',*columnas]].copy()
    resumen=(base.groupby(['fruta','calidad_revisada']).agg(ejemplares=('id_ejemplar','size'),
              fotografias=('id_origen','nunique'),grupos_conservadores=('grupo_particion','nunique')).reset_index())
    history=annotations/'historial_exploracion.csv'
    explored=set(pd.read_csv(history).id_origen) if history.is_file() else set()
    try:
        split=particionar_por_grupos(base,explored)
    except ValueError as error:
        ready_evaluation=False;reason=str(error)
    else:
        ready_evaluation=True;reason='Particiones agrupadas con cobertura completa.'
    status={'protocolo':'calidad_visual_v1','revisor':'Asistente - revision visual',
            'alcance':'Base inicial revisada; anotaciones de un revisor, sin acuerdo interanotador medido.',
            'ejemplares':len(base),'fotografias':int(base.id_origen.nunique()),
            'grupos_conservadores':int(base.grupo_particion.nunique()),'caracteristicas':columnas,
            'lista_para_ajuste_inicial':True,'lista_para_evaluacion_independiente':ready_evaluation,
            'motivo_evaluacion':reason,
            'limitacion_grupos':'Se agrupan campañas de captura conservadoramente; el número de fotografías no equivale a frutos independientes.',
            'limitacion_mascaras':'Contornos aproximados revisados visualmente; no son máscaras de referencia pixel a pixel.',
            'limitacion_tamano':'Ocupación visible en la fotografía; no tamaño físico ni centímetros.'}
    return registros,resumen,status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proyecto',default='.')
    args=parser.parse_args()
    _,summary,status=exportar_base_modelado(args.proyecto)
    print(summary.to_string(index=False))
    print('Ajuste inicial:',status['lista_para_ajuste_inicial'])
    print('Evaluación independiente:',status['lista_para_evaluacion_independiente'])
    print(status['motivo_evaluacion'])


if __name__=='__main__':
    main()
