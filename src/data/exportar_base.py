"""Base vigente: manzana roja, banano y limón; solo revisiones explícitas y vigentes."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .exportar_base_modelado import exportar_base_modelado
from .identidad_frutos import aplicar_identidades,exportar_identidades
from .base import guardar_base
from .preparacion_imagenes import construir_grupos,guardar_tabla,guardar_bytes_si_cambio,particionar_por_grupos


def guardar_revision(proyecto,codigo,calidad,decision_mascara,observaciones,revisor):
    """Confirmar o rechazar una propuesta exacta desde el revisor del cuaderno 04."""
    proyecto=Path(proyecto).resolve()
    if calidad not in {'buena','intermedia','mala',''} or decision_mascara not in {'aceptada','rechazada','pendiente'}:
        raise ValueError('Decisión o calidad inválida.')
    if not revisor.strip() or not observaciones.strip():raise ValueError('Registrar revisor y motivo.')
    path=proyecto/'data/annotations/drive_20261124/revision_visual_v2.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    review=next(r for r in manifest['lote_detallado'] if r['codigo_revision']==codigo)
    anterior={k:review.get(k,'') for k in ['calidad_revisada','decision_calidad','decision_mascara','revisor','observaciones']}
    data=pd.read_csv(proyecto/'experiments/results/integracion_drive/propuestas_mascaras.csv',keep_default_na=False)
    row=data.set_index('codigo_revision').loc[codigo]
    for file,sha in [(proyecto/row.ruta_local,row.sha_original),
                     (Path(row.ruta_mascara),row.sha_mascara),(Path(row.ruta_cnn),row.sha_cnn)]:
        if hashlib.sha256(file.read_bytes()).hexdigest()!=sha:raise ValueError('Cambió el original o la propuesta.')
    review.update(calidad_revisada=calidad,decision_calidad='confirmada' if calidad else 'pendiente',
        decision_mascara=decision_mascara,observaciones=observaciones.strip(),revisor=revisor.strip(),
        sha_original=row.sha_original,sha_mascara=row.sha_mascara,sha_cnn=row.sha_cnn,
        firma_preparacion=row.firma_preparacion)
    review['decision_imagen']=('incluir' if calidad and decision_mascara=='aceptada' else
                              'excluir_mascara' if decision_mascara=='rechazada' else 'pendiente')
    review['fecha_revision_utc']=datetime.now(timezone.utc).isoformat()
    posterior={k:review.get(k,'') for k in anterior}
    if anterior!=posterior:
        review.setdefault('historial_decisiones',[]).append({
            'fecha_utc':review['fecha_revision_utc'],'anterior':anterior,'posterior':posterior})
    guardar_bytes_si_cambio(path,json.dumps(manifest,ensure_ascii=False,indent=2).encode('utf-8'))
    # Conservar la máscara revisada fuera de la carpeta de salidas regenerables.
    if decision_mascara=='aceptada':
        target=proyecto/'data/annotations/drive_20261124/mascaras_revisadas'/row.id_ejemplar/row.sha_mascara
        guardar_bytes_si_cambio(target/'mascara.png',Path(row.ruta_mascara).read_bytes())
        guardar_bytes_si_cambio(target/'receta.json',(Path(row.ruta_mascara).parent/'preparacion.json').read_bytes())


def exportar(proyecto):
    project=Path(proyecto).resolve();annotations=project/'data/annotations/drive_20261124'
    config=json.loads((project/'config/alcance_actual.json').read_text(encoding='utf-8'))
    manifest=json.loads((annotations/'revision_visual_v2.json').read_text(encoding='utf-8'))
    inventory=pd.read_csv(annotations/'inventario_drive.csv',keep_default_na=False)
    reviews=pd.DataFrame(manifest['lote_detallado']).set_index('codigo_revision')
    if not reviews.index.is_unique:raise ValueError('Hay códigos de revisión repetidos.')
    proposed=pd.read_csv(project/'experiments/results/integracion_drive/propuestas_mascaras.csv',keep_default_na=False)
    candidates=proposed.set_index('codigo_revision');rows=[]
    for code,review in reviews.iterrows():
        if str(review.get('decision_imagen','')).startswith('excluir_'):continue
        if review.decision_calidad!='confirmada' or review.decision_mascara!='aceptada':continue
        row=candidates.loc[code].copy()
        source=inventory.set_index('codigo_revision').loc[code]
        for field in ['sha_original','sha_mascara','sha_cnn','firma_preparacion']:
            if review[field]!=row[field]:raise ValueError('Revisión desactualizada: '+code)
        if row.sha_original!=source.sha256:raise ValueError('Cambió el inventario: '+code)
        for file,sha in [(project/row.ruta_local,row.sha_original),
                         (Path(row.ruta_mascara),row.sha_mascara),(Path(row.ruta_cnn),row.sha_cnn)]:
            if hashlib.sha256(file.read_bytes()).hexdigest()!=sha:raise ValueError('Archivo modificado: '+code)
        if row.fruta=='manzana' and review.manzana_roja!='si':raise ValueError('Manzana fuera del alcance rojo.')
        if review.calidad_revisada not in config['calidades'] or not review.revisor.strip():
            raise ValueError('Etiqueta sin calidad válida o revisor.')
        row.update(review)
        row['codigo_revision']=code
        row['ruta_original']=str((project/source.ruta_local).resolve())
        rows.append(row.to_dict())
    new=pd.DataFrame(rows)
    # Revalidar la base histórica antes de recuperar únicamente manzana y banano.
    exportar_base_modelado(project)
    old=pd.read_csv(project/'experiments/results/preparacion_imagenes/base_preparada.csv',keep_default_na=False)
    old=old.loc[old.fruta.isin(['manzana','banano'])].copy()
    old['fuente']='publica_kaggle';old['codigo_revision']='v1_'+old.id_ejemplar
    old['ruta_original']=[str((project/'data/raw/fruit-quality-classification'/p).resolve()) for p in old.ruta_relativa]
    old['revisor']='Asistente - revision visual'
    base=pd.concat([old,new],ignore_index=True)
    if not base.id_ejemplar.is_unique:raise ValueError('Ejemplares duplicados.')
    if not set(base.fruta)<=set(config['frutas']):raise ValueError('Fruta fuera del alcance vigente.')
    features=[c for c in old if c.startswith(('color_','textura_','forma_'))]
    if len(features)!=43 or not np.isfinite(base[features].to_numpy(dtype=float)).all():
        raise ValueError('Características incompletas o no finitas.')
    base=aplicar_identidades(project,base)
    base['grupo_particion']=construir_grupos(base)
    core=['id_ejemplar','id_origen','codigo_revision','fuente','fruta','calidad_revisada','grupo_ejemplar',
          'grupo_particion','sha_original','sha_mascara','sha_cnn','firma_preparacion','revisor',
          'ruta_original','ruta_mascara','ruta_cnn','id_fruto_propuesto','certeza_identidad',
          'revisor_identidad',*features]
    output=project/'data/base'
    summary=base.groupby(['fuente','fruta','calidad_revisada']).agg(
        ejemplares=('id_ejemplar','size'),grupos_conservadores=('grupo_particion','nunique')).reset_index()
    guardar_tabla(summary,output/'resumen_clases.csv')
    expected=pd.MultiIndex.from_product([config['frutas'],config['calidades']],names=['fruta','calidad_revisada'])
    coverage=base.groupby(['fruta','calidad_revisada']).size().reindex(expected,fill_value=0).rename('ejemplares').reset_index()
    guardar_tabla(coverage,output/'cobertura.csv')
    try:particionar_por_grupos(base,base.id_origen,frutas=config['frutas'])
    except ValueError as error:reason=str(error)
    else:raise AssertionError('Una base completamente explorada no debe generar una prueba independiente.')
    status={'frutas':config['frutas'],'ejemplares':len(base),
        'nuevos_drive':len(new),'anteriores_compatibles':len(old),'caracteristicas':features,
        'grupos_conservadores':int(base.grupo_particion.nunique()),
        'combinaciones_ausentes':coverage.loc[coverage.ejemplares.eq(0),['fruta','calidad_revisada']].to_dict('records'),
        'originales_con_decision':len(reviews),
        'mascaras_aceptadas_drive':int(reviews.decision_mascara.eq('aceptada').sum()),
        'mascaras_rechazadas_drive':int(reviews.decision_mascara.eq('rechazada').sum()),
        'fotos_multiples_excluidas':int(reviews.get('decision_imagen',pd.Series(index=reviews.index,dtype=object)).eq('excluir_foto_multiple').sum()),
        'calidades_pendientes_con_mascara_aceptada':int((reviews.decision_mascara.eq('aceptada')&~reviews.decision_calidad.eq('confirmada')).sum()),
        'lista_para_ajuste_tecnico':set(base.calidad_revisada)==set(config['calidades']),
        'lista_para_evaluacion_independiente':False,'motivo':reason,
        'agrupacion':('Propias: identidades visuales propuestas y bloques de incertidumbre. '
                     'Públicas: campañas conservadoras; sin identidades físicas verificadas.'),
        'fotos_propias_base':int(base.fuente.eq('propia').sum()),
        'identidades_propias_propuestas_en_base':int(base.loc[base.fuente.eq('propia'),'id_fruto_propuesto'].replace('',np.nan).nunique()),
        'grupos_seguros_propios_en_base':int(base.loc[base.fuente.eq('propia'),'grupo_particion'].nunique()),
        'alcance':'Lote de desarrollo; calidad externa y contornos aproximados de un revisor.'}
    if config.get('entrada_cnn')=='fotografia_completa_proporcion_conservada':
        from .entrada_cnn import guardar_con_fotografias_completas
        base,features,status=guardar_con_fotografias_completas(project,base[core],status)
    else:
        base,features,status=guardar_base(project,base[core],status)
    # Lotes revisados posteriores (p. ej. drive_20261126): volver a añadirlos para que
    # reconstruir la base desde el flujo del 24/11 no los elimine sin aviso.
    if (project/'data/annotations/drive_20261126/revision_visual.json').is_file():
        from .incorporar_lote import incorporar
        base,_,status=incorporar(project)
    # Inventario completo: las imágenes sin revisión detallada permanecen pendientes.
    full=inventory.merge(reviews.reset_index().drop(columns=['parametros'],errors='ignore'),
                         on='codigo_revision',how='left',validate='1:1',suffixes=('','_revision'))
    full['decision_calidad']=full.decision_calidad.fillna('pendiente_revision_detallada')
    full['decision_mascara']=full.decision_mascara.fillna('pendiente')
    defaults=np.where(full.decision_mascara.eq('aceptada')&full.decision_calidad.eq('confirmada'),
                      'incluir',np.where(full.decision_mascara.eq('rechazada'),'excluir_mascara','pendiente'))
    if 'decision_imagen' not in full:full['decision_imagen']=defaults
    else:full['decision_imagen']=full.decision_imagen.fillna(pd.Series(defaults,index=full.index))
    full['estado_revision']=np.where(full.decision_imagen.eq('incluir'),'incluida',
        np.where(full.decision_imagen.str.startswith('excluir_'),'excluida','pendiente'))
    full['cribado_visual']='inspeccionada_en_galeria_sin_aprobacion_automatica'
    full['varios_frutos_visibles']=full.codigo_revision.isin(manifest.get('codigos_varios_frutos',[]))
    if (project/'data/annotations/identidad_propias_v1.json').is_file():
        _,_,photo_ids=exportar_identidades(project)
        columns=['codigo_revision','id_fruto_propuesto','certeza_identidad',
                 'grupo_evaluacion_seguro','estado_identidad','ids_frutos_visibles','ids_alternativos']
        full=full.merge(photo_ids[columns],on='codigo_revision',how='left',validate='1:1')
    guardar_tabla(full,annotations/'etiquetas_y_mascaras.csv')
    # Todo lo visto para desarrollar este procesamiento queda registrado como explorado.
    explored=inventory[['id_imagen','fuente','fruta']].rename(columns={'id_imagen':'id_origen'})
    explored['uso']='desarrollo_procesamiento_y_revision_visual'
    guardar_tabla(explored,annotations/'historial_exploracion.csv')
    return base,summary,status


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--proyecto',default='.')
    args=parser.parse_args();base,summary,status=exportar(args.proyecto)
    print(summary.to_string(index=False));print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
