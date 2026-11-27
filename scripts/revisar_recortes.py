"""Auditoría y comparación controlada del encuadre de entrada CNN sin alterar la base."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2

P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
from src.data.base import cargar_base
from src.data.preparacion_imagenes import leer_rgb,preparar_cnn
from src.models.evaluacion import crear_pliegues_externos,crear_cv_interno,calcular_metricas,guardar_json,guardar_grafica
from src.utils.archivos import guardar_tabla


def encajar(rgb,lado=224,mascara=False):
    h,w=rgb.shape[:2];escala=lado/max(h,w);nw,nh=max(1,round(w*escala)),max(1,round(h*escala))
    salida=np.zeros((lado,lado),dtype=np.uint8) if mascara else np.full((lado,lado,3),128,dtype=np.uint8)
    reducido=cv2.resize(rgb,(nw,nh),interpolation=cv2.INTER_NEAREST if mascara else cv2.INTER_AREA)
    x,y=(lado-nw)//2,(lado-nh)//2;salida[y:y+nh,x:x+nw]=reducido
    return salida


def preparar_amplio(rgb,mascara):
    ys,xs=np.where(mascara);m=max(2,round(.12*max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)))
    x0,y0=max(0,xs.min()-m),max(0,ys.min()-m)
    x1,y1=min(rgb.shape[1],xs.max()+m+1),min(rgb.shape[0],ys.max()+m+1)
    return encajar(rgb[y0:y1,x0:x1])


def ejecutar(comparar=False):
    import torch
    from torchvision.transforms import functional as F
    from src.models.cnn_calidad import CNNTransferencia,ConfigTransferencia,transformacion_transferencia
    base,_,estado=cargar_base(P);salida=P/'experiments/results/comparacion_modelos'
    if not all(Path(r).is_file() for r in base.ruta_original):
        raise FileNotFoundError('Se necesitan todos los originales para mantener iguales las 291 imágenes de comparación.')
    torch.set_num_threads(4);torch.hub.set_dir(str(P/'data/external/torch'))
    red=CNNTransferencia(ConfigTransferencia(),pesos_preentrenados=True).eval() if comparar else None
    normalizar=transformacion_transferencia();auditoria=[];vectores=[];ejemplos={}
    codigos=['pub_banano_077','pub_banano_002','eq_limon_010','eq_manzana_019','v1_295e5823596f47c3_R034','v1_2e32661c1e9c56f3']
    for n,f in enumerate(base.itertuples(),1):
        rgb=leer_rgb(f.ruta_original)
        with Image.open(f.ruta_mascara) as im:m=(np.asarray(im)>0).astype(np.uint8)
        actual,caja=preparar_cnn(rgb,m)
        guardada=leer_rgb(f.ruta_cnn)
        amplio=preparar_amplio(rgb,m);completa=encajar(rgb)
        x0,y0,x1,y1=caja;mini=encajar(m[y0:y1,x0:x1],mascara=True)
        perdidas=[]
        for angulo in [-10,10]:
            v=Image.fromarray(mini)
            a=np.asarray(F.rotate(v,angulo,expand=False));b=np.asarray(F.rotate(v,angulo,expand=True))
            perdidas.append(max(0.,1-float(a.sum())/max(1,float(b.sum()))))
        auditoria.append({'id_ejemplar':f.id_ejemplar,'codigo_revision':f.codigo_revision,'fruta':f.fruta,'calidad':f.calidad_revisada,
                          'grupo':f.grupo_particion,'ancho_original':rgb.shape[1],'alto_original':rgb.shape[0],
                          'mascara_toca_borde_original':bool(m[0].any() or m[-1].any() or m[:,0].any() or m[:,-1].any()),
                          'caja_contiene_toda_mascara':bool(m.sum()==m[y0:y1,x0:x1].sum()),
                          'preparacion_coincide':bool(np.array_equal(completa if estado.get("entrada_cnn")=="fotografia_completa_proporcion_conservada" else actual,guardada)),
                          'perdida_maxima_rotacion':max(perdidas),'ocupacion_mascara_original':float(m.mean())})
        if f.codigo_revision in codigos:
            vista=cv2.resize(rgb,(round(rgb.shape[1]*min(1,700/max(rgb.shape[:2]))),round(rgb.shape[0]*min(1,700/max(rgb.shape[:2])))))
            mm=cv2.resize(m,(vista.shape[1],vista.shape[0]),interpolation=cv2.INTER_NEAREST)
            cont,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(vista,cont,-1,(20,210,70),2)
            ejemplos[f.codigo_revision]=(vista,actual,amplio,completa)
        if comparar:
            with torch.inference_mode():z=red.extraer(torch.stack([normalizar(Image.fromarray(v)) for v in [actual,amplio,completa]]))
            vectores.append(z.clone())
        if n%50==0:print('Recortes:',n,'/',len(base),flush=True)
    tabla=pd.DataFrame(auditoria);guardar_tabla(tabla,salida/'recortes_auditoria.csv')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(ejemplos),4,figsize=(12.5,3*len(ejemplos)),squeeze=False,layout='constrained')
    for i,codigo in enumerate([c for c in codigos if c in ejemplos]):
        for j,(ax,img) in enumerate(zip(axes[i],ejemplos[codigo])):
            ax.imshow(img);ax.axis('off')
            ax.set_title((['Original y contorno','Control: máscara y margen 4%','Margen 12%, RGB conservado','Foto completa, proporción fija'][j] if i==0 else '')+'\n'+codigo,fontsize=9)
    guardar_grafica(fig,salida/'recortes_comparacion_visual.png');plt.close(fig)
    resumen={'sha_base':estado['sha_base'],'imagenes':len(base),'preparaciones_coincidentes':int(tabla.preparacion_coincide.sum()),
             'cajas_que_contienen_mascara':int(tabla.caja_contiene_toda_mascara.sum()),'mascaras_tocan_borde_original':int(tabla.mascara_toca_borde_original.sum()),
             'rotaciones_con_perdida_mayor_1_por_ciento':int(tabla.perdida_maxima_rotacion.gt(.01).sum()),
             'perdida_rotacion_maxima':float(tabla.perdida_maxima_rotacion.max()),'cambio_base':False}
    guardar_json(resumen,salida/'recortes_resumen.json');print(resumen,flush=True)
    if comparar:
        z=torch.stack(vectores).detach().clone()
        comparar_entradas(base,estado,{'recorte_mascara_sin_aumentos':z[:,0:1], 'amplio_RGB':z[:,1:2], 'foto_completa':z[:,2:3]},salida)



def comparar_entradas(base,estado,representaciones,salida):
    from dataclasses import asdict,replace
    from sklearn.metrics import f1_score,precision_score,recall_score
    from src.models.cnn_calidad import candidatos_transferencia,entrenar_cabezal,CALIDADES
    candidatos=[c for c in candidatos_transferencia() if not c.usar_aumentos]
    guardar_json({'sha_base':estado['sha_base'],'semilla':42,'entradas':list(representaciones),
                  'candidatos':[asdict(c) for c in candidatos], 'aumentos':False,
                  'seleccion':'F1 macro interno agrupado; duración media de épocas internas',
                  'alcance':'Ablación de encuadre; mismo extractor, normalización, cabezales y selección. No reemplaza pesos finales.'},salida/'recortes_plan.json')
    partes,asignacion,_=crear_pliegues_externos(base)
    y=base.calidad_revisada.map({c:i for i,c in enumerate(CALIDADES)}).to_numpy();g=base.grupo_particion.to_numpy()
    metricas=[];predicciones=[];busquedas=[];elegidos=[];membresias=[]
    for nombre,z in representaciones.items():
        oof=np.full(len(base),'',dtype=object)
        for exterior,(tr,va) in enumerate(partes,1):
            cv=crear_cv_interno(y[tr],g[tr],semilla=42+exterior*100);internos=list(cv.split(np.zeros(len(tr)),y[tr],g[tr]))
            tabla=[]
            for numero,cfg in enumerate(candidatos):
                scores=[];epocas=[]
                for interior,(a,b) in enumerate(internos,1):
                    assert not set(g[tr[a]])&set(g[tr[b]]) and not set(g[tr])&set(g[va])
                    _,_,info=entrenar_cabezal(z[tr[a]],y[tr[a]],cfg,42+exterior*100+interior,z[tr[b]],y[tr[b]])
                    scores.append(info['mejor_f1_val']);epocas.append(info['mejor_epoca'])
                    if numero==0:
                        for etapa,indices in [('entrenamiento',tr[a]),('validacion',tr[b])]:
                            membresias.extend({'entrada':nombre,'exterior':exterior,'interior':interior,'etapa':etapa,'id_ejemplar':base.iloc[i].id_ejemplar,'grupo':g[i]} for i in indices)
                tabla.append({'entrada':nombre,'exterior':exterior,'candidato':numero,**asdict(cfg),
                              'f1_interno':float(np.mean(scores)),'epocas_seleccionadas':max(cfg.min_epocas,round(np.mean(epocas)))})
            mejor=max(tabla,key=lambda r:r['f1_interno']);cfg=replace(candidatos[mejor['candidato']],max_epocas=mejor['epocas_seleccionadas'])
            cabeza,_,_=entrenar_cabezal(z[tr],y[tr],cfg,42+exterior);cabeza.eval()
            import torch
            with torch.inference_mode():indices=cabeza(z[va,0]).argmax(1).numpy()
            oof[va]=np.array(CALIDADES)[indices];busquedas.extend(tabla);elegidos.append(mejor)
            print('Comparación de encuadre:',nombre,'bloque',exterior,'terminado',flush=True)
        assert not (oof=='').any()
        real=base.calidad_revisada
        metricas.append({'entrada':nombre,**calcular_metricas(real,oof),
                         'recall_intermedia':float(recall_score(real,oof,labels=['intermedia'],average='macro',zero_division=0)),
                         'precision_intermedia':float(precision_score(real,oof,labels=['intermedia'],average='macro',zero_division=0)),
                         'aciertos_intermedia':int(((real=='intermedia')&(oof=='intermedia')).sum())})
        d=base[['id_ejemplar','grupo_particion','fruta','fuente','calidad_revisada']].copy();d['entrada'],d['prediccion'],d['pliegue_validacion']=nombre,oof,asignacion;predicciones.append(d)
    guardar_tabla(pd.DataFrame(metricas),salida/'recortes_metricas.csv');guardar_tabla(pd.concat(predicciones,ignore_index=True),salida/'recortes_predicciones.csv')
    guardar_tabla(pd.DataFrame(busquedas),salida/'recortes_busqueda.csv');guardar_tabla(pd.DataFrame(elegidos),salida/'recortes_seleccion.csv');guardar_tabla(pd.DataFrame(membresias),salida/'recortes_particiones.csv')
    print(pd.DataFrame(metricas).to_string(index=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--comparar',action='store_true');args=parser.parse_args();ejecutar(args.comparar)
