"""Inferencia de calidad y medida de tamaño relativo con los modelos del proyecto."""
from pathlib import Path
import json
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from src.data.entrada_cnn import preparar_imagen_completa,ENTRADA_CNN
from src.data.base import cargar_base
from src.data.preparacion_imagenes import leer_rgb, mascara_grabcut, validar_mascara, extraer_caracteristicas, diagnosticar_mascara


def medir_tamano(mascara):
    """Medir sobre el encuadre original, antes del reescalado CNN. No expresa centímetros."""
    m = np.asarray(mascara) > 0
    if m.ndim != 2 or m.sum() < 25:
        raise ValueError('Se necesita una máscara bidimensional con al menos 25 píxeles de fruta.')
    h,w = m.shape; ys,xs = np.where(m)
    diametro = float(2 * np.sqrt(m.sum() / np.pi))
    return {'area_px': int(m.sum()), 'diametro_equivalente_px': diametro,
            'diametro_normalizado': diametro / min(h,w), 'ocupacion': float(m.mean()),
            'ancho_relativo': float((xs.max()-xs.min()+1)/w), 'alto_relativo': float((ys.max()-ys.min()+1)/h),
            'encuadre_px': [int(w),int(h)], 'unidad': 'relativo al encuadre; sin escala física'}


class MotorInferencia:
    """Usa los modelos v2 (experimentos_v2, entrenados solo con desarrollo) si existen; si no, los anteriores."""
    V2 = {'CNN': 'CNN_MobileNetV3_cabezal', 'Random Forest': 'RF_ajustado', 'SVM': 'SVM_ajustado'}

    def __init__(self, proyecto=None):
        self.proyecto = Path(proyecto or Path(__file__).resolve().parents[2]).resolve()
        self.base,self.columnas,self.estado = cargar_base(self.proyecto)
        self.cache = {}
        self._red_rasgos = None
        self.rutas = {
            'CNN': self.proyecto/'experiments/checkpoints/comparacion_modelos/cnn_calidad.pt',
            'Random Forest': self.proyecto/'experiments/checkpoints/modelos_tradicionales/random_forest.joblib',
            'SVM': self.proyecto/'experiments/checkpoints/ajuste_modelos/svm_ajustado.joblib'}
        self.v2 = {}
        manifiesto = self.proyecto/'experiments/checkpoints/experimentos_v2/manifiesto.json'
        if manifiesto.is_file():
            datos = json.loads(manifiesto.read_text(encoding='utf-8'))
            if datos.get('sha_base') == self.estado['sha_base']:
                modelos = datos['modelos']
                for visible, interno in self.V2.items():
                    if interno in modelos:
                        self.v2[visible] = {'nombre': interno, **modelos[interno]}
                final = datos['modelo_final']['modelo']
                if final in modelos and final not in self.V2.values():
                    self.v2['Modelo final (' + final + ')'] = {'nombre': final, **modelos[final]}
                for visible, info in self.v2.items():
                    self.rutas[visible] = self.proyecto/info['archivo']

    def disponibles(self):
        return [n for n,p in self.rutas.items() if p.is_file()]

    def _red_cnn(self):
        """Extractor MobileNetV3 congelado (ImageNet) para modelos con rasgos CNN."""
        if self._red_rasgos is None:
            import torch
            from src.models.cnn_calidad import CNNTransferencia,ConfigTransferencia
            torch.hub.set_dir(str(self.proyecto/'data/external/torch'))
            torch.set_num_threads(min(4,torch.get_num_threads()))
            self._red_rasgos = CNNTransferencia(ConfigTransferencia(), pesos_preentrenados=True).eval()
        return self._red_rasgos

    def _cargar(self,nombre):
        if nombre not in self.rutas:
            raise ValueError('Modelo no reconocido.')
        if nombre in self.cache:
            return self.cache[nombre]
        ruta = self.rutas[nombre]
        if not ruta.is_file():
            raise FileNotFoundError('Faltan los pesos de '+nombre+'. Ejecutar python -m src.models.experimentos_base_ampliada.')
        if nombre in self.v2:
            if ruta.suffix == '.pt':
                import torch
                from src.models.cnn_calidad import crear_cabezal,ConfigTransferencia
                paquete = torch.load(ruta,map_location='cpu',weights_only=False)
                modelo = crear_cabezal(ConfigTransferencia(**paquete['parametros']))
                modelo.load_state_dict(paquete['state_dict']);modelo.eval()
                paquete['rasgos'] = 'cnn'
            else:
                import joblib
                paquete = joblib.load(ruta);modelo = paquete['modelo']
            paquete['variante'] = self.v2[nombre]['variante']
        elif nombre == 'CNN':
            import torch
            from src.models.cnn_calidad import CNNTransferencia,ConfigTransferencia
            torch.set_num_threads(min(4,torch.get_num_threads()))
            paquete = torch.load(ruta,map_location='cpu',weights_only=True)
            if paquete.get('arquitectura') != 'mobilenet_v3_small_congelada':
                raise ValueError('Los pesos CNN no corresponden a la arquitectura actual.')
            modelo = CNNTransferencia(ConfigTransferencia(**paquete['config']))
            modelo.load_state_dict(paquete['state_dict']);modelo.eval()
            if paquete.get('preparacion_cnn')!=self.estado.get('entrada_cnn'):
                raise ValueError('La preparación del modelo CNN no coincide con la base actual.')
        else:
            import joblib
            paquete = joblib.load(ruta);modelo=paquete['modelo']
        if paquete['sha_base'] != self.estado['sha_base']:
            raise ValueError('El modelo y la base tienen huellas distintas. Reentrenar antes de inferir.')
        self.cache[nombre] = (modelo,paquete)
        return self.cache[nombre]

    def usa_rasgos_manuales(self,nombre):
        """Los modelos con rasgos manuales o de fusión necesitan una máscara válida."""
        if nombre in getattr(self, 'v2', {}):
            _,paquete = self._cargar(nombre)
            return paquete['rasgos'] in {'manual','fusion'}
        return nombre != 'CNN'

    def _predecir(self,entrada,rasgos,nombre):
        modelo,paquete = self._cargar(nombre)
        if nombre in self.v2:
            import torch
            from src.models.cnn_calidad import transformacion_transferencia
            partes = []
            if paquete['rasgos'] in {'manual','fusion'}:
                partes.append(np.asarray([[rasgos[c] for c in paquete['caracteristicas']]],dtype=float))
            if paquete['rasgos'] in {'cnn','fusion'}:
                x = transformacion_transferencia()(Image.fromarray(entrada)).unsqueeze(0)
                with torch.inference_mode():
                    partes.append(self._red_cnn().extraer(x).numpy())
            X = np.hstack(partes)
            if isinstance(modelo, torch.nn.Module):
                with torch.inference_mode():
                    p = modelo(torch.tensor(X,dtype=torch.float32)).softmax(1)[0].numpy()
            else:
                p = modelo.predict_proba(X)[0]
            pesos = np.asarray(paquete['pesos_decision']) if paquete['variante']=='umbral' else np.ones(3)
            scores = {c:float(v) for c,v in zip(paquete['calidades'],p)}
            calidad = paquete['calidades'][int(np.argmax(p*pesos))]
            return {'modelo':nombre,'calidad':calidad,'puntuaciones_no_calibradas':scores,
                    'variante_decision':paquete['variante'],'pesos_decision':pesos.tolist(),
                    'sha_base':self.estado['sha_base']}
        if nombre == 'CNN':
            import torch
            from src.models.cnn_calidad import transformacion_transferencia
            x=transformacion_transferencia()(Image.fromarray(entrada)).unsqueeze(0)
            with torch.inference_mode():
                valores=modelo(x).softmax(1)[0].numpy()
            scores={c:float(v) for c,v in zip(paquete['calidades'],valores)}
            calidad=max(scores,key=scores.get)
        else:
            tabla=pd.DataFrame([rasgos],columns=paquete['caracteristicas'])
            calidad=str(modelo.predict(tabla)[0]);scores={}
            if hasattr(modelo,'predict_proba'):
                scores={str(c):float(v) for c,v in zip(modelo.classes_,modelo.predict_proba(tabla)[0])}
        return {'modelo':nombre,'calidad':calidad,'puntuaciones_no_calibradas':scores,'sha_base':self.estado['sha_base']}

    def analizar(self,rgb,nombre='CNN',mascara=None,rectangulo=None):
        rgb=np.asarray(rgb,dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2])<16:
            raise ValueError('Cargar una imagen RGB con al menos 16 píxeles por lado.')
        entrada=preparar_imagen_completa(rgb)
        alertas=[]
        try:
            mascara=validar_mascara(rgb,mascara if mascara is not None else mascara_grabcut(rgb,rectangulo))
            tamano=medir_tamano(mascara)
            alertas=[a for a in diagnosticar_mascara(rgb,mascara)['alertas_mascara'].split(';') if a]
            rasgos=extraer_caracteristicas(rgb,mascara) if self.usa_rasgos_manuales(nombre) else None
        except (ValueError,cv2.error):
            if self.usa_rasgos_manuales(nombre):raise
            mascara=None;tamano=None;rasgos=None
            alertas=['No se pudo aislar el fruto para medir su tamaño. La calidad CNN usa la fotografía completa.']
        resultado=self._predecir(entrada,rasgos,nombre)
        resultado.update(tamano=tamano,alertas=alertas,origen='imagen_nueva',
                         preparacion_entrada='rasgos_desde_mascara' if self.usa_rasgos_manuales(nombre) else ENTRADA_CNN)
        return resultado,entrada,mascara

    def ejemplo(self,id_ejemplar,nombre='CNN'):
        filas=self.base.loc[self.base.id_ejemplar.eq(id_ejemplar)]
        if len(filas)!=1:
            raise ValueError('El ejemplar no existe en la base.')
        fila=filas.iloc[0]
        entrada=leer_rgb(fila.ruta_cnn)
        with Image.open(fila.ruta_mascara) as im:
            mascara=np.asarray(im)>0
        resultado=self._predecir(entrada,fila[self.columnas].to_dict(),nombre)
        resultado.update(tamano=medir_tamano(mascara),alertas=[],origen='ejemplo_de_entrenamiento',
                         etiqueta_revisada=fila.calidad_revisada,id_ejemplar=id_ejemplar)
        return resultado,entrada,None
