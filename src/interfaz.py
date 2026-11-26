"""Interfaz local: foto, cámara, calidad y medida relativa de tamaño."""
from concurrent.futures import ThreadPoolExecutor
import time
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import numpy as np
from PIL import Image,ImageTk
from src.data.preparacion_imagenes import leer_rgb
from src.inference.motor import MotorInferencia


class Aplicacion:
    def __init__(self,raiz):
        self.raiz=raiz;raiz.title('Estado de frutas | Calidad y tamaño relativo');raiz.geometry('1040x770')
        self.motor=MotorInferencia();self.pool=ThreadPoolExecutor(max_workers=1)
        self.futuro=None;self.camara=None;self.rgb=None;self.id_base=None;self.ultimo=None;self.ultima_inferencia=0
        self.modelo=tk.StringVar(value=(self.motor.disponibles() or ['CNN'])[0])
        self.estado=tk.StringVar(value='Carga una foto de un fruto sobre fondo simple o abre la cámara.')
        self.salida=tk.StringVar(value='Calidad: pendiente\nTamaño relativo: pendiente')
        cont=ttk.Frame(raiz,padding=16);cont.pack(fill='both',expand=True)
        ttk.Label(cont,text='Clasificación de calidad de frutas',font=('Segoe UI',19,'bold')).pack(anchor='w')
        ttk.Label(cont,text='Manzana roja · Banano · Limón | Un fruto completo por imagen').pack(anchor='w',pady=(3,12))
        barra=ttk.Frame(cont);barra.pack(fill='x')
        ttk.Button(barra,text='Cargar foto',command=self.cargar).pack(side='left',padx=3)
        ttk.Button(barra,text='Iniciar cámara',command=self.iniciar_camara).pack(side='left',padx=3)
        ttk.Button(barra,text='Detener cámara',command=self.detener_camara).pack(side='left',padx=3)
        ttk.Button(barra,text='Analizar',command=self.analizar).pack(side='left',padx=3)
        ttk.Label(barra,text='Modelo').pack(side='left',padx=(15,4))
        ttk.Combobox(barra,textvariable=self.modelo,values=self.motor.disponibles() or ['CNN','Random Forest','SVM'],state='readonly',width=15).pack(side='left')
        ttk.Button(barra,text='Guardar resultado',command=self.guardar).pack(side='right')
        ejemplos=ttk.Frame(cont);ejemplos.pack(fill='x',pady=9)
        ttk.Label(ejemplos,text='Demostración sin cámara').pack(side='left')
        for calidad in ['buena','intermedia','mala']:
            ttk.Button(ejemplos,text='Ejemplo '+calidad,command=lambda c=calidad:self.ejemplo(c)).pack(side='left',padx=4)
        vistas=ttk.Frame(cont);vistas.pack(fill='both',expand=True)
        self.original=ttk.Label(vistas,text='Imagen de entrada',anchor='center');self.original.pack(side='left',fill='both',expand=True)
        self.procesada=ttk.Label(vistas,text='Fotografía completa utilizada en el análisis',anchor='center');self.procesada.pack(side='left',fill='both',expand=True)
        ttk.Label(cont,textvariable=self.salida,font=('Segoe UI',12),wraplength=980,justify='left').pack(anchor='w',pady=8)
        ttk.Label(cont,textvariable=self.estado,wraplength=980).pack(anchor='w')
        ttk.Label(cont,text='El tamaño depende del encuadre y la distancia. Calidad visual de apoyo; requiere revisión humana.',wraplength=980,foreground='#555555').pack(anchor='w',pady=(6,0))
        raiz.protocol('WM_DELETE_WINDOW',self.cerrar);raiz.after(100,self.revisar)

    def mostrar(self,rgb,widget):
        im=Image.fromarray(rgb);im.thumbnail((475,390));foto=ImageTk.PhotoImage(im)
        widget.configure(image=foto,text='');widget.image=foto

    def cargar(self):
        if self.futuro is not None:return
        ruta=filedialog.askopenfilename(filetypes=[('Imágenes','*.jpg *.jpeg *.png *.bmp *.webp *.heic *.heif'),('Todos','*.*')])
        if ruta:
            try:
                self.detener_camara();self.id_base=None;self.rgb=leer_rgb(ruta);self.mostrar(self.rgb,self.original);self.analizar()
            except Exception as e:messagebox.showerror('Imagen',str(e))

    def ejemplo(self,calidad):
        self.detener_camara()
        if self.futuro is not None:return
        fila=self.motor.base.loc[self.motor.base.calidad_revisada.eq(calidad)].iloc[0]
        self.id_base=fila.id_ejemplar;self.rgb=leer_rgb(fila.ruta_cnn);self.mostrar(self.rgb,self.original)
        self.analizar()

    def analizar(self):
        if self.futuro is not None:return
        if self.rgb is None:
            self.estado.set('Carga una imagen o selecciona un ejemplo.');return
        nombre=self.modelo.get()
        self.rgb_en_proceso=self.rgb.copy();self.origen_camara=self.camara is not None
        self.estado.set('Analizando imagen…')
        self.futuro=self.pool.submit(self.motor.ejemplo,self.id_base,nombre) if self.id_base else self.pool.submit(self.motor.analizar,self.rgb.copy(),nombre)
        self.ultima_inferencia=time.monotonic()

    def revisar(self):
        if self.futuro is not None and self.futuro.done():
            try:
                r,entrada,mascara=self.futuro.result();self.ultimo=r;self.mostrar(entrada,self.procesada)
                if mascara is not None and not self.origen_camara:
                    import cv2
                    vista=self.rgb_en_proceso.copy();contornos,_=cv2.findContours(mascara.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vista,contornos,-1,(20,230,80),max(1,round(max(vista.shape[:2])/400)))
                    self.mostrar(vista,self.original)
                t=r['tamano'];scores=r['puntuaciones_no_calibradas']
                detalle=' | '.join(f'{k}: {v:.1%}' for k,v in scores.items())
                linea_tamano=(f"Diámetro equivalente: {t['diametro_equivalente_px']:.1f} px | Normalizado: {t['diametro_normalizado']:.3f} | Ocupación: {t['ocupacion']:.1%}" if t else 'Tamaño no estimable: revisa el encuadre y el fondo.')
                self.salida.set(f"Calidad estimada: {r['calidad'].upper()} | Modelo: {r['modelo']}\n{linea_tamano}\nPuntuaciones sin calibrar: {detalle}")
                aviso='Ejemplo de entrenamiento. Etiqueta revisada: '+r['etiqueta_revisada']+'. No es una prueba independiente.' if self.id_base else 'Revisa que el contorno verde incluya solo el fruto y todos sus defectos.'
                self.estado.set(aviso+(' Alertas de máscara: '+', '.join(r['alertas']) if r['alertas'] else ''))
            except Exception as e:
                self.estado.set('No se pudo analizar: '+str(e));self.detener_camara()
            finally:self.futuro=None
        self.raiz.after(100,self.revisar)

    def iniciar_camara(self):
        if self.futuro is not None:return
        import cv2
        self.detener_camara();self.id_base=None;self.camara=cv2.VideoCapture(0)
        if not self.camara.isOpened():
            self.detener_camara();self.estado.set('Cámara no disponible. Puedes cargar una foto o usar los ejemplos.');return
        self.camara.set(cv2.CAP_PROP_FRAME_WIDTH,640);self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        self.fotograma()

    def fotograma(self):
        if self.camara is None:return
        import cv2
        ok,bgr=self.camara.read()
        if not ok:self.detener_camara();self.estado.set('No se pudo leer la cámara.');return
        self.rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);self.mostrar(self.rgb,self.original)
        if time.monotonic()-self.ultima_inferencia>1.5:self.analizar()
        self.raiz.after(80,self.fotograma)

    def detener_camara(self):
        if self.camara is not None:self.camara.release();self.camara=None

    def guardar(self):
        if self.ultimo is None:return
        import json
        ruta=filedialog.asksaveasfilename(defaultextension='.json',filetypes=[('Resultado JSON','*.json')])
        if ruta:
            from pathlib import Path
            Path(ruta).write_text(json.dumps(self.ultimo,ensure_ascii=False,indent=2),encoding='utf-8')

    def cerrar(self):
        self.detener_camara();self.pool.shutdown(wait=False,cancel_futures=True);self.raiz.destroy()


def iniciar():
    raiz=tk.Tk();Aplicacion(raiz);raiz.mainloop()
