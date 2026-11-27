from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
p=Path(__file__).resolve().parents[1];out=p/'docs/figuras';out.mkdir(parents=True,exist_ok=True);plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
def guardar(fig,nombre):
 for ext in ['svg','pdf','png']:fig.savefig(out/f'{nombre}.{ext}',bbox_inches='tight',dpi=180)
 plt.close(fig)
m=pd.read_csv(p/'experiments/results/comparacion_modelos/diagnostico_alternativas_metricas.csv').set_index('modelo')
# Figura histórica del protocolo de 3 bloques; las referencias ausentes en esta copia se omiten.
etiquetas={'SVM_ajustado':'SVM\najustado','Random_Forest_inicial':'Random\nForest','CNN_recortada':'CNN\nrecortada','CNN_transferencia':'CNN\ncompleta','Gradient_Boosting':'Gradient\nBoosting'}
nombres=[n for n in etiquetas if n in m.index]
fig,ax=plt.subplots(figsize=(3.5,2.6));x=np.arange(len(nombres));w=.37
ax.bar(x-w/2,m.loc[nombres,'f1_macro'],w,label='F1 macro',color='#245c82');ax.bar(x+w/2,m.loc[nombres,'recall_intermedia'],w,label='Recall intermedia',color='#d48b26')
ax.set_xticks(x,[etiquetas[n] for n in nombres]);ax.tick_params(axis='x',labelsize=7);ax.tick_params(axis='y',labelsize=8);ax.set_ylim(0,1);ax.set_ylabel('Puntuación',fontsize=8);ax.legend(fontsize=7,frameon=False,ncol=2);ax.spines[['top','right']].set_visible(False);fig.tight_layout();guardar(fig,'resultados_modelos')
b=pd.read_csv(p/'data/base/datos.csv');d=pd.crosstab(b.fruta,b.calidad_revisada)[['buena','intermedia','mala']]
fig,ax=plt.subplots(figsize=(6.6,2.5));d.plot.barh(ax=ax,color=['#2d7759','#e1a13e','#a94748']);ax.set_xlabel('Imágenes');ax.set_ylabel('');ax.legend(frameon=False,ncol=3,loc='lower right');ax.spines[['top','right']].set_visible(False);fig.tight_layout();guardar(fig,'clases_fruta')
cm=pd.read_csv(p/'experiments/results/comparacion_modelos/matriz_confusion_CNN_transferencia.csv').set_index('real');fig,ax=plt.subplots(figsize=(3.8,3.1));ax.imshow(cm.to_numpy()/cm.sum(axis=1).to_numpy()[:,None],vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(3),cm.columns);ax.set_yticks(range(3),cm.index);ax.set_xlabel('Predicción');ax.set_ylabel('Etiqueta revisada')
for i in range(3):
 for j in range(3):ax.text(j,i,str(cm.iloc[i,j]),ha='center',va='center',color='white' if cm.iloc[i,j]/cm.iloc[i].sum()>.55 else '#111111')
fig.tight_layout();guardar(fig,'confusion_cnn')
fig,ax=plt.subplots(figsize=(3.5,3.5));ax.set_xlim(0,3);ax.set_ylim(0,3);ax.axis('off')
textos=[('Negocio','Apoyo a la inspección\nvisual'),('Datos','Kaggle, Drive, propias\ny deduplicación'),('Preparación','332 fotos revisadas,\nmáscaras y 43 rasgos'),('Modelado','SVM, RF, CNN, fusión;\najuste anidado'),('Evaluación','Prueba por grupos;\nerrores por fruta'),('Despliegue','Foto o cámara\ny tamaño relativo')]
coords=[(.05,2.12),(1.55,2.12),(1.55,1.08),(1.55,.04),(.05,.04),(.05,1.08)]
for (titulo,detalle),(x,y) in zip(textos,coords):
 ax.add_patch(FancyBboxPatch((x,y),1.35,.75,boxstyle='round,pad=0.035',facecolor='#edf3f6',edgecolor='#245c82'))
 ax.text(x+.675,y+.49,titulo,ha='center',fontweight='bold',fontsize=9);ax.text(x+.675,y+.20,detalle,ha='center',fontsize=7)
for i in range(6):
 x,y=coords[i];xx,yy=coords[(i+1)%6];dx,dy=xx-x,yy-y
 a=(x+.675+np.sign(dx)*.72,y+.375+np.sign(dy)*.44);z=(xx+.675-np.sign(dx)*.72,yy+.375-np.sign(dy)*.44)
 ax.annotate('',xy=z,xytext=a,arrowprops={'arrowstyle':'->','color':'#245c82','lw':1.5})
fig.tight_layout();guardar(fig,'crisp_dm')
print('Figuras vectoriales SVG/PDF y vistas PNG exportadas.')
