# Mejora de CNN y comparación con SVM y Random Forest

El cuaderno `08_comparacion_cnn.ipynb` ejecuta la CNN mejorada dentro del mismo
proyecto. Se mantiene **una sola base**, `data/base/`, con 291 imágenes,
43 características tradicionales y 22 grupos. No se modificaron las etiquetas,
las máscaras, los recortes ni las identidades para obtener esta mejora.
La huella del CSV sigue siendo `ce36085062344eed79c95062be33b2410b4320a600dbe517db849c4e71414de1`.

## Cambio de modelo

La CNN simple desde cero obtuvo F1 macro 0,237 y confundió muchas
imágenes buenas con intermedias o malas. Se reemplazó el entrenamiento actual por
**MobileNetV3 Small preentrenada en ImageNet, con extractor congelado y un nuevo
clasificador de las tres calidades**. Las convoluciones y sus estadísticas
BatchNorm permanecen fijas. El procedimiento sigue el enfoque de extractor fijo
descrito en el [tutorial oficial de transferencia de PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial).
Los pesos proceden del [modelo oficial de torchvision](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).

Se utiliza la imagen preparada completa de 224×224, sin recorte central adicional,
y normalización ImageNet: media (0,485; 0,456; 0,406), desviación (0,229; 0,224; 0,225).
El extractor produce 576 características por vista. El cabezal comienza con
LayerNorm sin parámetros y puede ser lineal o MLP de 64 unidades. Se fijan AdamW,
tasa 0,003, regularización 0,01 y dropout 0,2.

Se comparan seis candidatos: los dos cabezales combinados con exponentes 0, 0,5 y
1 sobre el peso inverso de frecuencia. Las frecuencias se calculan solo en el
entrenamiento de cada ajuste. El exponente 0 no pondera las clases; 0,5 modera el
efecto del desbalance y 1 conserva la ponderación inversa completa como control.

Para acelerar el cálculo en CPU se cachean cinco vistas deterministas: original,
espejo horizontal, giros de ±10° y brillo 0,92. Cada época toma una vista por
imagen de entrenamiento. La validación usa únicamente la vista original. El cache
no utiliza etiquetas ni aprende estadísticas de la base; incluye la huella, los
identificadores, la normalización y la versión del extractor para validarse.

## Evaluación y selección

Se conservan los mismos tres bloques externos por grupos y semilla 42 de los
cuadernos 06/07. Dentro de cada entrenamiento externo se prueban los seis candidatos
con hasta tres divisiones internas por grupos. El máximo es 120 épocas, con selección
desde la época 12 y paciencia 15. Se maximiza F1 macro interno y se desempatan las
épocas con menor pérdida interna no ponderada. El candidato ganador maximiza el
F1 macro interno medio; la duración se fija por la media redondeada de sus mejores
épocas internas, entre 12 y el máximo. Luego se reajusta sobre todo el entrenamiento
externo y se predice su bloque de evaluación una sola vez.

Los parámetros de esta corrida se fijaron antes de consultar sus resultados
externos. La arquitectura se propuso tras analizar la CNN anterior: por ello son
**resultados exploratorios de desarrollo**, incluso manteniendo validación anidada.
No constituyen una nueva prueba final independiente ni permiten afirmar una
superioridad estadísticamente demostrada con solo tres bloques.

Las predicciones externas de SVM y Random Forest se reutilizan solo después de
verificar huella de base, imágenes, etiquetas, fuentes, grupos y bloques. Se
conservan las predicciones de la CNN simple anterior como referencia histórica
en el mismo informe; los pesos CNN actuales son los de transferencia.

## Resultados externos

| Modelo | Exactitud | F1 macro | Recall macro |
|---|---:|---:|---:|
| Referencia mayoritaria | 69,1 % | 0,272 | 0,333 |
| SVM inicial | 39,5 % | 0,299 | 0,365 |
| Random Forest inicial | 54,6 % | 0,429 | 0,479 |
| SVM ajustado | 44,7 % | 0,373 | 0,414 |
| Random Forest ajustado | 44,3 % | 0,388 | 0,430 |
| CNN simple | 27,1 % | 0,237 | 0,393 |
| CNN transferencia | 70,4 % | 0,472 | 0,501 |

La CNN actual mejora 0,236 puntos absolutos de F1
macro respecto a la CNN anterior y 0,044 frente a
Random Forest inicial. Estas cifras se calculan sobre las 291 predicciones
externas de cada modelo, no sobre el entrenamiento del prototipo final.

| Calidad | Imágenes | Aciertos CNN actual | Recall CNN actual | Aciertos RF inicial |
|---|---:|---:|---:|---:|
| Buena | 201 | 192 | 95,5 % | 139 |
| Intermedia | 75 | 6 | 8,0 % | 11 |
| Mala | 15 | 7 | 46,7 % | 9 |

La recuperación de imágenes buenas aumenta, pero la categoría intermedia sigue
siendo el principal problema: solo 6 de 75
frente a 11 de Random Forest inicial.
La CNN anterior acertaba 57 intermedias a costa de clasificar como buenas solo
17 de 201. La mejora del F1 global no representa una mejora en todas las clases.
Hay únicamente 15 ejemplos malos y 67 de las 75 intermedias pertenecen a una misma
campaña pública de banano. Se necesitan más frutos y campañas independientes,
especialmente intermedios, para evaluar mejor esa transición de calidad.

Al dar el mismo peso total a cada grupo, el F1 macro de CNN es
**0,531**, frente a **0,405**
de Random Forest inicial y **0,374**
de SVM ajustado. Esta sensibilidad debe acompañar la lectura de las cifras por imagen.
El cuaderno incluye el desglose por bloque, fruta y fuente y las matrices completas.
Los F1 macro de los tres bloques son 0,552; 0,547 y 0,266. La diferencia entre
bloques muestra inestabilidad ante las campañas reservadas, especialmente en
la clase intermedia. El resultado global no elimina esa limitación.

## Prototipo actual

La auditoría registra 66 ajustes internos, tres externos y uno final.
Se utilizaron 3, 2 y 3 divisiones internas en los bloques externos, y 3 en la
selección global. La cantidad se adapta a los grupos disponibles por clase.
Los bloques externos seleccionaron 17, 18, 17 épocas, respectivamente.
La búsqueda global seleccionó cabezal `lineal`, exponente de peso
`1.0` y **23 épocas** para entrenar con toda la
base de desarrollo. Tiene 1731 parámetros entrenables.
Esta selección interna no sustituye las puntuaciones externas de comparación.

Los pesos se guardan en `experiments/checkpoints/comparacion_modelos/`:
`cnn_pliegue_1.pt`, `cnn_pliegue_2.pt`, `cnn_pliegue_3.pt` y `cnn_calidad.pt`.
Cada paquete declara `arquitectura=mobilenet_v3_small_congelada` y la huella de
base. El final incluye todo el extractor y el cabezal: al recargarlo no se
descargan pesos ni se depende de una ruta absoluta del equipo de entrenamiento.
La celda «Modelos guardados» del cuaderno muestra la recarga correcta.

## Ejecución en otro computador

Después de seguir [la instalación](instalacion.md), ejecutar desde el proyecto:

```text
python scripts/comprobar_proyecto.py --cnn
python scripts/ejecutar_cuaderno.py notebooks/08_comparacion_cnn.ipynb --timeout 10800
python scripts/verificar_comparacion.py
```

También puede entrenarse desde terminal:

```text
python -m src.training.entrenar_modelos --proyecto . --epocas-cnn 120 --hilos-cpu 4
```

La primera ejecución descarga unos 10 MB de pesos oficiales en
`data/external/torch/checkpoints/`. Después reutiliza esa descarga. No requiere
GPU ni recuperar fotos originales. El comprobador de arquitectura no descarga
pesos. Las salidas y los modelos generados siguen excluidos por `.gitignore`;
`data/base/`, el código y el cuaderno permanecen compartidos en Git.

## Archivos y verificación

La misma carpeta `experiments/results/comparacion_modelos/` contiene el plan,
auditoría de particiones, búsquedas internas, épocas, probabilidades CNN, métricas
y predicciones externas consolidadas. `cnn_referencia_anterior.csv` y su JSON
preservan únicamente la referencia histórica anterior. `cnn_caracteristicas.pt`
es un cache de cálculo regenerable. Las probabilidades softmax no están calibradas.

Se comprobaron **2037 predicciones externas de siete modelos**, con una por imagen
y modelo. Los pesos de cada bloque reproducen las etiquetas y probabilidades
guardadas. Se verificaron las separaciones por grupo, las selecciones internas y
globales, la integridad de la base y la recarga final. Pasaron las diez pruebas
del proyecto, incluidas las que comprueban que BatchNorm permanece congelada y
que las vistas aumentadas de validación no influyen en el entrenamiento.

## Diagnóstico de la calidad intermedia y otros modelos — 26 de noviembre de 2026

Se añadió este análisis al mismo cuaderno 08. Se conservan la base, sus grupos y
las salidas del experimento CNN; las alternativas amplían el informe sin cambiar
los pesos del prototipo actual.

### Hallazgos comprobados

- De 75 intermedias, 70 son bananos, 3 manzanas y 2 limones. Solo cinco grupos
  contienen intermedias. Las 67 de una campaña pública representan el 89,3 % de la clase.
- En el tercer bloque, las 67 intermedias de esa campaña son evaluación externa.
  El entrenamiento contiene solo 8 intermedias: 3 bananos, 3 manzanas y 2 limones.
  La CNN reconoce 3/67 de esa campaña y 3/8 de las demás intermedias.
- En la evaluación del segundo bloque, las dos intermedias de limón pertenecen a
  un único grupo; el entrenamiento no contiene ningún limón intermedio. Esto es
  falta de soporte de esa combinación fruta/calidad, aunque existan las tres
  calidades globales en entrenamiento.
- La CNN clasifica 63 intermedias como buenas, 6 como malas y acierta 6. En 59 de
  las 75 intermedias, esa clase ocupa el segundo lugar de probabilidad. La mediana
  de su probabilidad es 0,123; en 21 errores hacia buena, la red asigna a buena al
  menos 0,9. Son probabilidades softmax sin calibrar, no confianza validada.
- Las anotaciones son decisiones de apariencia externa de un anotador. Las
  imágenes públicas de esta campaña llegaron sin etiqueta y recibieron revisión
  visual. No hay un acuerdo entre anotadores medido ni una escala física continua
  de deterioro. Por eso una etiqueta intermedia no equivale al punto medio de la
  probabilidad de buena y mala.

La evidencia apunta principalmente a concentración por campaña, confusión entre
fruta y calidad, y una transición visual heterogénea. Son explicaciones plausibles,
no causas aisladas mediante un experimento controlado. La red tiene el extractor
ImageNet congelado: que sus rasgos representen mejor la identidad del fruto que
lesiones locales es una hipótesis adicional que requiere comparar representaciones.

Se inspeccionaron dos galerías: doce ejemplos repartidos por probabilidad dentro
de la campaña dominante y las ocho intermedias restantes. Se observan diferencias
de color/madurez, intensidad de manchas, resolución y contornos aproximados. Esto
orienta una revisión de criterios; no justifica cambiar etiquetas según aciertos
o probabilidades. No se modificó ninguna anotación.

### Comparación exploratoria de alternativas

Se probaron cuatro familias con espacios fijados antes de consultar sus resultados:
regresión logística regularizada (6 candidatos), SVM sobre los rasgos CNN (8),
Extra Trees (4) y HistGradientBoosting (4). Los dos primeros usan 576 rasgos
MobileNet de la vista original; los árboles utilizan las 43 características
existentes de color, textura y forma. No se añadieron imágenes ni se usaron
identificadores, fuente o grupo como variables predictoras.

La búsqueda selecciona F1 macro interno medio, con hasta tres divisiones por
grupos y las mismas semillas que CNN. StandardScaler se ajusta dentro de cada
pipeline de logística/SVM. Boosting tiene desactivada la parada temprana automática
para evitar una validación aleatoria que mezcle grupos. Son 176 ajustes internos
y 12 externos. Este cribado no entrena prototipos finales de las alternativas.

| Modelo | Exactitud | F1 macro | Intermedias correctas | Precisión intermedia | Recall intermedia |
|---|---:|---:|---:|---:|---:|
| CNN transferencia | 70,4 % | 0,472 | 6/75 | 46,2 % | 8,0 % |
| Gradient Boosting | 64,9 % | 0,487 | 19/75 | 41,3 % | 25,3 % |
| Random Forest inicial | 54,6 % | 0,429 | 11/75 | 18,6 % | 14,7 % |
| SVM ajustado | 44,7 % | 0,373 | 12/75 | 13,8 % | 16,0 % |
| SVM CNN | 67,0 % | 0,420 | 6/75 | 54,5 % | 8,0 % |
| Logistica CNN | 65,3 % | 0,401 | 4/75 | 44,4 % | 5,3 % |
| Extra Trees | 42,3 % | 0,357 | 10/75 | 11,0 % | 13,3 % |

HistGradientBoosting recupera 19 intermedias frente a 6 de CNN, y su F1 macro
es 0,487 frente a 0,472. En la campaña dominante reconoce 17/67 frente a 3/67.
Tiene un coste: clasifica como intermedias 46 imágenes, de las que 27 pertenecen
a otras calidades; CNN predice 13 intermedias, con 7 falsos positivos. Además,
los aciertos en buenas bajan de 192 a 163, mientras que los aciertos en malas
se mantienen en 7. La exactitud baja de 70,4 % a 64,9 %.

Con peso total igual por grupo, CNN conserva F1 macro 0,531 y Boosting obtiene
0,484. Por tanto, Boosting es el candidato más prometedor del cribado para mejorar
la recuperación de intermedias; el resultado no demuestra superioridad en todos
los criterios. El incremento global es pequeño y sigue habiendo pocos grupos.
Cambiar la familia por sí solo no resuelve la cobertura de entrenamiento.

### Mejoras prioritarias con la base actual

1. **Revisar el criterio visual por fruta.** Diferenciar cambio de madurez/color,
   defectos superficiales localizados y deterioro extendido. Registrar evidencia
   visible y una segunda revisión de los casos fronterizos, sin etiquetar según
   predicciones ni cambiar grupos para subir las métricas. La calidad intermedia
   y la incertidumbre del clasificador se tratan por separado.
2. **Priorizar Boosting y probar fusión de características.** Combinar las medidas
   dentro de las máscaras con los rasgos CNN, ajustando escalado, selección de
   características o PCA dentro de cada entrenamiento interno. Evaluar la ganancia
   frente al Boosting de 43 características; no asumirla de antemano.
3. **Elegir costes y umbrales dentro de la validación interna.** Comparar F1,
   precisión y recall de intermedia, y el daño a las otras clases. La CNN ya probó
   pesos de clase con exponentes 0, 0,5 y 1; simplemente aumentarlos no aporta
   información de campañas ausentes. No elegir el umbral con las 75 predicciones
   externas. Cualquier calibración también debe respetar los grupos.
4. **Probar un ajuste parcial de la CNN.** Descongelar solo el último bloque con
   tasa pequeña, manteniendo el resto fijo, y comparar dentro del mismo esquema.
   Los aumentos que cambien el color deben ser leves, porque ese color interviene
   en la calidad visual. La ganancia de esta propuesta aún no se ha medido.
5. **Ponderar la campaña durante el entrenamiento.** Probar pesos por grupo junto
   con pesos de clase, evitando que muchas vistas dominen un ajuste. Rotaciones,
   sobremuestreo y SMOTE no crean variedad independiente ni solucionan combinaciones
   fruta/calidad ausentes. No aplicar transformaciones o remuestreo antes de dividir.

Como otras arquitecturas visuales se pueden estudiar
[ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
o [EfficientNet-B0](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html)
preentrenadas. Su mayor capacidad no garantiza una mejora con 291 imágenes;
priorizaría primero el Boosting medido y la fusión. Un modelo ordinal sería una
hipótesis posterior, cuando el criterio de deterioro sea consistente por fruta;
no hay una escala continua confirmada con la que evaluar una regresión actualmente.

### Reproducción y comprobación

```text
python scripts/diagnosticar_intermedia.py --comparar-alternativas
python scripts/diagnosticar_intermedia.py --verificar
```

El cuaderno 08 ejecuta también estas etapas. Sin el indicador de alternativas,
el script genera solo el diagnóstico, sin volver a entrenar modelos. Las salidas
`diagnostico_*` se guardan en la misma carpeta de comparación: cobertura,
confusiones por grupo, lista de 75 intermedias, galerías, plan de alternativas,
búsqueda, selección, membresías y predicciones. No son otra base de datos.

Se recalcularon las métricas de las 3201 predicciones de 11 modelos y se
reprodujeron los doce ajustes externos de las alternativas. Se comprobó que las
referencias previas conservan sus predicciones y que los grupos están separados.
Los resultados siguen siendo de desarrollo; la repetición de experimentos sobre
la misma base no equivale a una prueba final independiente.

Referencias técnicas: [validación agrupada](https://scikit-learn.org/stable/modules/cross_validation.html),
[regresión logística](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[SVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html),
[Extra Trees](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html),
[HistGradientBoosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html)
y [transferencia en PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial).
Como apoyo metodológico del flujo scikit-learn se utilizó Scientific Agent Skills:
Kassis, T., Agarwal, V., He, Y., Patel, D., y Brueckner, A. M. (2026).
[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).
