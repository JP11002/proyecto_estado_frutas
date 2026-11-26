# CNN con fotografías completas

El proyecto conserva una sola base: 291 fotografías, 291 máscaras originales,
43 características, 22 grupos y las etiquetas buena (201), intermedia (75) y mala
(15). La CNN vigente recibe **toda la fotografía**, reescalada conservando la
proporción y centrada sobre relleno gris a 224×224. No se recorta alrededor de la
máscara, no se elimina el fondo y no se usa CenterCrop. Las máscaras siguen
sirviendo para las características tradicionales y el tamaño relativo.

La huella vigente de datos.csv es
`47bcd15916e6aa787ecaf1341cac98c31f392fd957499b07ca3e1592572528fc`.
El cambio de huella corresponde a las nuevas entradas CNN y su trazabilidad;
las etiquetas, los grupos y las características tradicionales se comprobaron
sin cambios. Todas las pesas actuales declaran la huella y el procesamiento.

## Modelo y búsqueda

MobileNetV3 Small preentrenada en ImageNet mantiene congelados convoluciones y
BatchNorm. Produce 576 rasgos y un cabezal propio, con LayerNorm sin parámetros.
Se comparan doce candidatos: cabezal lineal o MLP de 64 unidades, exponente de
peso inverso de clase 0, 0,5 o 1 y aumentos activados o desactivados. AdamW usa
lr 0,003 y regularización 0,01; el MLP usa dropout 0,2.

Las vistas son original, espejo horizontal, giros de ±10 grados con expansión
para conservar esquinas y brillo 0,92. Las vistas aumentadas solo intervienen
en entrenamiento; validación e inferencia usan la original. El cache incluye
huella de base, identificadores y procesamiento para evitar reutilizar rasgos
del recorte anterior. Se aplica normalización ImageNet.

Se mantienen tres bloques externos por grupos, semilla 42. Las divisiones
internas son 3, 2 y 3 y la selección global usa tres: 132 ajustes internos,
tres externos y uno final. Máximo 120 épocas, mínimo 12 y paciencia 15; selección
por F1 macro interno y duración media de mejores épocas internas. El prototipo
final seleccionó cabezal lineal, exponente 0,5, aumentos activados y 20 épocas
(1731 parámetros entrenables). Los bloques externos eligieron 18, 16 y 14 épocas;
el primero usa aumentos y los otros dos no. La puntuación externa no se usa para
elegir el cabezal final.

## Resultados externos

| Modelo | Exactitud | F1 macro | Recall macro |
|---|---:|---:|---:|
| Mayoritaria | 0,691 | 0,272 | 0,333 |
| SVM inicial | 0,395 | 0,299 | 0,365 |
| RF inicial | 0,546 | 0,429 | 0,479 |
| SVM ajustado | 0,447 | 0,373 | 0,414 |
| RF ajustado | 0,443 | 0,388 | 0,430 |
| CNN simple (histórica) | 0,271 | 0,237 | 0,393 |
| CNN recortada (histórica) | 0,704 | 0,472 | 0,501 |
| CNN foto completa | 0,674 | 0,458 | 0,505 |
| Logística CNN | 0,646 | 0,400 | 0,467 |
| SVM CNN | 0,674 | 0,430 | 0,462 |
| Extra Trees | 0,423 | 0,357 | 0,396 |
| Gradient Boosting | 0,649 | 0,487 | 0,510 |

CNN completa acierta 176/201 buenas, 13/75 intermedias y 7/15 malas. Su matriz
(filas y columnas: buena, intermedia, mala) es [[176,6,19],[57,13,5],[6,2,7]].
La precisión intermedia es 61,9 % y su recall 17,3 %. La referencia recortada
acertaba seis intermedias (recall 8,0 %). Conservar la foto completa mejora esa
recuperación, pero reduce F1 macro de 0,472 a 0,458 y exactitud de 70,4 a 67,4 %.
Con igual peso por grupo, F1 macro pasa de 0,531 a 0,376. No hay una mejora global.
Los tres F1 macro externos actuales son 0,495, 0,462 y 0,207.

Gradient Boosting sobre las 43 características mantiene F1 macro 0,487 y 19/75
intermedias correctas, con precisión intermedia 41,3 % y exactitud 64,9 %.
Las otras alternativas sobre rasgos CNN se recalcularon con las fotos completas.
La comparación principal verifica 2328 predicciones de ocho modelos y el
diagnóstico 3492 de doce. CNN simple y CNN recortada se conservan como referencias
históricas con las mismas identidades, etiquetas, máscaras, grupos y bloques.
Los pesos de inferencia vigentes usan únicamente fotografías completas.

## Por qué sigue fallando la calidad intermedia

70 de 75 intermedias son bananos y 67 pertenecen a una sola campaña. Solo cinco
grupos contienen intermedias. En el tercer bloque el entrenamiento dispone de
ocho intermedias y la evaluación reserva la campaña de 67. La CNN actual confunde
57 intermedias con buenas y cinco con malas. Cambiar el encuadre no resuelve la
escasez de campañas independientes ni la cobertura fruta/calidad. El fondo y el
menor tamaño del fruto al conservar capturas lejanas pueden afectar al extractor;
son hipótesis, no causas demostradas. Algunos originales heredados contienen
varios frutos aunque la etiqueta describa el ejemplar revisado.

Las etiquetas son visuales y no tienen acuerdo entre anotadores medido. No hay
una escala física continua que haga intermedia igual al promedio de buena y
mala. Los softmax no son confianza calibrada. Son resultados exploratorios de
desarrollo, no una prueba final independiente. No se cambiaron etiquetas para
mejorar las cifras.

## Ejecutar y verificar

Seguir docs/instalacion.md. Desde la raíz del proyecto:

```text
python -m src.training.entrenar_modelos --proyecto . --epocas-cnn 120 --hilos-cpu 4
python scripts/diagnosticar_intermedia.py --comparar-alternativas
python scripts/diagnosticar_intermedia.py --verificar
python scripts/verificar_comparacion.py
python scripts/ejecutar_cuaderno.py notebooks/08_comparacion_cnn.ipynb --timeout 10800
python -m src.main
```

El entrenamiento ya está completado en este equipo. Los cuadernos pueden
reutilizar los resultados comprobando la huella con FRUTAS_USAR_RESULTADOS=1;
sin esa variable vuelven a entrenar. Pasaron 18 pruebas y se verificó la recarga
de los pesos por bloque reproduciendo sus etiquetas y probabilidades.

Los paquetes cnn_pliegue_1.pt, cnn_pliegue_2.pt, cnn_pliegue_3.pt y cnn_calidad.pt
se guardan en experiments/checkpoints/comparacion_modelos/. Incluyen el extractor
completo, cabezal, procesamiento y huella; la aplicación no necesita descargar
pesos ni disponer de los originales. Los resultados y modelos se excluyen de
Git y se incluyen en el ZIP local de entrega. La base preparada sí se comparte.

En la aplicación la CNN analiza la foto completa incluso si no se obtiene una
máscara válida; en ese caso se muestra que el tamaño no es estimable. SVM y
Random Forest requieren la segmentación para calcular sus características.
