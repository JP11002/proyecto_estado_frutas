# Ajuste de modelos y CNN

Este documento reúne el modelado del proyecto. La sección 1 es la evaluación vigente
(base de 332 imágenes, prueba independiente por grupos, cuaderno 09). Las secciones 2 y 3
conservan, como historial, el ajuste tradicional y la CNN con el protocolo anterior de tres
bloques (cuadernos 07 y 08); sus cifras corresponden a la base de 291 imágenes.

## 1. Experimentos v2: base ampliada, prueba independiente y decisiones justificadas

Código: [`src/models/experimentos_base_ampliada.py`](../src/models/experimentos_base_ampliada.py).
Figuras: [`scripts/figuras_experimentos_v2.py`](../scripts/figuras_experimentos_v2.py).
Salidas: `experiments/results/experimentos_v2/` (`metricas.csv`, `matrices_confusion.csv`,
`metricas_por_fuente_y_fruta.csv`, `predicciones.csv`, `ajustes.csv`, `decisiones.json`, `seleccion.json`).

```text
python -m src.models.experimentos_base_ampliada --proyecto .
python scripts/figuras_experimentos_v2.py
```

### 1. Decisiones de metodología y su justificación

| # | Decisión | Por qué |
|---|---|---|
| D1 | **Prueba independiente por grupos, congelada** en `data/base/particion_desarrollo_prueba.csv` | Las cifras anteriores eran solo de desarrollo. Una prueba que no participa en ninguna elección estima la generalización sin sesgo de selección. |
| D2 | Regla de prueba: entre `StratifiedGroupKFold` de 4, 5 y 6 bloques (semilla 42), el bloque con las tres clases, **sin campañas de más del 10 %** de la base y con **más grupos**; desempate por distribución de clases más cercana a la global | La unidad independiente es el grupo (fruto o campaña), no la foto. Tres campañas reúnen 245 fotos; si una entra en prueba, la prueba mide un solo grupo y el entrenamiento pierde casi toda una combinación fruta-calidad. La regla usa solo etiquetas y grupos, y se fijó antes de entrenar. |
| D3 | Desarrollo con **validación cruzada externa agrupada de 5 bloques** y **búsqueda de hiperparámetros anidada** (validación interna agrupada, hasta 3 bloques) | Los grupos evitan que vistas del mismo fruto queden a ambos lados (fuga). La anidación evita estimar el desempeño con los mismos datos que eligieron los hiperparámetros. |
| D4 | **Métrica de selección: F1 macro** de las predicciones externas de desarrollo; desempate por recall macro | La clase buena es el 63 %: la exactitud premia predecir la mayoritaria (referencia: 66 % de exactitud con F1 macro 0,27). F1 macro pesa igual las tres clases y combina recall (no perder intermedias y malas) con precisión (no inventar defectos). |
| D5 | **Selección del modelo final antes de mirar la prueba** (`seleccion.json` se escribe antes de predecir la prueba) | Si se eligiera por la prueba, esta dejaría de ser independiente. |
| D6 | **Dos condiciones de datos con las mismas filas de evaluación**: `anterior` (sin el lote del 26/11) y `ampliada` | Aísla el efecto de los datos nuevos del efecto del modelo. Las cifras del informe anterior usaban otro protocolo y no son comparables directamente. |
| D7 | **Pesos de clase** (`class_weight="balanced"`, o pesos en la entropía cruzada) en lugar de sobremuestreo o SMOTE | Duplicar o interpolar fotos de pocas campañas no crea variabilidad independiente; con grupos, además, puede inflar la validación. |
| D8 | Variante **"+umbral"**: probabilidades × pesos (1, a, b) elegidos en predicciones internas para maximizar F1 macro; a, b ∈ [0,25; 8] en escala logarítmica | Regla de Bayes con costes por clase: corrige el sesgo hacia la mayoritaria y actúa directamente sobre el recall de intermedia y mala. Se ajusta solo con datos de entrenamiento. |
| D9 | **Rasgos de MobileNetV3 Small congelada** (576 por imagen, ImageNet) | Con unas 300 imágenes no es viable aprender filtros profundos. La red congelada no ve etiquetas (sin fuga) y aporta textura y bordes que las 43 características manuales no capturan. |
| D10 | **Modelo profundo**: cabezal propio (lineal o MLP de 64) con AdamW, pesos de clase y aumentos (espejo, giros de 10°, brillo) solo en entrenamiento; 12 configuraciones elegidas por validación interna | Cumple "transfer learning con capas congeladas y capas propias" del enunciado. Los aumentos se aplican dentro del entrenamiento, nunca como copias en la base. |
| D11 | Modelos nuevos: **regresión logística**, **HistGradientBoosting** (sin parada temprana aleatoria) y **fusión** de rasgos manuales y CNN | Logística: referencia lineal regularizada, estable con pocos datos. HistGB: interacciones no lineales con regularización L2; la parada temprana usaría una partición aleatoria que mezclaría grupos. Fusión: combina forma y color en la máscara con textura global. |
| D12 | **Intervalos al 95 % por bootstrap de grupos** (2 000 remuestreos) en la prueba | Con pocos grupos independientes, remuestrear fotos sueltas subestimaría la incertidumbre. |
| D14 | **Pesos por combinación fruta-calidad** (variantes `_pesos_fruta`): w_i = N / (C·n_celda) | El cuaderno 08 mostró que la CNN aprendió la fruta como atajo: de 10 bananos buenos predijo 9 como intermedios, y no acertó ninguno de los 11 limones intermedios. Con pesos solo por clase, "banano = intermedia" sigue siendo rentable; con pesos por combinación, cada celda fruta-calidad pesa lo mismo y ese atajo deja de rendir. |
| D15 | **Qué cuenta como modelo profundo**: clasificadores sobre la CNN congelada (cabezal propio entrenado con AdamW, o capa softmax lineal con regularización L2) | Una regresión logística multinomial sobre los rasgos de la red es matemáticamente un cabezal lineal softmax con entropía cruzada y regularización L2: es transferencia con capas congeladas y una capa propia. La fusión (rasgos manuales + CNN) se trata como híbrida. |
| D13 | **Métricas por fuente y por fruta** | El lote nuevo tiene fondos y resoluciones distintos: si el modelo aprende el fondo, el recall subiría solo en esa fuente. El desglose permite detectarlo. |

Las rejillas de hiperparámetros se fijaron antes de ver resultados. SVM ajustado y RF
ajustado usan exactamente las rejillas del proyecto (`espacios_hiperparametros`); RF
inicial usa la configuración fija del cuaderno 06.

### 2. Resultados (cuaderno 09, ejecutado el 26/11/2026)

Partición congelada: desarrollo 289 imágenes y 20 grupos (191/78/20); prueba 43 imágenes y
11 grupos (18/16/9), de las fuentes lote 26/11 (21), Kaggle (11) y propia (11).

| Modelo | F1 desarrollo | F1 prueba (IC95 %) | Recall int. (prueba) | Recall mala (prueba) |
|---|---:|---:|---:|---:|
| Referencia mayoritaria | 0,265 | 0,197 | 0,00 | 0,00 |
| RF inicial | 0,291 | 0,417 (0,25–0,52) | 0,31 | 0,56 |
| SVM ajustado | 0,420 | 0,324 (0,18–0,45) | 0,31 | 0,33 |
| CNN MobileNetV3 + cabezal propio | 0,480 | 0,397 (0,18–0,54) | 0,50 | 0,22 |
| Logística sobre CNN | 0,483 | 0,385 (0,20–0,57) | 0,63 | 0,22 |
| Logística CNN + pesos fruta + costes (mejor profundo) | 0,542 | 0,428 (0,25–0,59) | 0,56 | 0,22 |
| **Fusión + pesos fruta + costes (modelo final)** | **0,572** | **0,395 (0,19–0,58)** | 0,50 | 0,22 |

Todas las cifras, incluidas las variantes, están en `metricas.csv`.

**Hallazgos**

1. **Atajo por fruta.** Con pesos solo por clase, los modelos asocian banano con intermedia y
   limón o manzana con buena. Con el protocolo anterior de 3 bloques (cuaderno 08) eso daba
   F1 0,660 y 80,7 % de exactitud a la CNN, pese a acertar 0 de 11 limones intermedios.
   Con la misma logística sobre CNN, los pesos fruta-calidad suben en limón el recall de
   intermedia de 0,11 a 0,89 y el F1 de 0,59 a 0,83 (desarrollo).
2. **Efecto del lote nuevo** (desarrollo, mismas filas): mejora los modelos con rasgos CNN
   (logística CNN 0,421 → 0,483; fusión 0,432 → 0,558) y empeora los de rasgos manuales
   (RF inicial 0,518 → 0,291; HistGB 0,440 → 0,326). Los rasgos manuales de recortes pequeños
   y borrosos cambian de distribución. En la prueba las diferencias quedan dentro de los IC95 %.
3. **Generalización.** En los modelos con rasgos CNN, el F1 interno (0,46–0,58) y el externo de
   desarrollo (0,48–0,57) son parecidos, pero la prueba baja a 0,39–0,44 porque sus fuentes difieren.
   Con 11 grupos de prueba, ninguna diferencia entre modelos útiles es significativa; todos
   superan a la referencia mayoritaria.
4. **Regla de costes.** Los pesos elegidos para el modelo final, (1; 0,25; 0,5), favorecen
   "buena" porque el desarrollo está dominado por campañas buenas: suben el F1 de desarrollo
   (0,512 → 0,572), pero en la prueba bajan el recall de mala (0,33 → 0,22). No se cambió la
   selección tras ver la prueba.
5. **Errores relevantes en la prueba.** 5 de 9 malas se predicen buenas y 10 de 18 buenas se
   marcan como defectuosas. En las fotos propias de prueba se aciertan las 5 buenas y ninguna
   de las 6 defectuosas. El sistema no debe decidir de forma autónoma.

**Validación anidada sobre las 332 imágenes** (5 bloques agrupados, semillas 42, 7 y 2026;
`anidada_332_resumen.csv`). Complementa la prueba; no se usó para elegir el modelo.

| Modelo | F1 macro (media ± desv.) | Recall int. | Recall mala |
|---|---:|---:|---:|
| Referencia mayoritaria | 0,258 | 0,00 | 0,00 |
| RF inicial | 0,396 ± 0,034 | 0,20 | 0,51 |
| SVM ajustado | 0,330 ± 0,053 | 0,23 | 0,25 |
| CNN cabezal + costes | 0,575 ± 0,097 | 0,38 | 0,59 |
| Logística CNN + costes | 0,508 ± 0,018 | 0,20 | 0,51 |
| Logística CNN + pesos fruta | 0,450 ± 0,013 | 0,21 | 0,61 |
| Modelo final (fusión + pesos fruta + costes) | 0,460 ± 0,065 | 0,19 | 0,54 |

6. **Estabilidad de la selección.** En la validación anidada el cabezal CNN obtiene la mejor
   media, con la mayor variación entre semillas (0,51–0,69). El orden entre modelos depende de
   la partición. No se cambió el modelo final porque esta validación incluye las imágenes de
   prueba. Los pesos fruta-calidad reducen el atajo (bananos buenos 0,07 → 0,43; limones
   intermedios 0,18 → 0,52, con la logística CNN) a costa de bajar el F1 macro global
   (0,472 → 0,450).

**Datos que más ayudarían:** limones y manzanas intermedios y malos, bananos buenos y más
grupos independientes (sesiones o frutos distintos).

## 2. Historial: ajuste de SVM y Random Forest (cuaderno 07)

El cuaderno `07_ajuste_modelos_tradicionales.ipynb` continúa el EDA y entrenamiento
del cuaderno 06. Se ejecutó completo sobre las mismas 291 imágenes, 43 características
y 22 grupos conservadores, sin cambiar etiquetas, máscaras ni particiones externas.

### Diseño y reproducción

Se comparan 18 configuraciones de SVM y 12 de Random Forest. Para cada uno de los
tres bloques externos se realiza una búsqueda en tres pliegues internos agrupados,
limitados al entrenamiento externo. La selección maximiza el promedio del F1 macro
interno con las tres etiquetas fijas; luego se reajusta sobre ese entrenamiento y
se predice el bloque externo. Se mantiene `random_state=42` y no se cambian los
espacios de búsqueda al observar los resultados externos.

SVM prueba kernel lineal/RBF, `C` de 0,1, 1 y 10, pesos normales/balanceados y,
para RBF, `gamma='scale'` o 0,01. `StandardScaler` se ajusta dentro del pipeline.
Random Forest conserva 300 árboles y pesos balanceados, y prueba profundidad sin
límite/8, mínimo de 1/2/4 imágenes por hoja y `max_features='sqrt'`/0,7.
Ambos espacios incluyen la configuración inicial.

Las etapas de entrenamiento contienen las tres calidades globales. Una validación
interna carece de imágenes intermedias; se documenta sin inventar muestras o
separar fotografías del mismo grupo. Los valores internos sirven para seleccionar
parámetros; las métricas siguientes reúnen las predicciones externas OOF.

Para reproducir, ejecutar el cuaderno 07 con las dependencias del proyecto. El 06
permite revisar primero el EDA; si sus resultados faltan, el 07 genera la referencia
inicial necesaria. En un clon nuevo se usa la base del proyecto, sin descargar
originales. El módulo
reutilizable es `src/models/ajuste_tradicionales.py`.

### Resultados

| Versión | Exactitud | F1 macro | Recall macro |
|---|---:|---:|---:|
| Referencia mayoritaria | 69,1 % | 0,272 | 0,333 |
| SVM inicial | 39,5 % | 0,299 | 0,365 |
| SVM ajustado | 44,7 % | 0,373 | 0,414 |
| Random Forest inicial | 54,6 % | 0,429 | 0,479 |
| Random Forest ajustado | 44,3 % | 0,388 | 0,430 |

SVM mejora 0,074 puntos de F1 macro; Random Forest disminuye 0,041. La decisión
es conservar Random Forest inicial como referencia principal del desarrollo y
SVM ajustado como alternativa. La búsqueda no justifica reemplazar automáticamente
un modelo inicial por su versión ajustada. Esta decisión es exploratoria y no
una selección confirmada mediante una prueba independiente.

El recall de intermedia pasa de 11/75 a 12/75 en SVM y permanece en 11/75 en Random
Forest. SVM ajustado reconoce 110/201 buenas y 8/15 malas; Random Forest ajustado
reconoce 109/201 buenas y 9/15 malas. El ajuste todavía no resuelve la clase
intermedia. Al dar peso total igual a cada grupo, F1 macro es 0,374 para SVM ajustado
y 0,367 para Random Forest ajustado; Random Forest inicial conserva 0,405.

### Análisis de errores

El grupo público de bananos `b3614848dd857cd2` contiene 72 imágenes y concentra 67
de las 75 intermedias de toda la base. En su evaluación externa, SVM ajustado
reconoce 11 de esas 67 intermedias, frente a 8 de su versión inicial. Random Forest
reconoce 7 con ambas versiones. Mantener esa campaña completa fuera del entrenamiento
revela una dificultad que un reparto aleatorio por fotografía podría ocultar.

Los parámetros seleccionados cambian entre bloques y las puntuaciones externas
varían mucho. Que la búsqueda incluya la configuración inicial no garantiza que
la seleccionada con otra validación interna mejore en los grupos externos. La
cobertura por fruta/calidad y la asociación entre procedencia y etiqueta limitan
la estabilidad de esta selección. Las tablas por fruta y grupo permiten localizar
los errores; la galería muestra ejemplos fijos por identificador y conserva las
etiquetas revisadas.

### Modelos y archivos guardados

Se guardan dos prototipos nuevos tras una búsqueda por grupos sobre toda la base
de desarrollo. Sus parámetros pueden diferir de los seleccionados en cada bloque:

- SVM: kernel lineal, `C=0.1`, `class_weight=None`, con escalado en pipeline.
- Random Forest: 300 árboles, `max_depth=None`, `min_samples_leaf=2`,
  `max_features=0.7`, `class_weight='balanced'`, semilla 42.

La puntuación usada en esta búsqueda final no se interpreta como prueba adicional.
Las métricas OOF evalúan el procedimiento de selección dentro de cada entrenamiento,
no un único prototipo final con parámetros idénticos en todos los bloques.

En `experiments/checkpoints/ajuste_modelos/` quedan
`svm_ajustado.joblib`, `random_forest_ajustado.joblib` y `metadata.json`.
Los archivos incluyen modelos, orden de características, clases, parámetros y huella
de la base. Las versiones iniciales permanecen en su carpeta original.

En `experiments/results/ajuste_modelos/` quedan el plan de búsqueda, particiones
externas e internas, cobertura, resultados de todos los candidatos, parámetros
seleccionados, predicciones OOF, métricas globales/por clase/fruta/pliegue, errores
por grupo, matrices de confusión y galería de errores.

Se verificó la separación de grupos en los tres bloques externos y sus nueve
divisiones internas, las 1455 predicciones únicas de las cinco versiones,
la coincidencia de las referencias anteriores y la selección por puntuación
interna. Las métricas se recalcularon desde las predicciones. Ambos prototipos se
recargaron correctamente y el cuaderno terminó sin errores.

### Continuación del proyecto

Esta fase deja entrenados y comparados los dos modelos tradicionales, con un
experimento inicial y otro de ajuste documentados. Se puede continuar con la
inferencia desde fotografías y el modelo de aprendizaje profundo exigido por el
proyecto, conservando estos modelos como referencia. Las próximas capturas deben
priorizar manzanas/limones intermedios, frutos malos y bananos buenos de campañas
distintas. La prueba final debe reservar capturas independientes; no es un requisito
para continuar ahora con la implementación.

## 3. Historial: CNN con fotografía completa (cuaderno 08)

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

### Modelo y búsqueda

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

### Resultados externos

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

### Por qué sigue fallando la calidad intermedia

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

### Ejecutar y verificar

Seguir la sección de instalación del README. Desde la raíz del proyecto:

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
