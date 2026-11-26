# Ajuste de SVM y Random Forest — 24 de noviembre de 2026

El cuaderno `07_ajuste_modelos_tradicionales.ipynb` continúa el EDA y entrenamiento
del cuaderno 06. Se ejecutó completo sobre las mismas 291 imágenes, 43 características
y 22 grupos conservadores, sin cambiar etiquetas, máscaras ni particiones externas.

## Diseño y reproducción

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

## Resultados

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

## Análisis de errores

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

## Modelos y archivos guardados

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

## Continuación del proyecto

Esta fase deja entrenados y comparados los dos modelos tradicionales, con un
experimento inicial y otro de ajuste documentados. Se puede continuar con la
inferencia desde fotografías y el modelo de aprendizaje profundo exigido por el
proyecto, conservando estos modelos como referencia. Las próximas capturas deben
priorizar manzanas/limones intermedios, frutos malos y bananos buenos de campañas
distintas. La prueba final debe reservar capturas independientes; no es un requisito
para continuar ahora con la implementación.
