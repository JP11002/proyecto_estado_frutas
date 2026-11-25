# EDA y primer entrenamiento — 24 de noviembre de 2026

El cuaderno ejecutado `notebooks/06_eda_y_modelos_tradicionales.ipynb` explora la
base vigente y entrena SVM y Random Forest. No es necesario resolver más identidades
para reproducir este experimento: se usan los 22 grupos conservadores disponibles.

## Base y hallazgos del EDA

Se usan 291 imágenes de manzana roja, banano y limón, con 43 características de
color, textura y forma calculadas dentro de las máscaras revisadas. No hay valores
faltantes o no finitos en esas características, variables constantes, filas de
características idénticas ni huellas de original repetidas en esta base.

| Fruta | Buena | Intermedia | Mala | Total |
|---|---:|---:|---:|---:|
| Manzana roja | 121 | 3 | 7 | 131 |
| Banano | 8 | 70 | 5 | 83 |
| Limón | 72 | 2 | 3 | 77 |
| Total | 201 | 75 | 15 | 291 |

La clase buena representa el 69,1 %. Las 70 imágenes de banano intermedio son el
93,3 % de todas las intermedias; la calidad está fuertemente asociada con la fruta
y la procedencia. El número de fotografías por grupo también es desigual. Las
figuras muestran balance, fuentes, cobertura por grupos, ejemplos, distribuciones
y correlaciones. Los tamaños y formas siguen siendo relativos al encuadre.

## Diseño del experimento

- Objetivo: `calidad_revisada`, con las clases buena, intermedia y mala.
- Un clasificador conjunto para las tres frutas; entradas limitadas a las 43
  características. Fruta, fuente e identificadores se usan para analizar resultados.
- Validación: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)`.
  Los mismos pliegues para los tres modelos. Ningún grupo aparece a ambos lados
  de un pliegue; cada imagen tiene exactamente una predicción fuera del entrenamiento.
- SVM: `StandardScaler` dentro del pipeline, `SVC(C=1, kernel='rbf', gamma='scale',
  class_weight='balanced')`.
- Random Forest: 300 árboles, `min_samples_leaf=2`, `class_weight='balanced'`,
  `random_state=42`, `n_jobs=1`.
- Referencia: predecir siempre la clase mayoritaria del entrenamiento.
- Sin búsqueda de hiperparámetros ni selección de pliegues por sus puntuaciones.

Las tres calidades globales están presentes en entrenamiento y validación de cada
pliegue. De las 54 celdas pliegue/etapa/fruta/calidad, 12 no tienen imágenes; esto
limita las conclusiones específicas por fruta. El bloque tercero concentra 67
imágenes intermedias de las 75 existentes, porque se mantienen juntas las campañas.

## Resultados fuera del entrenamiento (OOF)

Se calculan sobre las 291 predicciones OOF reunidas, no como promedio de las tres
puntuaciones por pliegue. F1 macro y recall macro consideran las tres calidades por
igual. Al estar las tres presentes, recall macro equivale a exactitud balanceada.

| Modelo | Exactitud | F1 macro | Recall macro |
|---|---:|---:|---:|
| Referencia mayoritaria | 69,1 % | 0,272 | 0,333 |
| SVM | 39,5 % | 0,299 | 0,365 |
| Random Forest | 54,6 % | 0,429 | 0,479 |

Random Forest tiene el mayor F1 macro de los dos modelos en este experimento.
Ninguno supera la exactitud de la referencia, que acierta todas las buenas y
ninguna de las otras calidades. Por eso la exactitud global por sí sola resulta
insuficiente. Random Forest recupera 139/201 buenas, 11/75 intermedias y 9/15 malas;
SVM recupera 97/201, 11/75 y 7/15, respectivamente.

Al asignar peso total igual a cada grupo, F1 macro es 0,405 para Random Forest,
0,368 para SVM y 0,232 para la referencia. Estas métricas adicionales muestran
sensibilidad al tamaño de las campañas, sin convertir los grupos en un censo de
frutos independientes. Las tablas incluyen resultados por pliegue, fruta, fuente
y clase, y las predicciones individuales para revisar errores.

La debilidad principal es la clase intermedia, con recall del 14,7 % en ambos
modelos. Las puntuaciones varían entre pliegues y no permiten declarar un ganador
definitivo. La base ya había intervenido en la revisión de máscaras y etiquetas:
esta es una evaluación de desarrollo, no una prueba final independiente.

## Archivos y reproducción

Abrir el cuaderno 06 con las dependencias de `requirements.txt` y ejecutarlo de
arriba abajo. La base debe estar exportada por el cuaderno 04 y las imágenes
preparadas deben estar disponibles; no se descarga contenido durante el entrenamiento.
El módulo `src/models/modelos_tradicionales.py` concentra las operaciones reutilizables.

Salidas en `experiments/results/eda_modelos_v1/`:

- `diagnostico_eda.json`, conteos, cobertura y estadísticos de características.
- Cinco figuras PNG del EDA y las matrices de confusión.
- `asignacion_pliegues.csv`, `cobertura_pliegues.csv` y `predicciones_oof.csv`.
- `metricas_globales.csv`, `metricas_pliegues.csv`, `metricas_por_clase.csv`,
  `metricas_por_fruta.csv` y `metricas_por_fuente.csv`.
- `configuracion_entrenamiento.json`, con configuración, versiones y huella de la base.

Prototipos en `experiments/checkpoints/modelos_tradicionales_v1/`:
`svm.joblib`, `random_forest.joblib` y `metadata.json`. Cada archivo contiene el
modelo ajustado con toda la base de desarrollo, la lista ordenada de características,
las clases y la huella de la base. La carga se comprobó con predicciones coincidentes.
Se requieren las mismas dependencias/versiones registradas para reproducirlos;
cargar únicamente archivos joblib de procedencia confiable.

Ejemplo de inferencia sobre características ya extraídas:

```python
from pathlib import Path
import joblib

proyecto = Path.cwd()  # Ejecutar desde la raíz del proyecto.
paquete = joblib.load(proyecto / 'experiments/checkpoints/modelos_tradicionales_v1/random_forest.joblib')
prediccion = paquete['modelo'].predict(nuevas_caracteristicas[paquete['caracteristicas']])
```

El siguiente experimento puede ajustar hiperparámetros mediante validación interna
por grupos, conservando esta referencia. La ampliación debe priorizar manzanas y
limones intermedios, frutos malos y bananos buenos de diferentes capturas. La
necesidad de una prueba independiente se reserva para la evaluación final y no
bloquea el entrenamiento actual.
