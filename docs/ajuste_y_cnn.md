# Ajuste de hiperparámetros y comparación con CNN simple

Este documento describe el flujo implementado para cumplir con el entrenamiento, ajuste de hiperparámetros y comparación entre modelos tradicionales de Machine Learning (SVM y Random Forest) y Deep Learning (CNN simple desde cero en PyTorch).

## Diseño metodológico sin fuga de datos

- **Partición externa compartida:** `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)` en [`src/models/evaluacion.py`](../src/models/evaluacion.py). Ningún grupo (`grupo_particion`) aparece a la vez en entrenamiento y validación.
- **Ajuste interno agrupado:**
  - En [`src/models/modelos_tradicionales.py`](../src/models/modelos_tradicionales.py), `evaluar_tradicionales_anidados` ejecuta `GridSearchCV` con `StratifiedGroupKFold` dentro de cada pliegue de entrenamiento externo optimizando `f1_macro`.
  - **SVM:** `Pipeline(StandardScaler, SVC(probability=True, class_weight='balanced'))` explorando `kernel` (`rbf`, `linear`), `C` (`0.1, 1.0, 5.0, 10.0`) y `gamma` (`scale, 0.01, 0.1`).
  - **Random Forest:** explorando `n_estimators` (`200, 400`), `max_depth` (`None, 8`), `min_samples_leaf` (`1, 2, 4`) y `max_features` (`sqrt, 0.3`), con `class_weight='balanced'`.
- **CNN simple en PyTorch (`src/models/cnn_calidad.py`):**
  - Entrenada desde cero sobre las imágenes RGB `224x224` generadas por `preparar_cnn`.
  - Arquitectura `CNNCalidadFruta`: 3 bloques `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d`, `AdaptiveAvgPool2d((4, 4))` y cabezal denso (`1024 -> 128 -> 3`) con `Dropout`.
  - Compensación de desbalance mediante `CrossEntropyLoss(weight=...)` calculada únicamente con las frecuencias del conjunto de entrenamiento de cada pliegue.
  - Aumento de datos ligero (volteos, rotación de hasta 15° con relleno gris `128`, ajuste moderado de brillo y contraste) aplicado únicamente en entrenamiento.
  - Búsqueda agrupada interna sobre `lr` (`1e-3, 3e-4`), `dropout` (`0.2, 0.4`) y `weight_decay` (`1e-4, 5e-4`). En la evaluación externa, `val_idx` nunca participa en *early stopping* ni en selección de arquitectura.

## Reproducción

Con la base `data/processed/base_revisada_v2/base_svm_rf.csv` y las imágenes `entrada_cnn.png` disponibles:

```powershell
python -m src.training.entrenar_modelos --proyecto .
```

Para ejecutar únicamente el ajuste de los modelos tradicionales (sin PyTorch):

```powershell
python -m src.training.entrenar_modelos --proyecto . --solo-ml
```

También puede ejecutarse paso a paso desde el cuaderno [`notebooks/07_ajuste_hiperparametros_y_cnn.ipynb`](../notebooks/07_ajuste_hiperparametros_y_cnn.ipynb).

## Salidas generadas

- `experiments/results/comparacion_modelos_v2/`:
  - `metricas_globales.csv`, `metricas_pliegues.csv`, `metricas_por_clase.csv`, `metricas_por_fruta.csv`, `metricas_por_fuente.csv`.
  - `busqueda_hiperparametros_ml.csv`, `resumen_ajuste_ml.json`, `cnn_busqueda_interna.csv`, `cnn_busqueda_global.csv`, `cnn_historial_epocas.csv`.
  - `matrices_confusion.png`, `comparacion_modelos.png`, `curvas_entrenamiento_cnn.png`.
- `experiments/checkpoints/comparacion_v2/`:
  - `svm_ajustado.joblib`, `random_forest_ajustado.joblib`, `cnn_calidad.pt` y `metadata.json`.
