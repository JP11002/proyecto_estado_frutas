# Clasificación de calidad de frutas y verduras mediante visión por computadora

**Proyecto final de Algoritmos y Programación III**  
Universidad Icesi | Cali, Colombia | Semestre 2026-2

## Descripción del proyecto

Este proyecto busca desarrollar un sistema de clasificación automática de la calidad de frutas y verduras mediante técnicas de visión por computadora y aprendizaje automático.

A partir de imágenes, el sistema busca identificar el estado de un producto en tres categorías:

- **Buena:** producto en buen estado, sin señales evidentes de deterioro.
- **Intermedia:** producto que presenta algunas señales de deterioro.
- **Mala:** producto que presenta deterioro evidente.

Adicionalmente, se contempla la extracción de características visuales para estimar el tamaño de los productos.

## Estado del proyecto

**En desarrollo.**

Se han implementado procesos de preparación y análisis de datos, extracción de características y modelos iniciales de clasificación. Se continúa trabajando en la optimización, evaluación e integración de los componentes de la solución.

## Integrantes

| Nombre | GitHub |
|---|---|
| Juan Pablo Serrano | [@JP11002](https://github.com/JP11002) |
| Felipe Calderon | [@usuario](https://github.com/usuario) |
| Juan Andres Castaño | [@usuario](https://github.com/usuario) |

## Objetivo del proyecto

Desarrollar una herramienta que permita evaluar automáticamente la calidad visual de frutas y verduras a partir de fotografías, utilizando técnicas de procesamiento de imágenes y modelos de aprendizaje automático.

La solución busca facilitar la identificación de productos según su estado de conservación y explorar el uso de tecnologías de inteligencia artificial en procesos de selección y evaluación de alimentos. De esta manera, se busca aportar herramientas que puedan apoyar la toma de decisiones y reducir la evaluación subjetiva de la calidad de los productos.

## Métodos utilizados

- Procesamiento y análisis exploratorio de imágenes.
- Extracción de características visuales.
- Aprendizaje automático (*Machine Learning*).
- Aprendizaje profundo (*Deep Learning*).
- Clasificación supervisada.
- Validación cruzada y evaluación de modelos.
- Optimización de hiperparámetros.
- Visualización e interpretación de resultados.

## Tecnologías utilizadas

- **Python:** lenguaje principal de desarrollo.
- **Jupyter Notebook:** experimentación y análisis.
- **Pandas y NumPy:** manipulación de datos.
- **OpenCV:** procesamiento de imágenes.
- **Scikit-learn:** implementación y evaluación de modelos tradicionales.
- **TensorFlow / Keras:** desarrollo de redes neuronales.
- **Matplotlib y Seaborn:** visualización de datos y resultados.
- **Streamlit:** desarrollo de la interfaz gráfica.

## Descripción del proyecto

El sistema utiliza imágenes de frutas y verduras para identificar características visuales relacionadas con su estado de conservación y clasificarlas en tres categorías de calidad: buena, intermedia y mala.

El conjunto de datos se construye a partir de diferentes fuentes:

- Dataset público de clasificación de calidad de frutas disponible en Kaggle.
- Fotografías tomadas por los integrantes del equipo.
- Imágenes públicas complementarias.

Actualmente, el proyecto contempla tres tipos de productos:

- Manzana roja.
- Banano.
- Limón.

Para abordar el problema se consideran tres enfoques de modelado:

1. **Support Vector Machine (SVM):** modelo tradicional de clasificación.
2. **Random Forest:** modelo basado en conjuntos de árboles de decisión.
3. **Red neuronal convolucional (CNN):** modelo de aprendizaje profundo para el análisis directo de imágenes.

Los resultados se analizan mediante métricas como precisión, recall, F1-score y matrices de confusión, con el propósito de comparar el comportamiento de los modelos e identificar sus fortalezas y limitaciones.

Uno de los principales retos del proyecto es el desbalance entre las categorías de calidad, así como la variabilidad de las imágenes según la iluminación, el fondo, la perspectiva y las condiciones en las que fueron tomadas.

## Estructura del repositorio

```text
proyecto_estado_frutas/
│
├── config/
|
├── data/
│   ├── raw/
│   ├── external/
│   └── processed/
|   └── annotations/
│
├── docs/
│   ├── etica.md
|   ├── ajuste_y_cnn.md
|   ├── eda_y_modelos_tradicionales.md
|   ├── revision_base_v1.md
│   ├── analisis_de_impactos.md
│   └── DiagramaCrispDm.drawio.svg
│
├── experiments/
│   ├── checkpoints/
│   └── results/
│
├── notebooks/
|
├── sources/
|
│
├── src/
│   ├── models/
│   ├── training/
|   ├── utils/
│   └── main.py
│
├── requirements.txt
└── README.md
