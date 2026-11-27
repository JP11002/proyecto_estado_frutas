# Clasificación de calidad de frutas mediante visión por computadora

**Proyecto final de Algoritmos y Programación III**
Universidad Icesi | Cali, Colombia | Semestre 2026-2

## Descripción del proyecto

Este proyecto desarrolla un sistema de clasificación automática de la calidad de frutas y
verduras mediante técnicas de visión por computadora y aprendizaje automático.

A partir de una imagen, el sistema identifica el estado de un producto en tres categorías:

- **Buena:** producto en buen estado, sin señales evidentes de deterioro.
- **Intermedia:** producto con defectos localizados (menos de un tercio de la superficie).
- **Mala:** producto con deterioro marcado o extendido.

Además, estima el tamaño relativo del producto a partir de su máscara (diámetro equivalente
normalizado por el encuadre; no expresa centímetros).

## Estado del proyecto

**Prototipo completo y evaluado.**

Se implementaron la preparación y revisión de datos, la extracción de características, los
modelos de clasificación, la evaluación con una prueba independiente por grupos y una
interfaz gráfica con foto y cámara. El informe IEEE está en `docs/informe_final.pdf` (fuente:
`report.md`).

## Integrantes

| Nombre | GitHub |
|---|---|
| Juan Pablo Serrano |  [@JP11002](https://github.com/JP11002) |
| Felipe Calderon | @usuario |
| Juan Andres Castaño | @usuario |

## Objetivo del proyecto

Desarrollar una herramienta que evalúe automáticamente la calidad visual de frutas a partir
de fotografías, utilizando procesamiento de imágenes y modelos de aprendizaje automático.

La solución busca apoyar la identificación de productos según su estado de conservación y
reducir la evaluación subjetiva de la calidad. Es una herramienta de **apoyo**: evalúa la
apariencia externa, no la seguridad alimentaria, y requiere revisión humana.

## Métodos utilizados

- Procesamiento y análisis exploratorio de imágenes.
- Segmentación del fruto (GrabCut) y extracción de 43 características de color, textura y forma.
- Aprendizaje automático: SVM, Random Forest, regresión logística y Gradient Boosting.
- Aprendizaje profundo: MobileNetV3 preentrenada y congelada con capas de clasificación propias.
- Clasificación supervisada con pesos por clase y por combinación fruta-calidad.
- Validación cruzada agrupada y anidada, y prueba independiente por grupos.
- Optimización de hiperparámetros con búsqueda en rejilla.
- Visualización e interpretación de resultados (métricas por clase, fruta y fuente).

## Tecnologías utilizadas

- **Python:** lenguaje principal de desarrollo.
- **Jupyter Notebook:** experimentación y análisis.
- **Pandas y NumPy:** manipulación de datos.
- **OpenCV y Pillow:** procesamiento de imágenes y cámara.
- **Scikit-learn:** modelos tradicionales y evaluación.
- **PyTorch / torchvision:** red neuronal convolucional (funciona en CPU).
- **Matplotlib:** visualización de datos y resultados.
- **Tkinter:** interfaz gráfica.

## Descripción del sistema

El sistema utiliza imágenes de frutas para identificar características visuales relacionadas
con su estado de conservación y clasificarlas en tres categorías de calidad.

El conjunto de datos (332 imágenes revisadas, 31 grupos) se construyó a partir de:

- El dataset público *Fruit Quality Classification* de Kaggle (copia de FruitNet).
- Fotografías tomadas por los integrantes del equipo.
- Colecciones compartidas del curso en Google Drive.

El proyecto contempla tres productos:

- Manzana roja.
- Banano.
- Limón.

Se comparan tres familias de modelos:

- **Support Vector Machine (SVM):** modelo tradicional de clasificación.
- **Random Forest:** modelo basado en conjuntos de árboles de decisión.
- **Red neuronal convolucional (CNN):** MobileNetV3 congelada con un clasificador propio.

El modelo final combina las características manuales con las de la CNN. Los resultados se
analizan con precisión, recall, F1 por clase, F1 macro y ponderado, y matrices de confusión.

Uno de los principales retos es el desbalance entre categorías (63 % de frutas buenas) y la
asociación entre fruta y calidad en los datos: con pesos solo por clase, el modelo aprendía a
reconocer la fruta en lugar del defecto. También influye la variabilidad de iluminación, fondo,
perspectiva y resolución entre fuentes.

## Resultados

Evaluación en una prueba independiente (43 imágenes de 11 grupos que no participaron en
ninguna decisión) y validación anidada sobre las 332 imágenes:

| Modelo | F1 macro prueba (IC95 %) | F1 macro anidada (3 semillas) |
|---|---:|---:|
| Referencia mayoritaria | 0,197 | 0,258 |
| Random Forest | 0,417 (0,25–0,52) | 0,396 ± 0,034 |
| SVM ajustado | 0,324 (0,18–0,45) | 0,330 ± 0,053 |
| CNN (MobileNetV3 + cabezal propio, con costes) | 0,397 (0,18–0,54) | 0,575 ± 0,097 |
| **Modelo final (fusión + pesos fruta-calidad + costes)** | **0,395 (0,19–0,58)** | **0,460 ± 0,065** |

Con pocos grupos independientes, las diferencias entre modelos no son significativas; todos
superan a la referencia. El análisis completo está en `docs/ajuste_y_cnn.md` y en el informe.

## Estructura del repositorio

```
proyecto_estado_frutas/
│
├── config/                  # Alcance del proyecto y datos de entrega
│
├── data/
│   ├── raw/                 # Originales descargados (no se versionan)
│   ├── external/            # Kaggle y pesos de la CNN (no se versionan)
│   ├── processed/           # Preparaciones intermedias (no se versionan)
│   ├── annotations/         # Etiquetas, decisiones y máscaras revisadas
│   └── base/                # Base única: datos.csv, imágenes, máscaras y partición
│
├── docs/
│   ├── etica.md
│   ├── ajuste_y_cnn.md
│   ├── eda_y_modelos_tradicionales.md
│   ├── revision_base_v1.md
│   ├── analisis_de_impactos.md
│   ├── DiagramaCrispDm.drawio.svg
│   ├── figuras/             # Figuras vectoriales del informe
│   └── informe_final.pdf
│
├── experiments/
│   ├── checkpoints/         # Modelos entrenados (se regeneran con los cuadernos)
│   └── results/             # Métricas, predicciones y gráficas
│
├── notebooks/               # 01–09: datos, EDA, modelos y evaluación
│
├── scripts/                 # Ejecución de cuadernos, figuras e informe
│
├── sources/                 # Material de referencia del curso
│
├── src/
│   ├── data/                # Carga, preparación e integración de la base
│   ├── inference/           # Motor de predicción y medida de tamaño
│   ├── models/              # Modelos y experimentos
│   ├── training/            # Entrenamiento y comparación
│   ├── utils/               # Funciones auxiliares
│   ├── interfaz.py          # Interfaz gráfica (foto y cámara)
│   └── main.py              # Punto de entrada
│
├── tests/                   # Pruebas unitarias
├── report.md                # Fuente del informe IEEE
├── requirements.txt
└── README.md
```

## Instalación

Python 3.11 o posterior. En Windows, desde la raíz del proyecto:

```text
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dl.txt
.venv\Scripts\python.exe scripts\comprobar_proyecto.py
```

En Linux o macOS se usa `.venv/bin/python` en lugar de `.venv\Scripts\python.exe`. La primera
ejecución de la CNN descarga unos 10 MB de pesos de ImageNet en `data/external/torch/`.

## Ejecución

**Interfaz gráfica (foto o cámara):**

```text
.venv\Scripts\python.exe -m src.main
```

1. Elegir el modelo en la lista **Modelo**.
2. **Cargar foto** y luego **Analizar**, o **Iniciar cámara**: la cámara analiza un fotograma
   cada 1,5 segundos y **Detener cámara** la apaga. **Guardar resultado** exporta un JSON.
3. Los botones **Ejemplo buena / intermedia / mala** muestran imágenes de la base (son de
   entrenamiento, no una prueba).

Para la cámara se recomienda el modelo **CNN**, que no depende de segmentar el fruto en cada
fotograma, y una fruta sola frente a un fondo liso. Se usa la cámara predeterminada del
equipo; en Windows debe estar activado *Configuración → Privacidad → Cámara → Permitir que
las aplicaciones de escritorio accedan a la cámara*. La aplicación no guarda fotogramas.

**Una imagen desde la terminal:**

```text
.venv\Scripts\python.exe -m src.main --imagen ruta\a\foto.jpg --modelo CNN
```

## Documentación

- [Revisión e integración de la base](docs/revision_base_v1.md)
- [EDA y modelos tradicionales](docs/eda_y_modelos_tradicionales.md)
- [Ajuste de modelos y CNN](docs/ajuste_y_cnn.md)
- [Ética](docs/etica.md) y [análisis de impactos](docs/analisis_de_impactos.md)
- [Diagrama CRISP-DM](docs/DiagramaCrispDm.drawio.svg)
