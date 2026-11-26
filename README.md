# Clasificación de calidad de frutas y verduras mediante visión por computadora

**Proyecto final de Algoritmos y Programación III**  
Universidad Icesi · Cali, Colombia · Semestre 2026-2

**Integrantes:** Juan Pablo Serrano, Juan Castaño y Felipe Calderon.

Proyecto de clasificación de calidad externa de **manzana roja, banano y limón**,
con las categorías **buena, intermedia y mala**. Se comparan SVM y Random Forest;
también está implementado el flujo de comparación con una CNN.

## Descripción y objetivos

El sistema analiza imágenes para apoyar la inspección de calidad externa y
estimar tamaño relativo mediante características visuales. El alcance vigente
comprende manzana roja, banano y limón. Las categorías son buena, intermedia y
mala, según conservación de superficie y extensión de defectos visibles.

Se comparan aprendizaje automático tradicional y redes convolucionales, se
analizan errores y se documentan impactos y decisiones éticas. El prototipo está
implementado y evaluado; el cierre documental requiere añadir el video del equipo.


## Entrega y demostración

El [informe IEEE](docs/informe_final.pdf) resume el proyecto en cuatro páginas.
El enlace del video debe completarse antes de entregar; los integrantes y enlaces
se configuran en `config/entrega.json`. Ver la [lista de requisitos](docs/entrega.md)
y el [guion para grabar el video](docs/guion_video.md).

Para abrir la aplicación, desde el entorno con las dependencias instaladas:

```text
python -m src.main
```

Permite cargar una foto, usar cámara o mostrar ejemplos de la base, elegir CNN,
Random Forest o SVM y consultar calidad y tamaño relativo. Los ejemplos se marcan
como entrenamiento. La cámara requiere dispositivo y permisos del sistema.
La aplicación no guarda automáticamente fotogramas. El tamaño se mide sobre la
máscara del encuadre original y no representa centímetros.


## Una sola base

Todos los cuadernos y entrenamientos usan **`data/base/datos.csv`** a través de
`src/data/base.py`. La misma carpeta contiene `estado.json`, las imágenes y las
máscaras revisadas. Se incluye completa en Git y utiliza rutas relativas.
La misma carpeta se utiliza directamente en todos los equipos.

La base contiene **291 imágenes**, **43 características** y **22 grupos** de
evaluación. Hay 201 ejemplos buenos, 75 intermedios y 15 malos. Las fuentes y
decisiones se conservan en [integración de datos](docs/integracion_fuentes.md).

## Ejecución

Seguir la [guía de instalación](docs/instalacion.md). Desde el entorno instalado:

```text
python scripts/comprobar_proyecto.py
python scripts/ejecutar_cuaderno.py notebooks/06_eda_y_modelos_tradicionales.ipynb
python scripts/ejecutar_cuaderno.py notebooks/07_ajuste_modelos_tradicionales.ipynb
```

`requirements.txt` instala EDA, modelos tradicionales y Jupyter. La CNN utiliza
las dependencias opcionales de `requirements-dl.txt` y puede trabajar en CPU.

## Flujo de trabajo

- Los cuadernos 01–02 documentan la adquisición y preparación de imágenes.
- [03 — Comprobación inicial](notebooks/03_base_revisada_y_modelos_iniciales.ipynb)
  comprueba el ajuste sobre la base común.
- [04 — Integración y revisión](notebooks/04_integracion_fuentes_y_base.ipynb) y
  [05 — Identidades de frutos](notebooks/05_agrupacion_frutos_propios.ipynb)
  permiten revisar y actualizar esa misma base. Necesitan los originales para
  trabajar con fotografías de alta resolución.
- [06 — EDA y entrenamiento](notebooks/06_eda_y_modelos_tradicionales.ipynb)
  realiza el análisis y la comparación inicial.
- [07 — Ajuste tradicional](notebooks/07_ajuste_modelos_tradicionales.ipynb)
  ajusta parámetros con validación interna por grupos.
- [08 — Comparación con CNN](notebooks/08_comparacion_cnn.ipynb)
  ejecuta la comparación unificada con aprendizaje profundo.

Para incorporar revisiones aprobadas: `python scripts/actualizar_base.py`.
El proceso valida los archivos y actualiza `data/base/` directamente.

## Resultados de desarrollo

El experimento inicial obtuvo F1 macro de **0,429 en Random Forest** y **0,299 en
SVM**. El ajuste obtuvo **0,388 en Random Forest** y **0,373 en SVM**.
La CNN mejorada con MobileNetV3 Small congelada obtiene **0,472 de
F1 macro** y **70,4 % de exactitud**, frente a **0,237**
de la CNN simple anterior. Conserva la misma base y los mismos bloques por grupos.
La calidad intermedia sigue siendo débil (6/75 aciertos); son resultados
exploratorios de desarrollo y requieren una prueba final independiente.

El diagnóstico de intermedia se integra en el mismo cuaderno 08. Un cribado con
HistGradientBoosting obtuvo **0,487 de F1 macro** y **19/75 intermedias correctas**,
frente a 6/75 de CNN, con menor exactitud global. Las etiquetas y los modelos CNN
actuales se conservan; la comparación detallada incluye los costes por clase.

Ver [EDA](docs/eda_y_modelos_tradicionales.md),
[ajuste tradicional](docs/ajuste_modelos_tradicionales.md) y
[comparación con CNN](docs/entrenamiento_cnn.md).

## Archivos compartidos

Git conserva el código, cuadernos, documentación, anotaciones y `data/base/`.
Los originales descargados, preparaciones intermedias, entornos virtuales,
resultados y modelos generados se excluyen mediante `.gitignore`. Ver
[control de versiones](docs/control_versiones.md) y
[verificación del proyecto](docs/verificacion_proyecto.md).
