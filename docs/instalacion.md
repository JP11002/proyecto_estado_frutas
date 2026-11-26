# Ejecutar el proyecto en otro computador

El [registro de verificación](verificacion_proyecto.md) describe la prueba
real en una copia renombrada y un entorno instalado desde cero.

El repositorio incluye una sola base revisada de 291 imágenes y sus
43 características. Para hacer EDA, entrenar SVM/Random Forest o entrenar la CNN no
se necesita la carpeta `tmp` de otro equipo, descargar las fotos originales ni
recibir modelos entrenados. Es necesario incluir los nuevos archivos de
`data/base/` al guardar los cambios en Git.

## Windows

Instalar Python 3.11 y abrir una terminal en la raíz del proyecto. Los comandos
usan el ejecutable del entorno directamente y no requieren activar scripts de
PowerShell:

Si se clona por primera vez en Windows, permitir las rutas largas del historial
de anotaciones al clonar: `git -c core.longpaths=true clone URL_DEL_REPOSITORIO`.
Dentro del repositorio, `git config core.longpaths true` conserva esa opción para
las siguientes operaciones de Git en ese equipo.
La base única de entrenamiento usa rutas cortas para sus imágenes y máscaras.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/comprobar_proyecto.py
.\.venv\Scripts\python.exe scripts/ejecutar_cuaderno.py notebooks/06_eda_y_modelos_tradicionales.ipynb
```

Para abrir Jupyter con este entorno:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

## Linux y macOS

Con Python 3.11 o posterior disponible, desde la raíz del proyecto:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/comprobar_proyecto.py
.venv/bin/python scripts/ejecutar_cuaderno.py notebooks/06_eda_y_modelos_tradicionales.ipynb
```

Se recomienda Python 3.11 para reproducir el entorno probado. Las rutas se
resuelven respecto al proyecto, incluso cuando su carpeta cambia de nombre o
contiene espacios. No hace falta editar nombres de usuario o letras de disco.

## Flujo actual

1. `06_eda_y_modelos_tradicionales.ipynb`: EDA y entrenamiento inicial.
2. `07_ajuste_modelos_tradicionales.ipynb`: ajuste y comparación con la referencia.
   Si faltan las salidas del cuaderno 06, genera la referencia inicial necesaria.
3. `08_comparacion_cnn.ipynb`: comparación unificada con la CNN; requiere
   las dependencias opcionales descritas abajo y utiliza la misma base.

El ejecutor `scripts/ejecutar_cuaderno.py` crea un kernel temporal con el Python
usado para invocarlo. Así no depende de un kernel guardado en otro equipo. También
se pueden abrir los cuadernos desde Jupyter y seleccionar el entorno instalado.
Los modelos y las tablas generadas vuelven a aparecer en `experiments/` al ejecutar.

Desde una terminal también se puede ejecutar la comparación de los modelos
tradicionales, sin instalar PyTorch. Se reutilizan las predicciones verificadas
de 06/07; si faltan para la base actual, se generan antes de comparar:

```text
python -m src.training.entrenar_modelos --proyecto . --solo-ml
```

En estos ejemplos, `python` debe ser el ejecutable del entorno creado arriba.

## CNN opcional

La instalación base no obliga a descargar PyTorch. Para incluir la CNN:

```text
python -m pip install -r requirements-dl.txt
python scripts/comprobar_proyecto.py --cnn
python -m src.training.entrenar_modelos --proyecto . --epocas-cnn 120
```

La CNN funciona en CPU. En Windows/Linux, tras instalar `requirements.txt`, se
pueden instalar explícitamente las ruedas de CPU con:

```text
python -m pip install "torch>=2.5,<3" "torchvision>=0.20,<1" --index-url https://download.pytorch.org/whl/cpu
```

PyTorch publica las opciones por sistema en su [instalador oficial](https://pytorch.org/get-started/locally/).
La CNN actual usa MobileNetV3 Small preentrenada con extractor congelado y
entrena un nuevo clasificador. La primera ejecución descarga unos 10 MB de pesos
oficiales en `data/external/torch/checkpoints/`; después reutiliza ese cache local.
Se necesita conexión para esa primera descarga, además de instalar dependencias.
Los modelos finales guardan todos los pesos y se recargan sin descargar ImageNet.
La búsqueda interna utiliza características visuales fijas para reducir el cálculo
en CPU. `--cnn` solo comprueba arquitectura e integridad de datos, sin descargar
pesos ni ejecutar el entrenamiento real.

## Reproducir las versiones verificadas

Los rangos de `requirements.txt` permiten instalar dependencias compatibles. Para
usar las versiones numéricas verificadas de la entrega y comparar cifras con mayor fidelidad:

```text
python -m pip install -r requirements.txt -c constraints-reproducible.txt
```

La base conserva las características y los píxeles del experimento
original. La unificación de rutas cambia la huella del archivo CSV; las referencias
se regeneran para esa huella en el nuevo equipo. Cambios de biblioteca o plataforma
pueden modificar resultados, y no se debe reutilizar un archivo joblib con versiones
incompatibles: es preferible volver a entrenar.

## Cuadernos de revisión y fotos originales

Los cuadernos 01–05 documentan la adquisición, revisión y preparación. Los primeros
incluyen la versión histórica con naranja. Para revisar otra vez las fotos de alta
resolución se necesitan los originales; no se reconstruyen a partir de recortes.
El cuaderno `01_revision_datos_y_etiquetas.ipynb` contiene la descarga de Kaggle.
Para recuperar exactamente los archivos de Drive del listado versionado:

```text
python -m src.data.importar_drive --proyecto . --usar-listado
```

Esta descarga necesita conexión y acceso a las carpetas compartidas. Después se
sigue el flujo de preparación/revisión de esos cuadernos. Es independiente del
arranque de entrenamiento: los cuadernos 06/07 y la comparación unificada funcionan
con la base única sin volver a ejecutar los cuadernos históricos.

## Comprobaciones

```text
python scripts/comprobar_proyecto.py
python -m unittest discover -s tests -v
```

La primera orden verifica dependencias, integridad de la base y de las imágenes y
máscaras. Las pruebas comprueban rutas Windows/POSIX, la carga desde una carpeta
renombrada y los flujos de validación/entrenamiento. Las pruebas de CNN se omiten
si no se ha instalado el entorno opcional.

## Interfaz y entrega

Con los modelos generados, iniciar `python -m src.main`. Se puede cargar una foto,
recibir imágenes de cámara o usar ejemplos identificados como entrenamiento.
Las medidas son relativas al encuadre; no expresan tamaño físico. La lista de
requisitos, el informe IEEE y el video se gestionan desde [entrega](entrega.md).
Para regenerar el informe se usan las dependencias opcionales de
`requirements-informe.txt` y `scripts/generar_informe.py`; la exportación de PDF
con `scripts/exportar_pdf.ps1` requiere Word en Windows. El uso de la aplicación
no necesita esas dependencias de documentos.
