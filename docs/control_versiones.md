# Archivos que se guardan en Git

El `.gitignore` de la raíz del proyecto separa los archivos necesarios para
reproducir el trabajo de los datos descargados y las salidas que se generan al
ejecutar los cuadernos.

Se conservan el código de `src/`, los cuadernos `.ipynb`, la configuración del
proyecto, los archivos de dependencias, los scripts de instalación/ejecución, la
documentación y el contenido de `data/annotations/` y `data/base/`.
La base única incluye los 291 recortes necesarios para entrenar sin descargas.
Las etiquetas, decisiones, identidades, recetas y máscaras
revisadas son anotaciones del proyecto y deben conservarse. Los resultados
ejecutados que ya están dentro de los cuadernos también se mantienen.

Se excluyen:

- Datos originales, descargas y tablas procesadas: `data/raw/`, `data/external/`
  y `data/processed/`.
- Resultados generados y modelos: `experiments/results/` y
  `experiments/checkpoints/`, incluyendo futuras versiones de experimentos.
- Archivos de modelos guardados en otros lugares, como `.joblib`, `.pt` y `.onnx`.
- Entornos virtuales, cachés de Python/Jupyter, temporales, registros y archivos
  locales del sistema o editor.
- Credenciales locales, como `.env`, `kaggle.json` y secretos de Streamlit.
  Las plantillas `.env.example` y `.env.sample` sí se pueden versionar.

Ignorar un archivo no lo borra del computador. Una copia nueva del repositorio
puede comenzar directamente en el cuaderno 06 usando `data/base/`;
los modelos se vuelven a generar al entrenar. Los originales solo hacen falta
para repetir la adquisición y revisión de alta resolución. Las fuentes y el
flujo de preparación están descritos en `integracion_fuentes.md`, y el primer
experimento en `eda_y_modelos_tradicionales.md`.

La [guía de instalación](instalacion.md) contiene los pasos para Windows, Linux y
macOS. `.gitattributes` conserva los bytes de CSV e imágenes para que sus huellas
de integridad no cambien por la conversión de finales de línea al clonar.

Para verificar por qué se ignora un archivo, ejecutar desde la raíz del proyecto:

```powershell
git check-ignore -v --no-index experiments/checkpoints/modelos_tradicionales/svm.joblib
```

Las reglas nuevas no dejan de seguir automáticamente archivos ya agregados a Git.
Si se incluyó una salida por accidente, se puede retirar únicamente del seguimiento
con `git rm --cached -- ruta/al/archivo`, conservando la copia local. No es necesario
vaciar todo el índice ni volver a agregar todo el proyecto.
