# Archivos que se guardan en Git

El `.gitignore` de la raíz del proyecto separa los archivos necesarios para
reproducir el trabajo de los datos descargados y las salidas que se generan al
ejecutar los cuadernos.

Se conservan el código de `src/`, los cuadernos `.ipynb`, la configuración del
proyecto, `requirements.txt`, la documentación y el contenido de
`data/annotations/`. Las etiquetas, decisiones, identidades, recetas y máscaras
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
necesita obtener los datos originales y ejecutar los pasos de preparación antes
del cuaderno 06; los modelos se vuelven a generar al entrenar. Las fuentes y el
flujo de preparación están descritos en `integracion_fuentes_v2.md`, y el primer
experimento en `eda_y_modelos_tradicionales.md`.

Para verificar por qué se ignora un archivo, ejecutar desde la raíz del proyecto:

```powershell
git check-ignore -v --no-index experiments/checkpoints/modelos_tradicionales_v1/svm.joblib
```

Las reglas nuevas no dejan de seguir automáticamente archivos ya agregados a Git.
Si se incluyó una salida por accidente, se puede retirar únicamente del seguimiento
con `git rm --cached -- ruta/al/archivo`, conservando la copia local. No es necesario
vaciar todo el índice ni volver a agregar todo el proyecto.
