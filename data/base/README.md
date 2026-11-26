# Base del proyecto

Esta es la única base que utilizan los cuadernos, SVM, Random Forest y la CNN.

- `datos.csv`: 291 registros, etiquetas, grupos, procedencia y 43 características.
- `estado.json`: esquema y huella del archivo de datos.
- `imagenes/`: los 291 recortes RGB preparados para entrenamiento.
- `mascaras/`: sus 291 máscaras revisadas.
- `resumen_clases.csv` y `cobertura.csv`: resúmenes de esta misma tabla.

Las rutas son relativas al proyecto. La carpeta completa se incluye en Git.
`src/data/base.py` contiene el lector compartido; no busca otras copias.
`data/raw/` conserva originales y `data/processed/` es trabajo intermedio de
preparación. Ninguna de esas carpetas sustituye la base de entrenamiento.

Después de revisar nuevas etiquetas y máscaras, ejecutar:

```text
python scripts/actualizar_base.py
python scripts/comprobar_proyecto.py
```

La actualización valida los originales y revisiones y escribe aquí mismo.
Para entrenar en otro equipo basta con esta base y las dependencias del proyecto.
