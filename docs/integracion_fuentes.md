# Integración de fotografías

El 24 de noviembre de 2026 el equipo confirmó que las frutas del proyecto son
**manzana roja, banano y limón**. Limón sustituye a naranja; los datos y cuadernos de
la preparación inicial se conservan como antecedentes, sin cambiar las etiquetas de sus fotos.
La configuración vigente está en `config/alcance_actual.json`.

## Procedencia confirmada por el equipo

- Fuente pública: `12Wcfr_mRLxJ98EpHR9WwbMd7jp49kZxK`. Se importan únicamente las
  subcarpetas Apple, Banana y Lime. No se incorporan Orange ni las otras frutas.
- Fotografías propias: `1AyydHBltsfnj9YC4-F9xi9AeogtiK6Fx`. Se importan Manzanas,
  Bananos y Limones, conservando la clasificación declarada en sus subcarpetas.

El listado completo obtenido mediante la vista de carpeta compartida contiene
318 imágenes del alcance vigente: 267 públicas (120 manzanas, 77 bananos y 70
limones) y 51 propias (25 manzanas, 12 bananos y 14 limones). El listado inicial de
la interfaz mostraba únicamente 50 imágenes por carpeta pública; no se utilizó ese
recorte como inventario completo. El inventario corresponde al momento de consulta;
si se agregan archivos a Drive, debe actualizarse la importación explícitamente.

## Etiquetas declaradas de las fotografías propias

| Fruta | Buena | Intermedia | Mala |
|---|---:|---:|---:|
| Manzana | 18 | 7 | 0 |
| Banano | 0 | 0 | 12 |
| Limón | 9 | 5 | 0 |

Una carpeta vacía se registra como tal. No se crean ejemplos sintéticos para llenar
una categoría ni se aprueba automáticamente la calidad por el nombre de la carpeta.
Las fuentes públicas nuevas no tienen etiquetas de calidad declaradas. Las
decisiones visuales deben distinguirse de las etiquetas de origen.

## Conservación e integridad

Los originales se conservan sin conversión en
`data/raw/drive_20261124/{publica,propia}/{fruta}/`. El prefijo del nombre contiene
el identificador de Drive para evitar colisiones entre nombres iguales. Se guardan
el nombre y la ruta originales, el enlace, la fuente declarada y SHA-256. Los HEIC
se leen mediante `pillow-heif`, con orientación EXIF, sin sustituir los originales.

El registro reproducible de la importación queda en
`data/annotations/drive_20261124/`: listado de carpetas y archivos, resultados de
descarga e inventario de integridad y lectura. Los duplicados exactos se buscan por
bytes y por píxeles decodificados; esto no demuestra que dos vistas diferentes sean
frutos independientes. Las miniaturas y propuestas calculadas se excluyen de Git.

Desde la carpeta del proyecto:

```python
# Enumerar y descargar los originales autorizados, conservando los ya presentes.
# Ejecutar en terminal: python -m src.data.importar_drive --proyecto .

# Inventario y galerías de revisión (requiere las dependencias del proyecto).
# Ejecutar en terminal: python -m src.data.integrar_fuentes --proyecto . --galerias
```

La separación entre entrenamiento y prueba debe mantener juntas las vistas de un
mismo fruto, incluso cuando aparezca en diferentes días o estados. Los grupos
físicos requieren confirmación del equipo. Las fotografías utilizadas para ajustar
el procedimiento de segmentación forman parte del desarrollo y no se presentan
posteriormente como una prueba independiente.

## Resultado de la revisión ampliada

La base anterior de 69 ejemplos se amplió a **291**: **270 fotografías de Drive**
(245 de la fuente pública y 25 propias) más los 21 ejemplares compatibles de
manzana y banano de la versión 1. Las naranjas continúan fuera del alcance vigente.
La versión previa de las anotaciones se conserva en
`data/annotations/drive_20261124/historial/revision_antes_ampliacion_20261124.json`.

Las **318 fotografías nuevas tienen una disposición explícita**. Se comprobaron
contorno y detalle de 292 fotografías con un fruto aislable: 270 máscaras aceptadas
y 22 rechazadas por fondo/reflejos/otros objetos o por fruto incompleto en el borde.
Se corrigieron contornos de manzanas propias para retirar reflejos de la mesa.
Las 26 fotos con varios frutos se excluyen como ejemplares completos y se conservan
para una posible segmentación individual posterior. No quedan originales sin
decisión dentro de este lote. Una exclusión no se presenta como ejemplo confirmado.

## Resolución de los cinco casos dudosos

Se volvieron a examinar recortes ampliados procedentes de los originales, sin
basarse en la miniatura de 224 píxeles. Se asignó **mala calidad visual** a los
cinco casos por deterioro externo marcado, preservando la etiqueta declarada
intermedia y la decisión anterior en el historial. No se aplicó un umbral de
oscuridad ni se asignaron clases para completar la cobertura.

| Código | Archivo original | Resolución | Evidencia externa |
|---|---|---|---|
| eq_limon_011 | IMG_5752.HEIC | Mala | Placas extensas de superficie alterada, textura dañada y grietas. |
| eq_limon_013 | IMG_5754.HEIC | Mala | Lesiones en distintas zonas y grietas con daño del extremo. |
| eq_limon_014 | IMG_5755.HEIC | Mala | Placas extensas y grietas; aspecto parecido a la vista 011. |
| eq_manzana_019 | IMG_6120.HEIC | Mala | Arrugas generalizadas, deformación y pérdida aparente de volumen. |
| eq_manzana_020 | IMG_6121.HEIC | Mala | Arrugamiento generalizado y superficie notablemente deteriorada. |

Son cinco fotografías, no cinco frutos físicos cuya independencia se haya
confirmado. Permanecen agrupadas conservadoramente con las otras vistas de su
fuente y fruta. También se clasificó como mala la manzana 021 al comprobar su
deterioro visible, aplicando el mismo criterio.

La resolución está atribuida al asistente, como las otras anotaciones de esta
revisión. Es una decisión operativa de un solo anotador; no se ha medido acuerdo
entre revisores. Describe apariencia externa y no confirma inocuidad, firmeza,
estado interno, patógeno ni vida útil. Los contornos son aproximados, sin referencia
independiente píxel a píxel.

## Cobertura de la base preparada

| Fruta | Buena | Intermedia | Mala | Total |
|---|---:|---:|---:|---:|
| Manzana roja | 121 | 3 | 7 | 131 |
| Banano | 8 | 70 | 5 | 83 |
| Limón | 72 | 2 | 3 | 77 |
| Total | 201 | 75 | 15 | 291 |

Las nueve combinaciones están presentes, pero las frecuencias están desequilibradas.
La revisión inicial utilizaba 15 grupos conservadores de captura. La anotación de
identidades propias los refina a 22 grupos en la base vigente; ver
`agrupacion_frutos_propios.md`. Esos grupos no son un censo verificado de frutos
independientes. En particular, las clases intermedia y mala del limón pertenecen
a una sola campaña propia. El aumento de vistas no resuelve esa dependencia.

## Archivos vigentes y reproducción

- `notebooks/04_integracion_fuentes_y_base.ipynb`: resultados actualizados,
  revisor interactivo y comprobación técnica de ajuste.
- `data/base/datos.csv`: 291 registros y 43 características
  de color, textura y forma dentro de la máscara.
- `data/annotations/drive_20261124/etiquetas_y_mascaras.csv`: inventario de las 318
  fotografías, etiquetas originales, revisiones y motivos de exclusión.
- `data/annotations/drive_20261124/cinco_casos_resueltos.csv`: decisiones de los cinco
  casos y sus huellas de imagen y máscara.
- `data/annotations/drive_20261124/criterios_revision_ampliada.json`: aclaración de
  los criterios observables y sus límites.
- `data/annotations/drive_20261124/revision_visual_v2.json`: anotaciones y contornos,
  con historial de cambios de decisión. Las máscaras aceptadas y sus recetas quedan
  en `mascaras_revisadas/` dentro de esta misma carpeta.

Con originales e inventario presentes, `python -m src.data.preparar_drive --proyecto .`
reconstruye propuestas a partir de las recetas; `python -m src.data.exportar_base
--proyecto .` verifica revisiones y archivos y exporta la base. Las versiones de
preparación se conservan separadas; un cambio de receta o archivo invalida la
aprobación anterior. Las decisiones de otros revisores se respetan durante la
actualización asistida.

Todo el lote ya se utilizó en el desarrollo del procesamiento y se registra como
explorado. El cuaderno 04 conserva la comprobación técnica de ajuste. El nuevo
cuaderno `06_eda_y_modelos_tradicionales.ipynb` realiza el EDA y compara SVM y
Random Forest mediante tres pliegues agrupados sobre la base actual. Las dudas de
identidad permanecen unidas y no bloquean este primer experimento. Ver
[resultados y reproducción](eda_y_modelos_tradicionales.md).

El requisito de cinco grupos por combinación pertenece al plan anterior para
separar entrenamiento, validación y prueba; no es un requisito universal para
empezar el EDA ni entrenar. La marca `lista_para_evaluacion_independiente=False`
del exportador describe ese plan de prueba final. La comparación agrupada actual
es exploratoria: conserva el desbalance y documenta las combinaciones sin cobertura.
Una prueba final requiere capturas independientes reservadas antes de ajustar el
procesamiento.

El tamaño permanece relativo a la fotografía y no se transforma en centímetros.
