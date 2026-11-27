# Revisión e integración de la base

Este documento reúne cómo se construyó la base de datos: la revisión inicial con Kaggle
(sección 1), la integración de las colecciones de Drive y del lote del 26/11/2026
(sección 2) y la agrupación de fotografías propias por fruto para evitar fugas entre
entrenamiento y evaluación (sección 3).

## 1. Revisión inicial de etiquetas y máscaras (base v1)

> Antecedente: desde el 24/11/2026 el equipo sustituyó naranja por limón. La base
> vigente y la integración posterior se describen en la sección 2.

Se revisaron visualmente **194 fotografías** del conjunto público local (7.388
imágenes inventariadas). Se prepararon y comprobaron **102 regiones candidatas**:
43 máscaras aceptadas y 59 rechazadas. La base resultante contiene **43 ejemplares
de 43 fotografías**, organizados en 15 grupos conservadores de captura. No se ha
revisado individualmente todo el conjunto público.

| Fruta | Buena | Intermedia | Mala | Total |
|---|---:|---:|---:|---:|
| Manzana roja | 5 | 3 | 4 | 12 |
| Banano | 3 | 3 | 3 | 9 |
| Naranja | 11 | 5 | 6 | 22 |
| Total | 19 | 11 | 13 | 43 |

Las anotaciones están atribuidas a **Asistente - revision visual**. Constituyen una
revisión de un solo anotador, susceptible de corrección por el equipo; no equivalen
a aprobación humana ni se ha medido acuerdo entre anotadores.

### Criterio aplicado

- **Buena:** superficie visible íntegra, sin deterioro externo relevante.
- **Intermedia:** lesiones o defectos localizados, con una parte importante de la
  superficie visible conservada. No se deduce simplemente de una probabilidad entre
  buena y mala ni de la cantidad de píxeles oscuros.
- **Mala:** deterioro visible extendido o lesiones externas marcadas.
- Una vista ambigua, una manzana cuyo alcance rojo no pueda confirmarse o una
  fotografía sin un ejemplar aislable permanece pendiente o excluida.

Estos criterios describen **apariencia externa**; una fotografía no confirma
inocuidad, estado interno, firmeza ni vida útil. En banano, el color amarillo o verde
por sí solo no determina la calidad. El protocolo es `calidad_visual_v1`.

Se encontraron bananos dentro de `Apple_Good`, manzanas fuera del alcance rojo,
fotografías con varios frutos y máscaras que incluían manos, fondo u otros objetos.
Se conservaron las etiquetas originales del proveedor por separado. Cuatro
ejemplares aceptados proceden de regiones individuales de fotos con varios frutos;
tienen etiqueta propia e identificador vinculado al original, sin contar la foto
completa como otro ejemplar aceptado.

### Máscaras y características

Se comprobó el original, la superposición y el recorte de las regiones aceptadas.
Los contornos son aproximaciones revisadas para este prototipo; no constituyen
máscaras de referencia independientes, anotadas píxel a píxel. Las propuestas
automáticas rechazadas no entran en la base.

Cada ejemplar conserva la huella del original, la máscara, el recorte y la receta de
preparación. La exportación rechaza anotaciones desactualizadas o archivos cuya
huella cambió. Las máscaras aceptadas y sus recetas se conservan en
`data/annotations/mascaras_revisadas/`.

SVM y Random Forest reciben **43 características de color, textura y forma**,
calculadas sobre la región segmentada. Identificadores, nombres de archivo,
etiquetas y grupos no forman parte de X. Los indicadores de ocupación de la imagen
y diámetro equivalente quedan disponibles en la base de preparación para el estudio
de tamaño: no representan centímetros ni tamaño físico comparable sin controlar
distancia, cámara y escala.

### Archivos y reproducción

- `data/annotations/revision_asistente_v1.json`: decisiones de la revisión y
  contornos anotados, incluidos los rechazos.
- `data/annotations/revision_equipo.csv`: revisión de originales, con autoría explícita.
- `data/annotations/ejemplares_segmentados.csv`: cuatro regiones individuales aceptadas.
- `data/annotations/revision_mascaras.csv`: decisiones de máscaras y firmas.
- `notebooks/01_revision_datos_y_etiquetas.ipynb`: inventario y revisión de calidad.
- `notebooks/02_preparacion_imagenes.ipynb`: reconstrucción, revisión y exportación
  de máscaras y características.
- `notebooks/03_base_revisada_y_modelos_iniciales.ipynb`: validación de la base,
  ejemplos visuales y ajuste inicial de ambos modelos.
- La tabla de esta preparación inicial se utiliza como insumo de integración.
  La única entrada de entrenamiento vigente es `data/base/datos.csv`, con su
  estado en `data/base/estado.json`.

Con los originales e inventario disponibles, las anotaciones se pueden reconstruir
mediante `python -m src.data.revision_asistida --proyecto .`. La opción
`--guardar-revisiones` vuelve a escribir las decisiones del asistente registradas en
el manifiesto, preservando decisiones atribuidas a otros revisores. Para corregir
una decisión, editarla en los revisores de los cuadernos y ejecutar nuevamente la
exportación del 02 y el cuaderno 03. Las salidas calculadas están excluidas de Git;
las anotaciones y máscaras revisadas se conservan como evidencia reproducible.

### Alcance del arranque de modelos

La base permite comprobar el ajuste de SVM con estandarización dentro de un
Pipeline y de Random Forest. Ese ajuste usa el lote de desarrollo completo y **no
es una medición de desempeño**. No se publican métricas de entrenamiento como si
fueran resultados independientes ni se guardan estos ajustes como modelos finales.

Los grupos se definieron por campañas de captura de forma conservadora porque las
fotografías contienen vistas repetidas. Quince grupos no equivalen a quince frutos
físicos identificados con certeza. Algunas combinaciones de fruta y calidad tienen
solamente un grupo. El plan agrupado 60/20/20 exige al menos cinco grupos por
combinación como mínimo operativo, además de cobertura y separación de lo ya
explorado; este mínimo no garantiza suficiencia estadística. Actualmente no puede
construirse esa evaluación con el lote revisado. Las fotografías ya inspeccionadas
quedan registradas en el historial de exploración y se reservan para desarrollo.

Para una comparación válida hacen falta otros ejemplares y capturas, una segunda
revisión de casos difíciles y una prueba independiente que se reserve antes de
ajustar el procesamiento. Después podrán seleccionarse parámetros con validación
agrupada y compararse macro-F1, balanced accuracy, matriz de confusión y resultados
por fruta.

## 2. Integración de fuentes y lote del 26/11/2026

El 24 de noviembre de 2026 el equipo confirmó que las frutas del proyecto son
**manzana roja, banano y limón**. Limón sustituye a naranja; los datos y cuadernos de
la preparación inicial se conservan como antecedentes, sin cambiar las etiquetas de sus fotos.
La configuración vigente está en `config/alcance_actual.json`.

### Procedencia confirmada por el equipo

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

### Etiquetas declaradas de las fotografías propias

| Fruta | Buena | Intermedia | Mala |
|---|---:|---:|---:|
| Manzana | 18 | 7 | 0 |
| Banano | 0 | 0 | 12 |
| Limón | 9 | 5 | 0 |

Una carpeta vacía se registra como tal. No se crean ejemplos sintéticos para llenar
una categoría ni se aprueba automáticamente la calidad por el nombre de la carpeta.
Las fuentes públicas nuevas no tienen etiquetas de calidad declaradas. Las
decisiones visuales deben distinguirse de las etiquetas de origen.

### Conservación e integridad

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
## Enumerar y descargar los originales autorizados, conservando los ya presentes.
## Ejecutar en terminal: python -m src.data.importar_drive --proyecto .

## Inventario y galerías de revisión (requiere las dependencias del proyecto).
## Ejecutar en terminal: python -m src.data.integrar_fuentes --proyecto . --galerias
```

La separación entre entrenamiento y prueba debe mantener juntas las vistas de un
mismo fruto, incluso cuando aparezca en diferentes días o estados. Los grupos
físicos requieren confirmación del equipo. Las fotografías utilizadas para ajustar
el procedimiento de segmentación forman parte del desarrollo y no se presentan
posteriormente como una prueba independiente.

### Resultado de la revisión ampliada

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

### Resolución de los cinco casos dudosos

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

### Cobertura de la base preparada

| Fruta | Buena | Intermedia | Mala | Total |
|---|---:|---:|---:|---:|
| Manzana roja | 121 | 3 | 7 | 131 |
| Banano | 8 | 70 | 5 | 83 |
| Limón | 72 | 2 | 3 | 77 |
| Total | 201 | 75 | 15 | 291 |

Las nueve combinaciones están presentes, pero las frecuencias están desequilibradas.
La revisión inicial utilizaba 15 grupos conservadores de captura. La anotación de
identidades propias los refina a 22 grupos en la base vigente; ver
la sección 3. Esos grupos no son un censo verificado de frutos
independientes. En particular, las clases intermedia y mala del limón pertenecen
a una sola campaña propia. El aumento de vistas no resuelve esa dependencia.

### Archivos vigentes y reproducción

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

### Lote del 26 de noviembre de 2026 (carpetas propias, subcarpetas *Intermedios*)

Se volvieron a listar **solo las dos carpetas definidas en `src/data/importar_drive.py`**.
Desde el inventario del 24/11 aparecieron **2 101 archivos nuevos**, todos en
`Manzanas/Intermedio` (1 200), `Bananos/Intermedios` (500) y `Limones/Intermedios` (401).
Los originales se descargaron a `data/raw/drive_20261126/` (excluida de Git) y se
clasificaron por píxeles en `data/annotations/drive_20261126/inventario_nuevas.csv`:

| Fruta | Archivos | Duplicados exactos | Variantes aumentadas | Stock / un fruto | Originales |
|---|---:|---:|---:|---:|---:|
| Manzana | 1 200 | 516 | 570 | 18 ImageNet + 62 Fruits-360 | 34 |
| Banano | 500 | 0 | 460 | 0 | 40 |
| Limón | 401 | 25 | 351 | 0 | 25 |
| **Total** | 2 101 | 541 | 1 381 | 80 | **99** |

- **Variantes aumentadas** (`*_NNN`): rotaciones con esquinas negras, relleno gris o
  negro, reescalado a 640×480 y cambios de color de un mismo original. Incorporarlas
  repartiría copias de una foto entre entrenamiento y evaluación (fuga de datos).
  El aumento se aplica solo dentro de los pliegues de entrenamiento.
- **Stock**: `n07740461_*` son fotografías de ImageNet (manzanas buenas, una con marca
  de agua) y `r1_*_100` son 62 fotogramas de **una sola manzana** de Fruits-360. No
  corresponden a la etiqueta declarada y no se incorporan.
- **Resolución**: 48 originales tienen lado menor < 64 px; sus defectos y textura no son
  evaluables y se excluyen.

Los 51 originales restantes se revisaron visualmente con el protocolo `calidad_visual_v1`
(`revision_visual.json`): **41 incluidos** (8 buenos, 19 intermedios y 14 malos) y
10 excluidos por fruto incompleto o solapado (7), calidad ambigua (2) o manzana no roja (1).
La etiqueta declarada por la carpeta era *intermedia* en todos los casos; la revisión
la modificó en 22 de los 41 incluidos.

| Fruta | Buena | Intermedia | Mala | Total |
|---|---:|---:|---:|---:|
| Manzana roja | 2 | 3 | 0 | 5 |
| Banano | 2 | 7 | 9 | 18 |
| Limón | 4 | 9 | 5 | 18 |
| Total | 8 | 19 | 14 | 41 |

Las máscaras se reconstruyen desde recetas versionadas (GrabCut por rectángulo,
umbral HSV, o elipse/polígono manual refinado con GrabCut) y se guardan en
`data/annotations/drive_20261126/mascaras_revisadas/`. Los grupos son conservadores por
sesión o escena de captura (9 grupos nuevos, disjuntos de los 22 previos). La fuente en
la base es `propia_lote_20261126`, para poder analizar métricas por fuente.

Reproducción (con los originales en `data/raw/drive_20261126/`):

```text
python -m src.data.incorporar_lote --proyecto . --galeria
python scripts/exportar_medidas_tamano.py
```

El proceso es idempotente y verifica que las 291 filas previas no cambien, que sus
grupos se conserven y que los grupos nuevos no coincidan con los anteriores.

**Limitaciones del lote.** Son recortes y capturas de baja resolución con fondos (césped,
turquesa, lavanda, madera) distintos de la base previa: un modelo puede asociar el
fondo con la calidad, por lo que las métricas deben revisarse por fuente. La variedad
de algunos limones amarillos o anaranjados no está confirmada. El tamaño relativo
de un recorte depende del encuadre del recorte y no es comparable con el de una
fotografía completa. Las etiquetas proceden de un único revisor asistido por IA.

## 3. Agrupación de fotografías propias por fruto

Se compararon las 51 fotos propias con ampliaciones del original, marcas de la
superficie, forma, pedúnculo y secuencia de captura. Se anotaron 103 apariciones o
regiones, incluidas las fotos de varios frutos, con 18 identificadores de referencia
propuestos: 9 de manzana, 6 de limón y 3 de banano. Estos perfiles no se interpretan
como 18 frutos independientes verificados: algunas correspondencias siguen siendo
hipótesis. La certeza visual es cualitativa y no representa una probabilidad.

El usuario confirmó que las manzanas del **16 y del 23 de noviembre eran frutos
diferentes**. Esa confirmación permite separar los dos conjuntos. No se dedujo
independencia solamente de la fecha ni de su etiqueta de calidad.

### Correspondencias individuales utilizadas en la base

| Identificador propuesto | Fotografías individuales | Observación |
|---|---|---|
| EQ_MANZANA_01 | IMG_5732, IMG_5733 | Pedúnculo corto, campo claro y rayas del mismo costado. |
| EQ_MANZANA_02 | IMG_5737, IMG_5738 | Cavidad profunda y rayas; cambio de perspectiva. |
| EQ_MANZANA_03 | IMG_5739, IMG_5740 | Pedúnculo largo curvado con extremo claro. |
| EQ_MANZANA_04 | IMG_5741 | Pedúnculo fino y largo, referencia en foto de seis frutos. |
| EQ_MANZANA_05 | IMG_5742, IMG_5743, IMG_5744 | Pedúnculo grueso y marcas claras del costado. |
| EQ_MANZANA_06 | IMG_5745, IMG_5746, IMG_5747 | Pedúnculo corto y rayos amarillos alrededor de la cavidad. |
| EQ_MANZANA_07 | IMG_6120, IMG_6121 | El usuario confirmó que es la misma manzana en dos perspectivas. |
| EQ_MANZANA_08 | IMG_6122 | Cáliz y mancha cercana reconocibles en la foto de grupo. |
| EQ_LIMON_01 | IMG_5756, IMG_5758 | Mismas dos puntas y superficie; IMG_5757 también se asocia, pero está excluida por encuadre. |
| EQ_LIMON_05 | IMG_5751, IMG_5753 | Coinciden mancha en estrella y banda parda. |
| EQ_LIMON_06 | IMG_5752, IMG_5755; IMG_5754 probable | Coincidencia clara 5752–5755; correspondencia de 5754 incierta. |
| EQ_BANANO_01 | IMG_6132, IMG_6133 | Coinciden forma, parche claro del extremo y pequeña lesión circular. |

Los otros perfiles de referencia aparecen en fotografías de varios frutos. Se
registran en la tabla de ocurrencias con posición y, donde corresponde, centro
normalizado para mostrar la asociación. Una foto de varios frutos sigue excluida
como ejemplo completo; la identificación no aprueba una máscara ni añade ejemplos.

### Identidad y grupo de evaluación son datos distintos

El campo `id_fruto_propuesto` muestra la asociación visual. El campo
`grupo_evaluacion_seguro` mantiene juntos perfiles que podrían corresponder al
mismo fruto. Es posible tener varios identificadores propuestos dentro de un
solo grupo de evaluación.

- Las manzanas del 16/11 se agrupan por los seis perfiles individuales. Las fotos
  de seis frutos apoyan que se trata de ejemplares distintos presentes a la vez.
- El usuario confirmó que IMG_6120 e IMG_6121 son el perfil de manzana 07. Se
  retiró la alternativa 09. Los perfiles 07, 08 y 09 aparecen como frutos distintos
  en las fotos de grupo del 23/11; el 09 no tiene foto individual confirmada en la base.
- En limones se mantiene un bloque conservador común. La correspondencia entre
  algunas vistas de grupo y las caras sanas/dañadas aún no permite garantizar que
  todos los perfiles propuestos sean frutos físicos diferentes.
- IMG_6132 e IMG_6133 tienen el mismo identificador de banano y el mismo grupo.
- IMG_6138, con el plato y un objeto muy deteriorado/fragmentado, queda con identidad
  sin resolver. No se le asigna un nuevo fruto independiente ni se incorpora al entrenamiento.

Las **25 fotos propias utilizadas en la base** tienen identidad individual
propuesta. Representan 12 identificadores propuestos y **10 grupos conservadores**
de evaluación. La base completa conserva **291 ejemplos y 43 características**;
sus grupos pasan de 15 campañas amplias a **22 grupos conservadores**. Los grupos
de las fuentes públicas se mantienen. Estos 22 grupos no son un censo de frutos
independientes verificados.

### Archivos y revisión

- `data/annotations/identidad_propias_v1.json`: anotación principal, referencias,
  apariciones, alternativas y confirmación aportada por el usuario.
- `data/annotations/identidad_propias/fotos_propias_con_identidad.csv`: 51 fotos con
  sus identificadores, certeza, alternativas y grupo de evaluación.
- `data/annotations/identidad_propias/ocurrencias_por_fruto.csv`: 103 regiones o
  apariciones, con evidencia y ubicación.
- `data/annotations/identidad_propias/catalogo_frutos.csv`: perfiles de referencia.
- `data/annotations/identidad_propias/correspondencias_por_confirmar.csv`: asociaciones
  medias, alternativas o sin resolver, incluidas algunas fotos de grupo.
- `notebooks/05_agrupacion_frutos_propios.ipynb`: inspección visual, editor de identidades
  individuales y aplicación de los grupos a la base.

Los nombres y archivos originales permanecen intactos. Calidad, máscaras y sus
huellas se conservan; las identidades son una capa separada de anotación. Cada
identidad se valida contra la huella del original y las alternativas de una foto
individual deben pertenecer al mismo bloque seguro. Las correcciones desde el
cuaderno guardan autoría e historial. Una alternativa nueva puede unir grupos;
una corrección no separa automáticamente vínculos antiguos de incertidumbre.

Para regenerar las tablas y las imágenes de referencia, ejecutar desde la carpeta
del proyecto `python -m src.data.identidad_frutos --proyecto .`. Después ejecutar
el cuaderno 04 o la exportación del 05 para actualizar la base de modelos.

La comparación de SVM y Random Forest continúa pendiente: las clases escasas
conservan pocas unidades distintas y todo el lote ya se usó durante el desarrollo.
Las nuevas identidades permiten diseñar una partición más precisa, pero no crean
una prueba independiente ni resuelven por sí solas la falta de datos.
