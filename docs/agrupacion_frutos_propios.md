# Identificación de frutos en las fotografías propias

Se compararon las 51 fotos propias con ampliaciones del original, marcas de la
superficie, forma, pedúnculo y secuencia de captura. Se anotaron 103 apariciones o
regiones, incluidas las fotos de varios frutos, con 18 identificadores de referencia
propuestos: 9 de manzana, 6 de limón y 3 de banano. Estos perfiles no se interpretan
como 18 frutos independientes verificados: algunas correspondencias siguen siendo
hipótesis. La certeza visual es cualitativa y no representa una probabilidad.

El usuario confirmó que las manzanas del **16 y del 23 de noviembre eran frutos
diferentes**. Esa confirmación permite separar los dos conjuntos. No se dedujo
independencia solamente de la fecha ni de su etiqueta de calidad.

## Correspondencias individuales utilizadas en la base

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

## Identidad y grupo de evaluación son datos distintos

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

## Archivos y revisión

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
