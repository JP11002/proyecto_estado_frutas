# Revisión de etiquetas y máscaras — base inicial v1

> Antecedente: desde el 24/11/2026 el equipo sustituyó naranja por limón. La base
> vigente y la integración posterior se describen en `integracion_fuentes.md`.

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

## Criterio aplicado

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

## Máscaras y características

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

## Archivos y reproducción

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

## Alcance del arranque de modelos

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
