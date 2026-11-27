
# Análisis de impactos globales, sociales, ambientales y económicos (PI2)

El desarrollo de un sistema de clasificación automática de calidad y estimación del tamaño de frutas y verduras mediante técnicas de Machine Learning y Deep Learning tiene implicaciones que van más allá de su funcionamiento técnico. Aunque el objetivo principal del proyecto es automatizar el análisis de imágenes para identificar características visuales de los productos agrícolas, su posible implementación en mercados, supermercados y agroindustrias puede generar cambios en los procesos de selección, comercialización, aprovechamiento y distribución de alimentos.

Por esta razón, es necesario analizar los impactos potenciales de la solución desde diferentes perspectivas, considerando tanto los beneficios que podría proporcionar como los riesgos asociados a su utilización. Este análisis contempla cuatro dimensiones: social, económica, ambiental y global. Asimismo, se proponen estrategias de mitigación fundamentadas en las características técnicas del sistema y en los principios de responsabilidad social, sostenibilidad, equidad y bienestar público.

Es importante aclarar que, debido a que el proyecto corresponde a un prototipo académico, los impactos descritos representan principalmente efectos potenciales de una futura implementación. Su materialización dependerá de factores como el rendimiento de los modelos, la calidad y diversidad de los datos, las condiciones de operación y el contexto en el que se utilice la solución.

## 1. Impacto social

Desde la perspectiva social, el sistema puede contribuir a mejorar la manera en que se realizan los procesos de inspección y clasificación de frutas y verduras. Tradicionalmente, estas actividades dependen en gran medida de la observación y el criterio de las personas encargadas de evaluar los productos. Aunque la experiencia humana permite reconocer diferentes características de calidad, también pueden presentarse diferencias de criterio entre evaluadores, lo que genera cierta subjetividad e inconsistencia en los resultados.

La implementación de un sistema basado en visión por computadora podría ayudar a reducir estas diferencias mediante la aplicación de criterios de clasificación más uniformes. Al analizar características visuales como el tamaño relativo, la presencia de manchas, golpes, cambios de color y otros defectos visibles, el modelo puede proporcionar una clasificación que sirva como referencia para los responsables del proceso. Esto podría facilitar la inspección de grandes cantidades de productos y reducir el tiempo necesario para realizar evaluaciones individuales.

Otro posible beneficio social está relacionado con el aprovechamiento de alimentos. En algunos contextos comerciales, las frutas y verduras que presentan imperfecciones estéticas pueden recibir una valoración inferior o ser descartadas, incluso cuando sus características no necesariamente impiden su consumo. Una clasificación que diferencie adecuadamente los niveles de calidad podría facilitar la identificación de productos que todavía pueden comercializarse o destinarse a otros usos, contribuyendo potencialmente a un mejor aprovechamiento de los alimentos.

Sin embargo, la utilización de inteligencia artificial en este contexto también presenta riesgos importantes. Uno de ellos es la dependencia excesiva de las predicciones del modelo. Si los trabajadores o comerciantes consideran que el sistema siempre proporciona resultados correctos, podrían aceptar automáticamente sus clasificaciones sin realizar una inspección adicional. Esto resulta especialmente problemático cuando el modelo presenta errores o analiza productos cuyas características no estuvieron suficientemente representadas durante el entrenamiento.

Por ejemplo, una fruta que presenta una mancha superficial podría ser clasificada incorrectamente como de baja calidad, aunque dicha característica no represente necesariamente un problema para su aprovechamiento. De manera contraria, un producto con signos de deterioro podría recibir una clasificación favorable si el modelo no aprendió a reconocer adecuadamente ese tipo de defecto. En ambos casos, una decisión basada exclusivamente en la predicción podría perjudicar a comerciantes y consumidores.

También debe considerarse el posible efecto de la automatización sobre las actividades laborales. Si este tipo de soluciones se implementara a gran escala, algunas tareas de inspección y clasificación manual podrían transformarse o reducirse. Aunque la automatización puede mejorar la productividad, también podría generar preocupación entre los trabajadores cuyas actividades dependen de estos procesos.

Frente a estos riesgos, se considera que el sistema debe diseñarse como una herramienta de apoyo a la toma de decisiones y no como un reemplazo absoluto de la experiencia humana. La intervención de las personas continúa siendo importante para verificar los casos dudosos, interpretar situaciones que el modelo no reconoce correctamente y tomar decisiones que puedan afectar la comercialización o el aprovechamiento de los productos.

Como estrategias de mitigación, se propone evaluar el rendimiento del modelo mediante métricas que permitan identificar los errores por categoría, especialmente aquellos que puedan ocasionar consecuencias negativas para consumidores y comerciantes. El uso de matrices de confusión, precisión, exhaustividad (*recall*) y F1-score permite conocer qué tipos de productos o defectos generan mayores dificultades. Asimismo, es importante informar a los usuarios sobre las limitaciones del sistema y evitar que interpreten las predicciones como garantías absolutas de calidad o seguridad alimentaria.

Desde el punto de vista ético, estas decisiones se relacionan con la responsabilidad social y el principio del IEEE de velar por la seguridad, la salud y el bienestar del público. La tecnología debe contribuir a mejorar los procesos existentes sin generar una confianza injustificada en resultados automatizados ni afectar innecesariamente a las personas involucradas.

## 2. Impacto económico

En la dimensión económica, el sistema presenta oportunidades relacionadas con la optimización de los procesos de clasificación y selección de productos agrícolas. En mercados y agroindustrias, la inspección manual puede requerir una cantidad considerable de tiempo y personal, especialmente cuando se manejan grandes volúmenes de frutas y verduras. La utilización de un sistema automatizado podría agilizar parte de este proceso al analizar imágenes y generar predicciones de calidad y tamaño en un tiempo reducido.

Una clasificación más uniforme también podría facilitar la organización de los productos según sus características, permitiendo establecer procesos de selección más consistentes. Por ejemplo, los productos podrían agruparse según su tamaño relativo y la presencia de defectos visibles, lo que facilitaría su organización para diferentes canales de comercialización o procesos de transformación.

Otro beneficio económico potencial está relacionado con la reducción de pérdidas ocasionadas por decisiones de clasificación inconsistentes. Cuando un producto se clasifica incorrectamente, puede perder oportunidades de comercialización o recibir un tratamiento inadecuado dentro de la cadena de distribución. Una herramienta de apoyo basada en inteligencia artificial podría contribuir a disminuir este tipo de situaciones, siempre que presente un nivel de confiabilidad adecuado.

Además, la identificación de diferentes categorías de calidad podría ayudar a los comerciantes a establecer estrategias de aprovechamiento. Los productos que no cumplen con determinados estándares visuales podrían destinarse a procesos alternativos, en lugar de ser descartados automáticamente. Esto representa una oportunidad para mejorar el aprovechamiento económico de los recursos agrícolas.

No obstante, también existen costos y riesgos económicos asociados a la implementación. El desarrollo y funcionamiento de un sistema de visión por computadora requiere recursos tecnológicos como cámaras, dispositivos de procesamiento, almacenamiento y mantenimiento. Dependiendo de la complejidad del modelo, podrían necesitarse equipos con capacidades computacionales superiores, especialmente si se pretende realizar inferencia en tiempo real.

También deben considerarse los costos relacionados con la recolección, anotación y actualización de los datos. Para mantener un modelo confiable, es necesario disponer de imágenes representativas y etiquetas consistentes. Si cambian las variedades de frutas, las condiciones de captura o las características de los productos comercializados, podría ser necesario ampliar el conjunto de datos y volver a entrenar o ajustar el modelo.

Un riesgo económico adicional corresponde a las consecuencias de las predicciones incorrectas. Si el sistema clasifica como de baja calidad un producto que podría comercializarse, el vendedor podría experimentar pérdidas innecesarias. De igual manera, si un producto deteriorado recibe una clasificación favorable, podrían generarse reclamaciones, pérdidas de confianza y consecuencias económicas para los establecimientos que utilicen la solución.

Por esta razón, el rendimiento técnico del modelo debe analizarse en función de las consecuencias que pueden tener sus errores. No es suficiente considerar únicamente la exactitud general (*accuracy*), ya que esta métrica puede ocultar un desempeño deficiente en categorías específicas. Resulta necesario evaluar la precisión y el *recall* de cada clase, prestando especial atención a los errores que podrían tener mayores consecuencias comerciales o afectar el aprovechamiento de los productos.

Como estrategias de mitigación, se propone priorizar modelos que ofrezcan un equilibrio entre rendimiento y costo computacional. La comparación entre alternativas tradicionales de Machine Learning y modelos de Deep Learning permitirá identificar si la complejidad adicional de una arquitectura proporciona mejoras suficientemente importantes para justificar sus requerimientos de procesamiento.

Asimismo, se recomienda evaluar los costos de implementación y mantenimiento antes de una posible aplicación comercial, realizar pruebas en condiciones reales y mantener mecanismos de revisión humana para las clasificaciones que puedan representar pérdidas económicas importantes.

Desde una perspectiva ética, estas medidas se relacionan con la responsabilidad profesional y el deber de desarrollar soluciones que aporten beneficios reales a la sociedad. Una herramienta tecnológica no debería presentarse como económicamente conveniente sin considerar los costos de operación, sus limitaciones y las posibles consecuencias de sus errores.

## 3. Impacto ambiental

El impacto ambiental representa una dimensión especialmente importante para este proyecto, debido a su relación directa con el aprovechamiento de productos agrícolas y el posible desperdicio de alimentos.

Uno de los principales beneficios potenciales del sistema consiste en contribuir a una clasificación más consistente de las frutas y verduras según sus características visuales. En determinados procesos comerciales, los productos que presentan tamaños irregulares, manchas superficiales o imperfecciones estéticas pueden recibir una valoración inferior o ser descartados. Sin embargo, no todas estas características implican que el alimento sea inadecuado para su consumo o aprovechamiento.

Al proporcionar una clasificación más estructurada, el sistema podría facilitar la identificación de productos que todavía tienen posibilidades de comercialización o que pueden destinarse a otros procesos, como la elaboración de pulpas, jugos y otros productos derivados. De esta manera, una aplicación adecuada de la solución podría contribuir indirectamente a reducir el desperdicio de alimentos y aprovechar mejor los recursos utilizados durante su producción.

Este beneficio también tiene relación con el uso de recursos naturales. La producción agrícola requiere agua, suelo, energía, fertilizantes, transporte y trabajo humano. Cuando un producto se descarta innecesariamente, también se desperdician parcialmente los recursos empleados para producirlo y distribuirlo. Por tanto, mejorar los procesos de clasificación podría contribuir a disminuir pérdidas a lo largo de la cadena de comercialización.

Sin embargo, es importante reconocer que el sistema también tiene una huella ambiental asociada a su desarrollo y funcionamiento. El entrenamiento de modelos de Machine Learning y Deep Learning requiere recursos computacionales y consumo de energía. Particularmente, las redes neuronales convolucionales pueden necesitar múltiples iteraciones de entrenamiento y operaciones matemáticas sobre grandes cantidades de imágenes, lo que incrementa el uso de procesadores y, dependiendo de la arquitectura, de unidades de procesamiento gráfico.

El procesamiento continuo de imágenes en una aplicación en tiempo real también puede incrementar el consumo energético, especialmente si se utilizan equipos que permanecen funcionando durante periodos prolongados. Adicionalmente, una implementación que dependa de hardware especializado podría generar impactos asociados a la fabricación, transporte, mantenimiento y posterior disposición de estos dispositivos.

Otro aspecto que debe considerarse es la relación entre la complejidad del modelo y el beneficio ambiental obtenido. Un modelo considerablemente más grande no necesariamente representa una mejor alternativa si sus mejoras en clasificación son pequeñas frente al incremento en recursos computacionales. Por esta razón, la selección de la arquitectura debe considerar tanto las métricas de rendimiento como los requerimientos de procesamiento.

Como estrategias de mitigación, se propone optimizar los modelos y evitar entrenamientos innecesarios mediante una adecuada planificación de los experimentos. También se recomienda reutilizar los equipos disponibles, aprovechar los modelos de menor complejidad cuando ofrezcan resultados suficientes y limitar el procesamiento en tiempo real a las necesidades reales de la aplicación.

Desde el punto de vista de los datos, también es importante organizar adecuadamente las imágenes y evitar duplicaciones innecesarias, ya que una preparación eficiente del conjunto de datos puede reducir el almacenamiento y el procesamiento requerido.

Finalmente, debe aclararse que la reducción del desperdicio de alimentos es un beneficio potencial y no un resultado que pueda garantizarse únicamente mediante la clasificación automática. Para comprobar este impacto sería necesario realizar pruebas en un entorno real y medir variables como la cantidad de productos descartados antes y después de implementar el sistema.

Estas decisiones se relacionan con los principios de desarrollo sostenible y responsabilidad ambiental promovidos por el Código de Ética del IEEE. El objetivo no es solamente desarrollar un modelo preciso, sino procurar que su funcionamiento utilice los recursos tecnológicos de manera proporcional a los beneficios que puede generar.

## 4. Impacto global

Desde la dimensión global, el sistema tiene potencial de aplicación en diferentes contextos agrícolas y agroindustriales, debido a que la clasificación de frutas y verduras constituye una actividad relevante en las cadenas de producción, distribución y comercialización de alimentos de numerosos países.

La utilización de técnicas de visión por computadora podría facilitar la estandarización de ciertos procesos de inspección, especialmente en organizaciones que manejan grandes volúmenes de productos. Además, la posibilidad de analizar imágenes mediante una interfaz gráfica permitiría adaptar la solución a diferentes entornos de trabajo, siempre que se disponga del hardware y las condiciones necesarias para su funcionamiento.

La solución también podría contribuir a procesos de transformación digital en pequeñas y medianas empresas agrícolas, que podrían utilizar herramientas de análisis automatizado para apoyar la selección de productos y mejorar la organización de sus operaciones.

Sin embargo, uno de los principales desafíos para su aplicación global es la diversidad de condiciones en las que se producen y comercializan las frutas y verduras. Los productos agrícolas presentan diferencias relacionadas con sus variedades, regiones de cultivo, condiciones climáticas, características del suelo, métodos de producción y etapas de maduración.

Estas diferencias pueden modificar considerablemente la apariencia visual de los productos. Por ejemplo, una misma fruta puede presentar variaciones naturales de color, forma y tamaño dependiendo de su variedad y procedencia. Si el modelo se entrena principalmente con imágenes de una variedad específica, podría interpretar incorrectamente características normales de otras variedades.

También existen diferencias en las condiciones de captura. La iluminación, el tipo de cámara, la resolución, la distancia entre el dispositivo y el producto y las características del fondo pueden modificar la apariencia de las imágenes. Un modelo entrenado con fotografías tomadas en condiciones controladas podría presentar un rendimiento inferior cuando se utiliza en mercados o instalaciones con iluminación variable y fondos diferentes.

Otro aspecto relevante corresponde a los estándares de calidad utilizados en cada región. Los criterios comerciales para clasificar frutas y verduras pueden variar según el mercado de destino, las regulaciones aplicables y las características del producto. Por tanto, no debe asumirse que una clasificación desarrollada para un contexto específico será automáticamente válida en todos los países.

Además, una implementación internacional debe considerar las regulaciones relacionadas con la comercialización de alimentos, la protección de datos cuando existan imágenes que contengan información personal y las condiciones de utilización de los conjuntos de datos empleados para entrenar los modelos.

Como estrategias de mitigación, se propone ampliar progresivamente el conjunto de datos con fotografías de diferentes variedades, regiones y condiciones de captura. También es importante evaluar el modelo utilizando imágenes externas que no hayan participado en el entrenamiento, con el propósito de determinar si sus resultados se mantienen en contextos diferentes.

Antes de implementar la solución en una región distinta a aquella en la que fue desarrollada, sería necesario realizar una validación local, revisar los criterios de clasificación aplicables y, cuando corresponda, ajustar o reentrenar el modelo con datos representativos del nuevo contexto.

Desde el punto de vista técnico, estas medidas están relacionadas con la capacidad de generalización de los modelos de aprendizaje automático. Un buen desempeño sobre el conjunto de prueba original no garantiza que el sistema funcione correctamente ante distribuciones de datos diferentes. Por ello, la evaluación externa y la identificación de posibles cambios en los datos de entrada son fundamentales para evitar una aplicación inadecuada de la solución.

En términos éticos, este análisis se relaciona con los principios de equidad, responsabilidad profesional y consideración de los impactos globales de las tecnologías. Una solución de ingeniería responsable debe reconocer que los contextos agrícolas no son homogéneos y evitar generalizar resultados sin contar con evidencia suficiente.

## 5. Estrategias generales de mitigación y mejora

A partir del análisis de las cuatro dimensiones, se identifican algunas acciones transversales que pueden contribuir a disminuir los riesgos y fortalecer la implementación responsable del sistema.

En primer lugar, se considera fundamental mejorar continuamente la calidad y representatividad de los datos. La incorporación de imágenes de diferentes variedades, tamaños, estados de madurez, defectos y condiciones de iluminación permite reducir la dependencia del modelo respecto a características particulares del conjunto de entrenamiento. Asimismo, la revisión de las anotaciones y la definición de criterios comunes contribuyen a disminuir inconsistencias que podrían afectar los resultados.

En segundo lugar, se debe mantener un proceso de evaluación riguroso. La utilización de métricas como precisión, *recall*, F1-score y matrices de confusión permite identificar las categorías en las que los modelos presentan mayores dificultades. También resulta importante analizar los errores de clasificación y comparar los resultados de los modelos tradicionales de Machine Learning con los obtenidos mediante Deep Learning.

En tercer lugar, se propone considerar el costo computacional como parte del proceso de selección del modelo. La arquitectura que obtenga el mejor resultado no necesariamente será la alternativa más adecuada si requiere recursos tecnológicos desproporcionados para el problema que se busca resolver. Por ello, es conveniente analizar el tiempo de entrenamiento, el tiempo de inferencia y los requerimientos de hardware, además de las métricas de clasificación.

En cuarto lugar, se debe garantizar que la solución comunique claramente sus capacidades y limitaciones. Los usuarios deben comprender que el sistema realiza estimaciones basadas en características visuales y que sus resultados no constituyen una certificación de calidad comercial ni una evaluación de seguridad alimentaria. La intervención humana debe mantenerse en los casos que requieran una valoración adicional.

Finalmente, cualquier implementación futura debe acompañarse de una evaluación de sus resultados en condiciones reales. Esto permitiría determinar si las mejoras técnicas se traducen efectivamente en beneficios sociales, económicos y ambientales, evitando asumir que un buen desempeño experimental garantiza automáticamente un impacto positivo en la sociedad.

## 6. Conclusión del análisis de impactos

El análisis de las dimensiones social, económica, ambiental y global permite reconocer que el sistema de clasificación automática de frutas y verduras presenta oportunidades importantes para mejorar los procesos de selección y aprovechamiento de productos agrícolas. Su utilización podría contribuir a reducir la subjetividad de la inspección manual, optimizar los tiempos de clasificación, facilitar la organización de los productos y disminuir potencialmente el desperdicio de alimentos.

Sin embargo, estos beneficios dependen directamente de la calidad de los datos, la confiabilidad de las predicciones y las condiciones en las que se implemente la solución. Los errores de clasificación pueden generar consecuencias para comerciantes y consumidores, mientras que los costos tecnológicos y el consumo energético representan factores que deben considerarse durante su diseño y despliegue.

Asimismo, la diversidad de variedades agrícolas y condiciones de captura demuestra que los modelos no deben considerarse universalmente aplicables sin una validación adicional. La generalización requiere conjuntos de datos representativos y evaluaciones realizadas en contextos diferentes a los utilizados durante el entrenamiento.

En conclusión, el desarrollo responsable de esta solución exige considerar el rendimiento técnico como uno de varios criterios de evaluación. La eficiencia computacional, la supervisión humana, la representatividad de los datos, la transparencia de los resultados y la evaluación de los impactos reales son elementos necesarios para procurar que la tecnología genere beneficios sostenibles.

De esta manera, el proyecto busca demostrar que la aplicación de inteligencia artificial en el sector agrícola no debe orientarse únicamente a automatizar tareas, sino también a desarrollar herramientas que contribuyan al aprovechamiento de los recursos, al bienestar de las personas y a la mejora responsable de los procesos productivos.

## 7. Análisis final con evidencia (26/11/2026)

Tras la evaluación con prueba independiente (cuaderno 09, `docs/ajuste_y_cnn.md`), los
impactos se revisan con cifras medidas, no supuestas:

| Dimensión | Evidencia | Conclusión y mitigación |
|---|---|---|
| Social | En la prueba, 5 de 9 frutas malas se predijeron buenas; en las fotos propias no se acertó ninguna de las 6 defectuosas. | El riesgo de aprobar fruta deteriorada es real: el sistema solo puede usarse como apoyo con revisión humana. |
| Social (privacidad) | Una cámara en un punto de venta puede percibirse como vigilancia de clientes o trabajadores. | La interfaz encuadra solo la fruta, procesa localmente sin red y no guarda fotogramas; se recomienda señalizar su uso y orientarla a la banda o mesa de inspección. |
| Económica | 10 de 18 frutas buenas se marcaron intermedias o malas. | Descartar automáticamente produciría pérdidas; un piloto debe registrar el coste de cada tipo de error antes de cualquier uso comercial. |
| Ambiental | Clasificar tarda unos 0,03 s en CPU y el análisis completo de una foto de 12 MP unos 3,3 s; el experimento completo tardó unos 9 minutos en CPU. | No se requiere GPU; el consumo energético es bajo. No se midió reducción de desperdicio, por lo que no se afirma. |
| Global | El desempeño cambia entre fuentes (fondo, resolución, variedad) y entre frutas. | Antes de usarlo en otro mercado se necesita validación local por fruta y un protocolo de captura. |
