# Guion para el video de entrega

Duración objetivo: 8 minutos 30 segundos a 9 minutos 30 segundos. Límite de la consigna: 10 minutos. Participan Juan Pablo Serrano, Juan Castaño y Felipe Calderon. Grabar pantalla y voz con cualquier herramienta disponible. No es necesario editar escenas complejas. El enlace compartible debe quedar dentro del PDF antes de entregarlo.

## Preparación inmediata

1. Abrir el PDF del informe y el cuaderno 08 con sus salidas ya ejecutadas. No volver a entrenar durante la grabación.
2. Abrir la aplicación: `python -m src.main`, desde el entorno del proyecto.
3. Tener una foto con un fruto completo, fondo uniforme y buena luz. Los botones de ejemplo permiten demostrar sin cámara y se identifican como entrenamiento.
4. Probar el micrófono. Grabar a 1280 por 720 o superior y evitar mostrar información personal de otras ventanas.
5. Si la cámara no está disponible, demostrar carga de foto y explicar que la captura está implementada pero requiere dispositivo y permisos locales.

## 0:00 a 0:40 Presentación y problema

Juan Pablo Serrano:

“Somos Juan Pablo Serrano, Juan Castaño y Felipe Calderon. Presentamos un sistema de apoyo para clasificar la calidad visual de manzana roja, banano y limón. La inspección manual depende del criterio del evaluador y de las condiciones de captura. Nuestro objetivo es recibir la fotografía de un fruto, estimar su calidad como buena, intermedia o mala y calcular su tamaño relativo en la imagen. El sistema evalúa apariencia externa; no determina seguridad alimentaria ni estado interno.”

Mostrar: título del informe y objetivo.

## 0:40 a 2:10 Datos y CRISP-DM

Juan Pablo Serrano:

“Aplicamos las seis fases de CRISP-DM con evidencia en el repositorio: contexto del negocio, comprensión de datos, preparación, modelado, evaluación y despliegue. Usamos Kaggle y dos colecciones de Drive, una pública y otra propia. Recolectamos 51 fotos propias; 25 entraron en la base y las demás se excluyeron por contener varios frutos o por problemas de segmentación. El inventario conserva las decisiones.

La base única tiene 291 imágenes, sus máscaras revisadas, 43 características y 22 grupos conservadores. Hay 201 imágenes buenas, 75 intermedias y 15 malas. Una limitación importante es que 70 intermedias son bananos y 67 pertenecen a una sola campaña. No presentamos esos 22 grupos como 22 frutos físicos confirmados: algunos son campañas cuya independencia no conocemos.”

Mostrar: diagrama CRISP-DM, tabla de clases y base única. No mostrar todo el CSV ni leer cada columna.

## 2:10 a 3:05 Preparación y tamaño

Juan Pablo Serrano:

“Revisamos etiquetas, contornos y recortes. Conservamos los defectos oscuros dentro de la máscara para no borrar la evidencia que necesita el clasificador. Los modelos tradicionales usan color, textura y forma, mientras que la CNN recibe un recorte RGB de 224 por 224 con proporciones preservadas y relleno gris.

El tamaño se obtiene antes de ese recorte. Si el área segmentada es A, calculamos el diámetro del círculo de área equivalente como dos veces la raíz de A sobre pi. Lo dividimos por el lado menor de la fotografía para obtener una medida normalizada. También mostramos qué fracción de la imagen ocupa el fruto. Es una comparación de píxeles, no una medida en centímetros: la distancia de la cámara y el encuadre afectan el resultado.”

Mostrar: una máscara y las ecuaciones de tamaño.

## 3:05 a 4:25 Modelos y evaluación

Juan Castaño:

“Entrenamos SVM y Random Forest y ajustamos sus hiperparámetros mediante búsquedas internas. También entrenamos una CNN simple de tres bloques y después una CNN con MobileNetV3 preentrenada. En la segunda mantenemos fijo el extractor y añadimos un clasificador propio. Probamos dos tipos de cabezal y tres formas de ponderar las clases.

La evaluación utiliza tres bloques externos. Las fotos de un mismo grupo permanecen juntas para evitar que distintas vistas relacionadas aparezcan a ambos lados de la evaluación. Cada imagen recibe una predicción externa. Los parámetros se seleccionan dentro del entrenamiento externo; la evaluación no elige las épocas ni el umbral. La normalización aprendida se ajusta dentro del pipeline.

Usamos F1 macro como indicador principal porque da igual peso a las tres calidades. Lo acompañamos de precisión, recall, exactitud y matrices de confusión. La referencia mayoritaria siempre predice buena y permite mostrar por qué la exactitud aislada es insuficiente.”

Mostrar: tabla de hiperparámetros y explicación de los bloques.

## 4:25 a 6:05 Resultados y análisis crítico

Juan Castaño:

“SVM ajustado obtuvo F1 macro de 0,373 y Random Forest inicial de 0,429. La CNN simple tuvo 0,237; la CNN de transferencia mejoró a 0,472 y 70,4 por ciento de exactitud. No ocultamos que Random Forest ajustado quedó por debajo de su configuración inicial.

El problema más difícil es la clase intermedia. La CNN acierta 6 de 75 intermedias, 192 de 201 buenas y 7 de 15 malas. Cuando se evalúa la campaña que concentra 67 intermedias, el entrenamiento solo dispone de ocho intermedias de otras procedencias. Eso limita la generalización.

Comparamos además cuatro alternativas. Gradient Boosting sobre las 43 características obtuvo F1 macro de 0,487 y detectó 19 intermedias. Sin embargo, su exactitud bajó a 64,9 por ciento y aumentaron las falsas alarmas. La CNN conserva mejor F1 cuando damos el mismo peso a cada grupo. Por eso no declaramos un ganador universal: la selección depende del coste de confundir cada calidad.

Estas son métricas de desarrollo. Las imágenes ya participaron en revisiones de etiquetas y máscaras. Una prueba final independiente necesita nuevas capturas reservadas antes de ajustar el procesamiento.”

Mostrar: figura comparativa, matriz CNN y análisis del tercer bloque. Evitar leer todos los modelos de la tabla.

## 6:05 a 7:35 Demostración

Felipe Calderon:

“Esta es la interfaz local. Podemos cargar una fotografía o iniciar la cámara y seleccionar CNN, Random Forest o SVM. Aquí se muestra la calidad estimada, el recorte utilizado, el contorno propuesto y el tamaño relativo.

Voy a cargar una imagen de un fruto sobre fondo simple. Revisamos el contorno, porque si incluye fondo o corta parte del fruto puede alterar la predicción y el tamaño. El diámetro equivalente aparece en píxeles y normalizado respecto al encuadre. Las puntuaciones del clasificador no se presentan como confianza calibrada.

También hay ejemplos de la propia base para una demostración sin cámara. La aplicación los marca como imágenes de entrenamiento y muestra su etiqueta revisada; no los utilizamos para calcular las métricas de prueba.”

Acciones: cargar foto, analizar, señalar contorno y tamaño; cambiar de modelo y pulsar Analizar; si procede mostrar captura de cámara. No esconder una predicción equivocada: explicar que es un prototipo con errores documentados.

## 7:35 a 8:40 Ética e impactos

Felipe Calderon:

“Relacionamos las decisiones con el código de ética de IEEE: bienestar público, privacidad, honestidad y responsabilidad técnica. El sistema trabaja localmente, no almacena automáticamente las capturas de cámara y conserva la procedencia de las imágenes. Reconocemos los datos de terceros y la asistencia de IA en código y anotación.

En impacto social puede apoyar inspecciones, pero la confianza excesiva es un riesgo. Económicamente puede ayudar a estandarizar procesos, aunque los errores tienen coste. Ambientalmente podría facilitar aprovechamiento, pero no hemos medido reducción de desperdicio ni emisiones. En otros mercados cambiarían variedades y condiciones de luz, por lo que haría falta evaluación local.

El análisis final recomienda mantener supervisión humana y evitar decisiones automáticas de descarte con el rendimiento actual.”

Mostrar: tablas PI1 y PI2 del informe.

## 8:40 a 9:15 Conclusiones y acceso

Felipe Calderon:

“Entregamos una base única, cuadernos ejecutados, modelos comparados, análisis de errores, una interfaz local y documentación para reproducir el proyecto. Aprendimos que la variedad independiente de los datos puede limitar más que la complejidad del modelo. Las siguientes mejoras son revisar el criterio de intermedia por fruta, combinar características manuales y CNN y evaluar un ajuste parcial con nuevas capturas reservadas. El PDF incluye los enlaces al repositorio y a este video. Gracias.”

## Después de grabar

- Comprobar que el video se escuche, no exceda diez minutos y muestre la demostración.
- Subirlo a la plataforma elegida por el equipo y probar el enlace con acceso de lectura para el profesor.
- Enviar el enlace para incorporarlo en `config/entrega.json` y regenerar el PDF.
- Confirmar que el repositorio remoto contiene los cambios actuales; el repositorio local y GitHub son estados distintos.
- Subir el PDF definitivo a INTU. La sustentación remota es el 30 de noviembre, según la indicación del curso; la agenda y el enlace los comunica el profesor.
