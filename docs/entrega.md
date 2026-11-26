# Requisitos de la entrega

Integrantes: Juan Pablo Serrano, Juan Castaño y Felipe Calderon.
Repositorio configurado: https://github.com/JP11002/proyecto_estado_frutas
Sustentación remota: 30 de noviembre de 2026, según la indicación del curso; agenda y enlace pendientes de comunicación del profesor.

| Requisito | Evidencia y estado |
|---|---|
| Problema, contexto y objetivos | Informe, secciones I y II; cuaderno de comprensión del negocio |
| CRISP-DM aplicado con evidencia | Informe, sección III; figura vectorial personalizada y cuadernos 01–08 |
| Datos públicos y propios | Base única de 291 imágenes; inventario de 318 fotos nuevas, incluidas 51 propias recolectadas y 25 aceptadas |
| Calidad y tamaño | Tres etiquetas visuales; máscaras revisadas; tabla data/base/medidas_tamano.csv y medida normalizada en aplicación |
| EDA, desbalance y calidad de imágenes | Cuaderno 06 y distribución por fruta, clase y grupo |
| Dos modelos ML y uno DL | SVM, Random Forest y CNN; pesos locales y código de inferencia |
| Ajuste de hiperparámetros | Cuadernos 07/08; selección interna separada de evaluación externa |
| Comparación y métricas | 2037 predicciones de la comparación principal; diagnóstico ampliado de 3201 predicciones de 11 modelos |
| Análisis de errores y limitaciones | Informe VI y diagnóstico de intermedia; sin afirmar prueba final independiente |
| Interfaz con foto y cámara | python -m src.main; carga de foto y ejemplos probados visualmente; cámara implementada, requiere dispositivo y permisos locales |
| Formulación matemática PI3 | Informe II: clasificación, normalización, kernel, pérdida, métricas, tamaño y separación de grupos |
| Ética PI1 con código profesional | Informe VIII: tabla de decisiones, principios IEEE y medidas concretas |
| Impactos PI2 iniciales y finales | Informe VIII y docs/analisis_de_impactos.md: social, económico, ambiental y global |
| Informe IEEE de máximo 7 páginas | docs/informe_final.pdf, cuatro páginas, autores correctos, fuentes y figuras vectoriales |
| Enlace del repositorio dentro del PDF | Incluido; confirmar que los cambios locales estén publicados antes de enviar |
| Enlace del video dentro del PDF | PENDIENTE: grabar, publicar y añadir el enlace compartible |
| Video de máximo 10 minutos | PENDIENTE DEL EQUIPO; guion de aproximadamente nueve minutos en docs/guion_video.md |
| Repositorio documentado y reproducible | README, dependencias, base con rutas relativas, cuadernos y scripts; pruebas e inferencia verificadas |

## Abrir la demostración

En este equipo ya está abierto el prototipo. Desde un entorno instalado, ejecutar en la raíz:

```text
python -m src.main
```

Para usar una imagen de entrenamiento sin abrir interfaz:

```text
python -m src.main --ejemplar ID_DE_LA_BASE --modelo CNN
```

Para una fotografía nueva:

```text
python -m src.main --imagen ruta/a/foto.jpg --modelo CNN
```

La aplicación muestra tamaño relativo al encuadre. No convertir esa salida a centímetros ni presentar las imágenes de ejemplo como prueba independiente. La propuesta de máscara automática debe incluir todo el fruto y sus defectos; usar un fondo simple. Se verificó una foto propia original mediante segmentación automática e inferencia.

## Cerrar el PDF cuando exista el video

1. Grabar con el guion y comprobar que el video se escuche y no supere diez minutos.
2. Publicar el video con acceso de lectura para el profesor y probar el enlace.
3. Guardar ese enlace en el campo `video` de `config/entrega.json`.
4. Regenerar el DOCX con `python scripts/generar_informe.py` y exportar con `powershell -ExecutionPolicy Bypass -File scripts/exportar_pdf.ps1` si se dispone de Word en Windows. Las dependencias de autoría están en requirements-informe.txt.
5. Abrir el PDF y comprobar los dos enlaces, los autores y el número de páginas. El texto fuente está en report.md; la plantilla original se conserva.
6. Confirmar que el repositorio remoto contiene el código actual. Subir el PDF definitivo a INTU.

Mientras falte el enlace del video, el PDF está preparado pero NO está listo para la entrega definitiva que pide INTU.

## Comprobaciones

Pasaron 13 pruebas, incluidas geometría de tamaño, rutas portables, integridad de la base, agrupación y modelos. Se comprobó inferencia con CNN, Random Forest y SVM; también la creación de la interfaz y una predicción visible. El tamaño se deriva de máscaras revisadas; no existe referencia física ni anotación ordinal manual independiente para reportar exactitud de tamaño. La base no cambió su huella al añadir la tabla de medidas.

## Distribución

Los modelos y resultados generados se excluyen de Git por diseño. Para incluirlos en un archivo local de entrega se puede ejecutar `python scripts/empaquetar_entrega.py`. El ZIP es un paquete del mismo proyecto, sin crear otra carpeta de trabajo ni otra base. Contiene también el estado de los pendientes en su manifiesto. En un clon del repositorio se pueden regenerar los modelos ejecutando 06, 07 y 08, según la guía de instalación.
