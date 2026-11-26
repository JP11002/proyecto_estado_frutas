# Verificación de portabilidad — 25 de noviembre de 2026

Se creó una copia en una carpeta con otro nombre y espacios, incluyendo solo
archivos admitidos por las reglas de Git. Se excluyeron `.git`, `data/raw/`,
`data/processed/`, `experiments/results/` y `experiments/checkpoints/`. La copia
incluyó `data/base/`: CSV, estado, 291 recortes y 291 máscaras con rutas cortas.

Se creó además un entorno virtual nuevo y se instalaron las dependencias de
`requirements.txt` desde cero. No se utilizó `tmp/frutas_deps` ni el entorno
anterior. Posteriormente se instalaron las ruedas de CPU de PyTorch/torchvision.

## Entorno probado

Windows, Python 3.11.0, NumPy 2.4.6, pandas 3.1.1, Matplotlib 3.11.2,
scikit-learn 1.9.1, OpenCV headless 5.1.0, Pillow 13.0.0, joblib 1.6.0,
nbformat 5.11.1, nbclient 0.11.0, ipykernel 7.4.0 e ipywidgets 8.1.9.
CNN: PyTorch 2.15.0+cpu y torchvision 0.29.2+cpu. `pip check` no encontró
dependencias incompatibles.

## Comprobaciones realizadas

- El comprobador cargó las 291 imágenes, 43 características y 22 grupos, y verificó
  las huellas de recortes y máscaras sin originales descargados.
- El cuaderno 06 ejecutó todas sus celdas y generó EDA, modelos y métricas en la
  nueva carpeta, usando el Python de ese entorno.
- El cuaderno 07 tradicional ejecutó todas sus celdas, regeneró la referencia
  necesaria para la huella de la base única y guardó los modelos ajustados.
- Pasaron las cinco pruebas de `unittest`: ajuste tradicional y separación de
  grupos, entrenamiento/evaluación breve de CNN, rutas Windows/POSIX, integridad
  del snapshot y carga desde una carpeta renombrada sin datos procesados.
- La comprobación `--cnn` importó PyTorch/torchvision y realizó una inferencia en CPU.
- Se comprobó que las 291 etiquetas y grupos coinciden con la base original y que
  las 43 características son equivalentes con tolerancia absoluta de `1e-12`.
- Los archivos de la base están admitidos por `.gitignore`.

## Alcance de esta comprobación

La ejecución real se probó en Windows; la guía incluye comandos para Linux y
macOS, y las pruebas cubren la interpretación de rutas de ambos sistemas. La
verificación inicial de instalación utilizó una prueba sintética de CNN.
Posteriormente se completó el entrenamiento real y su comparación sobre las 291
imágenes; se documenta en [entrenamiento de CNN](entrenamiento_cnn.md).

Las fotos originales y el historial de adquisición no son necesarios para empezar
en los cuadernos 06/07. Para volver a revisar fotografías de alta resolución se
requiere recuperar los originales desde las fuentes, como indica `instalacion.md`.
El acceso futuro a esas fuentes depende de sus permisos y disponibilidad.

## Unificación de la base

Se consolidaron los datos en `data/base/datos.csv`, con imágenes y máscaras en
esa misma carpeta. Se retiraron las carpetas alternativas del proyecto y se
eliminó la elección entre copias: EDA, SVM, Random Forest y CNN importan el mismo
lector de `src/data/base.py`.

`scripts/actualizar_base.py` se ejecutó sobre las revisiones aprobadas y actualizó
esa base directamente, conservando los 291 ejemplos, sus etiquetas y 22 grupos.
El comprobador y las cinco pruebas pasaron tras la unificación. Una prueba
adicional confirma que una tabla temporal en `data/processed/` nunca reemplaza la
base oficial. También se comprobó una copia renombrada que contiene `data/base/`
y carece de `data/processed/`.

Los cuadernos 03, 04, 05, 06 y 07 se ejecutaron completos después de la unificación.
Los resultados de EDA y ajuste quedaron vinculados a la misma huella de
`data/base/datos.csv`.

Los cambios deben incluirse en el commit que se comparta con el equipo.

## Entrenamiento real de CNN y comparación

El cuaderno 08 terminó completo sobre `data/base/`, sin cambiar sus etiquetas ni
grupos. Se comprobaron 1746 predicciones externas de seis modelos; los pesos de
las tres CNN externas reprodujeron las predicciones guardadas y el prototipo final
se recargó correctamente. Pasaron las ocho pruebas. CNN obtuvo F1 macro 0,237,
frente a 0,373 de SVM ajustado y 0,429 de Random Forest inicial.

## Mejora de CNN sobre la misma base — 26 de noviembre de 2026

Se actualizó el mismo cuaderno 08 con MobileNetV3 Small congelada y búsqueda
interna de seis cabezales/ponderaciones. La base conserva las 291 imágenes,
43 características, 22 grupos y la misma huella. Se conservaron las predicciones
tradicionales y la referencia de la CNN anterior.

La CNN actual obtuvo F1 macro 0,472 y exactitud 70,4 %;
Random Forest inicial mantiene 0,429 y SVM ajustado
0,373. La CNN anterior conserva 0,237
como referencia histórica. El rendimiento intermedio sigue siendo limitado.

El cuaderno se ejecutó completo. El verificador comprobó 2037 predicciones de
siete modelos, separación de grupos y selección interna/global. Los pesos
externos reprodujeron sus etiquetas y probabilidades; la recarga final funcionó.
Pasaron las diez pruebas y el comprobador de integridad/arquitectura en CPU.
Los resultados corresponden al desarrollo, sin una prueba final independiente.

## Diagnóstico de intermedia y alternativas — 26 de noviembre de 2026

Se amplió y ejecutó el mismo cuaderno 08 completo, sin modificar la base ni sus
particiones. El análisis cubre las 75 intermedias de cinco grupos, su confusión
por fruta/campaña, la cobertura de entrenamiento y dos galerías con 20 imágenes.

Se compararon logística y SVM sobre rasgos CNN, Extra Trees y HistGradientBoosting
sobre las características existentes. Los espacios se fijaron antes de consultar
resultados. La selección usa validación interna por grupos; el escalado está
encapsulado en los pipelines y Boosting no usa parada temprana aleatoria.

El verificador de diagnóstico recalculó las 3201 predicciones de 11 modelos y
reprodujo los doce ajustes externos de las cuatro alternativas. Se conservaron
las predicciones previas y los grupos no se mezclaron. Boosting obtuvo F1 macro
0,487 y 19/75 intermedias frente a 0,472 y 6/75 de CNN, con menor exactitud global.

También se corrigió el recuento documentado de ajustes internos CNN: son 66,
con 3, 2 y 3 divisiones en los bloques externos y 3 en la selección global.
La base mantiene su huella y el cuaderno contiene todas sus salidas sin errores.


## Ejecución con fotografías completas — 26 de noviembre de 2026

Se migraron las 291 entradas CNN en la misma base a fotografía completa con
proporción conservada, relleno gris y resolución 224×224. No se modificaron las
etiquetas, las máscaras originales, los grupos ni las 43 características.
Huella vigente: `47bcd15916e6aa787ecaf1341cac98c31f392fd957499b07ca3e1592572528fc`.

Se completaron 132 ajustes internos CNN, tres externos y uno final. El cabezal
final es lineal, con aumentos, exponente de peso 0,5 y 20 épocas. F1 macro externo
0,458394; exactitud 0,673540; intermedias correctas 13/75. Los verificadores
reprodujeron predicciones y probabilidades de los pesos externos. La comparación
principal incluye 2328 predicciones de ocho modelos; el diagnóstico ampliado,
3492 de doce modelos. Las referencias CNN simple y recortada son históricas con
identidades y particiones verificadas, no pesos actuales de inferencia.

Pasaron 18 pruebas. Los cuadernos 06, 07 y 08 se ejecutaron de principio a fin
reutilizando, con comprobación de huella, los resultados del entrenamiento
completo realizado por terminal. Sin FRUTAS_USAR_RESULTADOS=1, los cuadernos
vuelven a ajustar los modelos. No se necesita repetir el entrenamiento para la
demostración.
