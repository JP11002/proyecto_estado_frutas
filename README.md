# Estado de frutas: alcance vigente

Manzana roja, banano y limón. Integración de fuentes públicas y fotografías propias del equipo: [registro de la versión 2](docs/integracion_fuentes_v2.md). La base anterior con naranjas se conserva como antecedente. Configuración: [alcance actual](config/alcance_actual.json).

Continuar en [06 — EDA y entrenamiento de SVM y Random Forest](notebooks/06_eda_y_modelos_tradicionales.ipynb).
El cuaderno está ejecutado: incluye EDA, comparación inicial mediante tres pliegues
agrupados, análisis de errores y guardado de ambos modelos. Resultados y reproducción:
[primer experimento](docs/eda_y_modelos_tradicionales.md).
La integración y revisión se conserva en [04 — Base vigente](notebooks/04_integracion_fuentes_y_base_v2.ipynb).
La salida actual contiene 291 ejemplares revisados: 270 de Drive y 21 compatibles
de la versión anterior. Las nueve combinaciones de fruta y calidad están presentes.
Los cinco casos dudosos quedaron resueltos con evidencia de apariencia externa.
Las 318 fotos nuevas tienen una disposición explícita: 270 incluidas, 22 descartadas
por máscara/encuadre y 26 con varios frutos apartadas. Los grupos actuales permiten
la comparación exploratoria; una prueba final requiere capturas independientes.

Las fotografías propias ya tienen una [propuesta de identidades y grupos seguros](docs/agrupacion_frutos_propios.md).
Revisar o corregir las correspondencias en [05 — Agrupación de frutos propios](notebooks/05_agrupacion_frutos_propios.ipynb).
Las 25 fotos propias de la base se vinculan con 12 identidades propuestas y 10 grupos
conservadores; la base completa utiliza 22 grupos, manteniendo unidas las dudas de identidad.

El primer entrenamiento obtiene F1 macro OOF de **0,429 para Random Forest** y
**0,299 para SVM**, frente a **0,272** de la referencia mayoritaria. Son métricas
de desarrollo con fuerte desbalance; no una prueba independiente ni una selección definitiva.

Los datos descargados, resultados generados, modelos y entornos locales se excluyen
mediante `.gitignore`. El código, los cuadernos y las anotaciones revisadas se
conservan; ver [qué archivos se guardan en Git](docs/control_versiones.md).

---

____________________________________________________________________________________


# Project Template
This is the template for the Project I course at the IA Master, Universidad Icesi, Cali Colombia

This template is based on the template proposed by the [Data Science Working Group] (https://github.com/sfbrigade/data-science-wg) Code for the [San Francisco’s Code initiative](https://github.com/sfbrigade/data-science-wg) 

*Instructions: Edit this template filling in the titles, information, and links! Feel free to stray a bit to suit your project but try to provide the main information for reviews and feedback purposes.*
## Remove this table after updating your project's information
____________________________________________________________________________________

# Project Name
This project is a part of the  **Proyecto 1 de Innovación Tecnológica** course in the Applied Artificial Intelligence Master, Universidad Icesi, Cali Colombia. 

#### -- Project Status: [Active, On-Hold, Completed]

## Contributing Members

**Team Leader: [Full Name](https://github.com/[github handle])(@slackHandle)**
**Instructor: [Full Name](https://github.com/[github handle])(@slackHandle)**

#### Other Members:

|Name     |  Email   | 
|---------|-----------------|
|[Full Name](https://github.com/[github handle])| @johnDoe        |
|[Full Name](https://github.com/[github handle]) |     @janeDoe    |

## Contact
* Feel free to contact the team leader or the instructor with any questions or if you are interested in contributing!


## Project Intro/Objective
The purpose of this project is ________. (Describe the main goals of the project and potential civic impact. Limit to a short paragraph, 3-6 Sentences)

### Partner
This section should be added when there's a partner institution 
* [Name of Partner organization/Government department etc..]
* Website for partner
* Partner contact: [Name of Contact], [slack handle of contact if any]
* If you do not have a partner leave this section out

### Methods Used
* Inferential Statistics
* Machine Learning
* Data Visualization
* Predictive Modeling
* etc.

### Technologies
* R 
* Python
* D3
* PostGres, MySql
* Pandas, jupyter
* HTML
* JavaScript
* etc. 

## Project Description
(Provide a more detailed overview of the project.  Talk a bit about your data sources and what questions and hypotheses you are exploring. What specific data analysis/visualization and modeling work are you using to solve the problem? What blockers and challenges are you facing?  Feel free to number or bullet point things here)

## Getting Started
Instructions for contributors
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.

    *If using offline data mention that and how contributors may obtain the data )*
    
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  

5. Follow setup [instructions](Link to file)

## Featured Notebooks/Analysis/Deliverables
* [Notebook/Markdown/Slide Deck Title](link)
* [Notebook/Markdown/Slide DeckTitle](link)
* [Blog Post](link)


