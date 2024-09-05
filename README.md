# Clasificación de Vinos usando PCA y Naive Bayes

## Descripción
Este proyecto tiene como objetivo clasificar diferentes tipos de vinos basados en sus propiedades químicas utilizando el Conjunto de Datos de Vinos (Wine Dataset) del repositorio UCI Machine Learning. El enfoque principal es aplicar Análisis de Componentes Principales (PCA) para la reducción de dimensionalidad y entrenar un modelo de clasificación utilizando el algoritmo Naive Bayes.

## Tabla de Contenidos
- [Descripción](#descripción)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Exploración de Datos](#exploración-de-datos)
- [Entrenamiento del Modelo](#entrenamiento-del-modelo)
- [Resultados](#resultados)
- [Conclusión](#conclusión)
- [Créditos](#créditos)

## Estructura del Proyecto
El proyecto está organizado en las siguientes carpetas:

wine-classification-pca-naivebayes/
│
├── data/
│   ├── raw/                         # Datos crudos descargados
│   │   ├── wine.data                # Conjunto de datos original
│   │   └── wine.names               # Descripción del conjunto de datos
│   ├── processed/                   # Datos procesados y limpios
│   │   └── cleaned_wine_data.csv    # Datos finales utilizados
│
├── docs/                            # Documentación
│   └── final_report.pdf             # Informe final del proyecto
│
├── environments/                    # Configuración del entorno virtual
│   └── env/                         # Entorno virtual de Python
│
├── models/                          # Modelos entrenados
│   ├── naive_bayes_model.pkl        # Modelo Naive Bayes entrenado
│   └── best_naive_bayes_model.pkl   # Mejor modelo entrenado
│
├── notebooks/                       # Notebooks de Jupyter
│   ├── data_exploration.ipynb       # Exploración de datos y visualización
│   └── model_training.ipynb         # Entrenamiento y evaluación del modelo
│
├── results/                         # Resultados del modelo
│   ├── evaluation_plots.png         # Gráficos de evaluación del modelo
│   └── model_metrics.txt            # Métricas del modelo
│
├── scripts/                         # Scripts para procesamiento y entrenamiento
│   ├── data_cleaning.py             # Limpieza y preparación de los datos
│   ├── data_visualization.py        # Visualización de los datos
│   └── train_model.py               # Script para entrenar el modelo
│
├── src/                             # Código fuente principal
│   ├── naive_bayes_classifier.py    # Implementación del clasificador Naive Bayes
│   └── pca_analysis.py              # Análisis de PCA
│
├── .gitignore                       # Archivos ignorados por git
├── README.md                        # Este archivo
└── requirements.txt                 # Dependencias del proyecto


## Requisitos
Para ejecutar este proyecto, se necesita instalar las siguientes bibliotecas de Python:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `joblib`

Puedes instalar todas las dependencias ejecutando el siguiente comando:

pip install -r requirements.txt

Instalación
Clonar el repositorio:

git clone https://github.com/SanMaBruno/wine-classification-pca-naivebayes.git

Navegar a la carpeta del proyecto:

cd wine-classification-pca-naivebayes

Configurar el entorno virtual (opcional pero recomendado):

python -m venv env
source env/bin/activate  # En Windows usa: env\Scripts\activate

Instalar las dependencias:

pip install -r requirements.txt

## Estructura de Datos
Coloca el archivo `wine.data` en la carpeta `data/raw/` o utiliza el archivo procesado `cleaned_wine_data.csv` ubicado en `data/processed/`.

## Exploración de Datos
El archivo `data_exploration.ipynb` contiene la exploración inicial de los datos. En este notebook se revisa:

- Estadísticas descriptivas del conjunto de datos.
- Visualización de características mediante gráficos.
- Verificación de valores faltantes.
- Distribución de las clases de vino.

## Entrenamiento del Modelo
El entrenamiento del modelo se realiza en el notebook `model_training.ipynb` y el script `train_model.py`. Los principales pasos son:

### Preprocesamiento de Datos:
- Escalado de los datos utilizando `StandardScaler`.
- División en conjunto de entrenamiento y prueba.

### Reducción de Dimensionalidad:
- Aplicación de PCA para reducir la cantidad de características del conjunto de datos.

### Entrenamiento del Modelo:
- Entrenamiento del clasificador Naive Bayes con los datos preprocesados.

### Evaluación del Modelo:
- Cálculo de la precisión y otras métricas de evaluación, como la matriz de confusión.

## Resultados
El modelo Naive Bayes obtuvo una exactitud del 100% al utilizar 2 componentes principales con PCA. Los resultados de la evaluación, incluyendo gráficos de la matriz de confusión y métricas detalladas, se encuentran en la carpeta `results/`.

## Conclusión
Este proyecto demuestra la efectividad de utilizar PCA para reducir la dimensionalidad y mejorar el rendimiento de un modelo Naive Bayes para la clasificación de vinos. Algunas conclusiones clave incluyen:

- PCA permitió reducir la cantidad de características sin pérdida significativa de precisión.
- El modelo Naive Bayes es adecuado para este conjunto de datos debido a su rendimiento sobresaliente.
- Este pipeline puede mejorarse con técnicas adicionales como el ajuste de hiperparámetros y validación cruzada.
