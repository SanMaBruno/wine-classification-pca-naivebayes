# Script for Data Visualization 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

def plot_distribution(df, column_name):
    """
    Genera un histograma para mostrar la distribución de una columna específica.

    Args:
        df (DataFrame): DataFrame de pandas que contiene los datos.
        column_name (str): Nombre de la columna para la cual se quiere mostrar la distribución.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column_name], kde=True)
    plt.title(f'Distribución de {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.show()

def plot_correlation_matrix(df):
    """
    Genera un mapa de calor para mostrar la matriz de correlación de las características del DataFrame.

    Args:
        df (DataFrame): DataFrame de pandas que contiene los datos.

    Returns:
        None
    """
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlación de las Características')
    plt.show()

def plot_pairwise_relationships(df, hue_column=None):
    """
    Genera un gráfico de pares para mostrar las relaciones entre todas las características numéricas.

    Args:
        df (DataFrame): DataFrame de pandas que contiene los datos.
        hue_column (str, optional): Columna que se utilizará para colorear los puntos según la categoría.

    Returns:
        None
    """
    sns.pairplot(df, hue=hue_column)
    plt.show()

def plot_pca_variance(pca):
    """
    Genera un gráfico de barras para mostrar la variación explicada por cada componente principal.

    Args:
        pca (PCA): Objeto PCA ajustado de sklearn.

    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.title('Varianza Explicada por cada Componente Principal')
    plt.xlabel('Componentes Principales')
    plt.ylabel('Varianza Explicada')
    plt.grid(True)
    plt.show()

def plot_pca_2d(df, pca, targets):
    """
    Genera un gráfico 2D de los primeros dos componentes principales para visualizar los datos después del PCA.

    Args:
        df (DataFrame): DataFrame de pandas que contiene los datos originales.
        pca (PCA): Objeto PCA ajustado de sklearn.
        targets (Series): Series de pandas que contiene las etiquetas de los datos.

    Returns:
        None
    """
    pca_components = pca.transform(df)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=targets, palette='Set1')
    plt.title('Gráfico 2D de los Primeros Dos Componentes Principales')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Cargar los datos limpiados
    file_path = '../data/cleaned_wine_data.csv'
    df = pd.read_csv(file_path)
    
    # Definir el nombre de la columna de la clase
    class_column = df.columns[0]

    # Paso 1: Visualizar la distribución de una columna
    plot_distribution(df, column_name='1')

    # Paso 2: Visualizar la matriz de correlación
    plot_correlation_matrix(df)

    # Paso 3: Visualizar relaciones entre pares de características
    plot_pairwise_relationships(df, hue_column=class_column)

    # Paso 4: Aplicar PCA y visualizar la varianza explicada
    pca = PCA(n_components=5)
    pca.fit(df.drop(class_column, axis=1))  # Eliminar la columna de la clase para PCA
    plot_pca_variance(pca)

    # Paso 5: Visualizar los datos en 2D usando PCA
    plot_pca_2d(df.drop(class_column, axis=1), pca, df[class_column])
