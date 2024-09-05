import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data(file_path):
    """
    Cargar datos desde un archivo CSV.

    Args:
        file_path (str): Ruta al archivo CSV.

    Returns:
        DataFrame: DataFrame de pandas con los datos cargados.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Datos cargados exitosamente desde {file_path}")
        return df
    except FileNotFoundError:
        print(f"El archivo {file_path} no fue encontrado.")
        return None
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def preprocess_data(df):
    """
    Preprocesar los datos: escalado y división en conjunto de entrenamiento y prueba.

    Args:
        df (DataFrame): DataFrame con los datos a preprocesar.

    Returns:
        tuple: Tupla con los datos de entrenamiento y prueba escalados y las etiquetas.
    """
    # Separar características y etiquetas
    X = df.iloc[:, 1:]  # Todas las columnas excepto la primera
    y = df.iloc[:, 0].astype(int)   # Primera columna como etiquetas, convertidas a int si no lo son

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Datos escalados y divididos en conjunto de entrenamiento y prueba.")
    return X_train_scaled, X_test_scaled, y_train, y_test

def perform_pca(X_train, X_test, n_components=None):
    """
    Aplicar PCA para la reducción de dimensionalidad y visualizar la varianza explicada.

    Args:
        X_train (ndarray): Conjunto de entrenamiento escalado.
        X_test (ndarray): Conjunto de prueba escalado.
        n_components (int or None): Número de componentes principales o None para todos.

    Returns:
        tuple: Conjunto de entrenamiento y prueba transformados, y el modelo PCA ajustado.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Visualizar la varianza explicada por cada componente principal
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Explicada')
    plt.title('Varianza Explicada por Componentes Principales')
    plt.show()

    return X_train_pca, X_test_pca, pca

def evaluate_model(X_train, X_test, y_train, y_test, n_components):
    """
    Evaluar el clasificador Naive Bayes con datos transformados por PCA.

    Args:
        X_train (ndarray): Conjunto de entrenamiento transformado.
        X_test (ndarray): Conjunto de prueba transformado.
        y_train (Series): Etiquetas de entrenamiento.
        y_test (Series): Etiquetas de prueba.
        n_components (int): Número de componentes principales usados.

    Returns:
        float: Precisión del modelo.
    """
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Exactitud con {n_components} componentes principales: {accuracy:.2f}")
    return accuracy

def main():
    # Cargar datos
    file_path = '../data/cleaned_wine_data.csv'
    df = load_data(file_path)

    if df is not None:
        # Preprocesar datos
        X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)

        # Evaluar modelos con diferentes números de componentes principales
        results = []
        for n in range(1, X_train_scaled.shape[1] + 1):  # Iterar desde 1 hasta el número total de características
            X_train_pca, X_test_pca, pca = perform_pca(X_train_scaled, X_test_scaled, n_components=n)
            accuracy = evaluate_model(X_train_pca, X_test_pca, y_train, y_test, n)
            results.append((n, accuracy))

        # Visualizar resultados de la precisión en función del número de componentes principales
        plt.figure(figsize=(8, 6))
        plt.plot([x[0] for x in results], [x[1] for x in results], marker='o')
        plt.title('Exactitud del modelo en función del número de componentes principales')
        plt.xlabel('Número de Componentes Principales')
        plt.ylabel('Exactitud')
        plt.grid(True)
        plt.show()

    else:
        print("El proceso se detuvo debido a un error al cargar los datos.")

if __name__ == "__main__":
    main()
