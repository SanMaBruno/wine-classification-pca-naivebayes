import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib  # Para guardar el modelo entrenado

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

def preprocess_data(df):
    """
    Preprocesar los datos: escalado y división en conjunto de entrenamiento y prueba.

    Args:
        df (DataFrame): DataFrame con los datos a preprocesar.

    Returns:
        tuple: Tupla con los datos de entrenamiento y prueba escalados y las etiquetas.
    """
    # Separar características y etiquetas
    X = df.iloc[:, 1:]
    y = df.iloc[:, 0].astype(int)

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Datos escalados y divididos en conjunto de entrenamiento y prueba.")
    return X_train_scaled, X_test_scaled, y_train, y_test

def train_and_evaluate(X_train, X_test, y_train, y_test, n_components):
    """
    Entrenar y evaluar el modelo Naive Bayes con PCA.

    Args:
        X_train (ndarray): Conjunto de entrenamiento.
        X_test (ndarray): Conjunto de prueba.
        y_train (Series): Etiquetas de entrenamiento.
        y_test (Series): Etiquetas de prueba.
        n_components (int): Número de componentes principales para PCA.

    Returns:
        GaussianNB: Modelo entrenado.
    """
    # Aplicar PCA
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA aplicado con {n_components} componentes principales.")

    # Entrenar el clasificador Naive Bayes
    classifier = GaussianNB()
    classifier.fit(X_train_pca, y_train)
    print("Modelo Naive Bayes entrenado.")

    # Evaluar el modelo
    y_pred = classifier.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Exactitud del modelo: {accuracy:.2f}")
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred))
    print("Reporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    return classifier

def save_model(model, file_path):
    """
    Guardar el modelo entrenado en un archivo.

    Args:
        model (GaussianNB): Modelo entrenado.
        file_path (str): Ruta del archivo donde guardar el modelo.
    """
    joblib.dump(model, file_path)
    print(f"Modelo guardado en {file_path}")

def main():
    # Cargar datos
    file_path = '../data/cleaned_wine_data.csv'
    df = load_data(file_path)

    if df is not None:
        # Preprocesar datos
        X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)

        # Entrenar y evaluar el modelo
        n_components = 2  # Usando 2 componentes principales basado en análisis previo
        model = train_and_evaluate(X_train_scaled, X_test_scaled, y_train, y_test, n_components)

        # Guardar el modelo entrenado
        save_model(model, '../models/naive_bayes_model.pkl')

    else:
        print("El proceso se detuvo debido a un error al cargar los datos.")

if __name__ == "__main__":
    main()
