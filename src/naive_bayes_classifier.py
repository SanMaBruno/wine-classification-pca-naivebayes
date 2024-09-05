import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

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

def apply_pca(X_train, X_test, n_components=2):
    """
    Aplicar PCA para la reducción de dimensionalidad.

    Args:
        X_train (ndarray): Conjunto de entrenamiento.
        X_test (ndarray): Conjunto de prueba.
        n_components (int): Número de componentes principales.

    Returns:
        tuple: Conjunto de entrenamiento y prueba transformados.
    """
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA aplicado con {n_components} componentes principales.")
    return X_train_pca, X_test_pca

def train_and_evaluate_classifier(X_train, X_test, y_train, y_test):
    """
    Entrenar y evaluar el clasificador Naive Bayes.

    Args:
        X_train (ndarray): Conjunto de entrenamiento transformado.
        X_test (ndarray): Conjunto de prueba transformado.
        y_train (Series): Etiquetas de entrenamiento.
        y_test (Series): Etiquetas de prueba.

    Returns:
        None
    """
    # Inicializar y entrenar el clasificador Naive Bayes
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Predecir con el modelo entrenado
    y_pred = classifier.predict(X_test)

    # Evaluar el modelo
    print("Evaluación del clasificador Naive Bayes:")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"Exactitud: {accuracy_score(y_test, y_pred):.2f}")

    # Visualización de la matriz de confusión
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicho')
    plt.ylabel('Verdadero')
    plt.show()

if __name__ == "__main__":
    # Cargar datos
    file_path = '../data/cleaned_wine_data.csv'
    df = load_data(file_path)

    if df is not None:
        # Preprocesar datos
        X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(df)

        # Aplicar PCA
        X_train_pca, X_test_pca = apply_pca(X_train_scaled, X_test_scaled, n_components=2)

        # Entrenar y evaluar el clasificador
        train_and_evaluate_classifier(X_train_pca, X_test_pca, y_train, y_test)
    else:
        print("El proceso se detuvo debido a un error al cargar los datos.")
