import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Cargar datos desde un archivo .data delimitado por comas.

    Args:
        file_path (str): Ruta al archivo .data.

    Returns:
        DataFrame: DataFrame de pandas con los datos cargados.
    """
    if not os.path.exists(file_path):
        print(f"El archivo {file_path} no fue encontrado.")
        return None
    
    try:
        # Leer el archivo de datos, especificando que está delimitado por comas
        data = pd.read_csv(file_path, header=None, delimiter=',')
        print(f"Datos cargados exitosamente desde {file_path}")
        return data
    except pd.errors.EmptyDataError:
        print("El archivo está vacío.")
        raise
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        raise

def explore_data(df):
    """
    Explorar el DataFrame para ver información básica y detectar valores faltantes.

    Args:
        df (DataFrame): DataFrame de pandas a explorar.

    Returns:
        None
    """
    print("\nPrimeras 5 filas del conjunto de datos:")
    print(df.head())
    
    print("\nDescripción estadística:")
    print(df.describe())
    
    print("\nValores faltantes en cada columna:")
    print(df.isnull().sum())

    print("\nInformación general del DataFrame:")
    print(df.info())

def handle_missing_values(df):
    """
    Manejar valores faltantes en el DataFrame.

    Args:
        df (DataFrame): DataFrame de pandas con posibles valores faltantes.

    Returns:
        DataFrame: DataFrame con valores faltantes manejados.
    """
    # Estrategia simple: eliminar filas con valores faltantes
    initial_shape = df.shape
    df_cleaned = df.dropna()
    final_shape = df_cleaned.shape

    print(f"\nDatos antes de eliminar valores faltantes: {initial_shape}")
    print(f"Datos después de eliminar valores faltantes: {final_shape}")
    return df_cleaned

def scale_data(df):
    """
    Escalar datos numéricos en el DataFrame.

    Args:
        df (DataFrame): DataFrame de pandas a escalar.

    Returns:
        DataFrame: DataFrame escalado.
    """
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    
    print("\nDatos escalados usando StandardScaler.")
    return df_scaled

def save_data(df, output_path):
    """
    Guardar el DataFrame limpio y escalado en un archivo CSV.

    Args:
        df (DataFrame): DataFrame de pandas a guardar.
        output_path (str): Ruta de destino para el archivo CSV.

    Returns:
        None
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Datos guardados exitosamente en {output_path}")
    except Exception as e:
        print(f"Error al guardar los datos: {e}")
        raise

if __name__ == "__main__":
    # Ruta al archivo de datos
    file_path = '../data/wine.data'  # Ajusta la ruta según la ubicación de tu archivo
    output_path = '../data/cleaned_wine_data.csv'

    # Paso 1: Cargar los datos
    df = load_data(file_path)

    # Verifica si el archivo se cargó correctamente
    if df is not None:
        # Paso 2: Explorar los datos
        explore_data(df)
        
        # Paso 3: Manejar valores faltantes
        df_cleaned = handle_missing_values(df)
        
        # Paso 4: Escalar los datos
        df_scaled = scale_data(df_cleaned)
        
        # Paso 5: Guardar los datos limpios y escalados
        save_data(df_scaled, output_path)
    else:
        print("El proceso no puede continuar sin los datos cargados correctamente.")
