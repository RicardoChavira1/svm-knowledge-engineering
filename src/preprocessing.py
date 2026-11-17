
# ============================================================
# PROYECTO FINAL: INGENIERIA DEL CONOCIMIENTO
# MAQUINAS DE SOPORTE VECTORIAL (SVM)
#
# MODULO 2: INGENIERIA DE CARACTERISTICAS
# Archivo: src/preprocessing.py
#
# FUNCIONES:
# 1. Separacion de variables X e y
# 2. Codificacion categorica con OneHotEncoder
# 3. Integracion mediante ColumnTransformer
# 4. Conversion a matrices NumPy float32
# 5. Prueba con el dataset real
# ============================================================


# ============================================================
# 1. IMPORTACION DE LIBRERIAS
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# 2. DEFINICION DE VARIABLES DEL DATASET
# ============================================================

# Variables predictivas que utilizara el modelo SVM

FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary"
]

# Variable objetivo: abandono del cliente
# 0 = permanece en el banco
# 1 = abandona el banco

TARGET_COLUMN = "Exited"

# Variables categoricas nominales

CATEGORICAL_COLUMNS = [
    "Geography",
    "Gender"
]


# ============================================================
# 3. SEPARACION DE VARIABLES X E y
# ============================================================

def separar_variables(dataset):

    """
    Separa las variables predictivas (X)
    de la variable objetivo (y).
    """

    columnas_necesarias = (
        FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    # Verificar que todas las columnas existan
    faltantes = [
        columna
        for columna in columnas_necesarias
        if columna not in dataset.columns
    ]

    if faltantes:
        raise ValueError(
            f"Faltan columnas en el dataset: {faltantes}"
        )

    # X contiene las caracteristicas predictivas
    X = dataset[FEATURE_COLUMNS].copy()

    # y contiene la variable objetivo
    y = dataset[TARGET_COLUMN].to_numpy(
        dtype=np.float32
    )

    return X, y


# ============================================================
# 4. CREACION DEL CODIFICADOR CATEGORICO
# ============================================================

def crear_codificador():

    """
    Convierte Geography y Gender a valores numericos.

    categories='auto':
        Detecta automaticamente las categorias.

    drop='first':
        Elimina la primera categoria de cada variable
        para evitar columnas redundantes.

    ColumnTransformer:
        Aplica OneHotEncoder solo a las columnas
        categoricas y conserva las numericas.
    """

    # Configurar OneHotEncoder
    one_hot = OneHotEncoder(
        categories="auto",
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    # Crear transformador de columnas
    transformer = ColumnTransformer(
        transformers=[
            (
                "one_hot",
                one_hot,
                CATEGORICAL_COLUMNS
            )
        ],
        remainder="passthrough"
    )

    return transformer


# ============================================================
# 5. CODIFICACION DE DATOS DE ENTRENAMIENTO
# ============================================================

def codificar_entrenamiento(X_train):

    """
    Ajusta el codificador con los datos recibidos
    y los convierte a una matriz NumPy float32.
    """

    # Los valores faltantes los trata el Modulo 1
    if X_train.isna().any().any():
        raise ValueError(
            "Existen valores nulos en X. "
            "Ejecuta primero la limpieza del Modulo 1."
        )

    # Crear ColumnTransformer
    transformer = crear_codificador()

    # Detectar categorias y transformar datos
    X_encoded = transformer.fit_transform(X_train)

    # Conversion estricta a float32
    X_encoded = np.asarray(
        X_encoded,
        dtype=np.float32
    )

    return X_encoded, transformer


# ============================================================
# 6. TRANSFORMACION DE DATOS DE PRUEBA
# ============================================================

def transformar_datos(X_test, transformer):

    """
    Transforma datos nuevos utilizando el mismo
    codificador ajustado durante el entrenamiento.

    No utiliza fit_transform para evitar
    volver a aprender categorias.
    """

    # Aplicar transformacion existente
    X_encoded = transformer.transform(X_test)

    # Conversion a NumPy float32
    X_encoded = np.asarray(
        X_encoded,
        dtype=np.float32
    )

    return X_encoded


# ============================================================
# 7. PRUEBA INDEPENDIENTE CON EL DATASET REAL
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("PROYECTO FINAL - INGENIERIA DEL CONOCIMIENTO")
    print("MODULO 2: INGENIERIA DE CARACTERISTICAS")
    print("=" * 55)

    # --------------------------------------------------------
    # CARGAR DATASET REAL
    # --------------------------------------------------------

    # Ubicar la carpeta principal del proyecto
    carpeta_proyecto = Path(__file__).resolve().parent.parent

    # Ruta del dataset
    ruta_dataset = (
        carpeta_proyecto
        / "data"
        / "raw"
        / "Churn_Modelling.csv"
    )

    if not ruta_dataset.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset en: {ruta_dataset}"
        )

    # Cargar archivo CSV
    dataset = pd.read_csv(ruta_dataset)

    print("\n[1] DATASET CARGADO CORRECTAMENTE")

    print("\nDimensiones originales:")
    print(dataset.shape)

    print("\nPrimeras 5 filas del dataset:")
    print(dataset.head())

    # --------------------------------------------------------
    # SEPARAR VARIABLES X E y
    # --------------------------------------------------------

    X, y = separar_variables(dataset)

    print("\n" + "-" * 55)
    print("[2] SEPARACION DE VARIABLES X E y")
    print("-" * 55)

    print("\nVariables predictivas X:")
    print(X.columns.tolist())

    print("\nVariable objetivo:")
    print(TARGET_COLUMN)

    print("\nPrimeros 10 valores de y:")
    print(y[:10])

    print("\nDimensiones de X antes de codificar:")
    print(X.shape)

    print("\nDimensiones de y:")
    print(y.shape)

    # --------------------------------------------------------
    # CODIFICACION CATEGORICA
    # --------------------------------------------------------

    print("\n" + "-" * 55)
    print("[3] CODIFICACION ONE HOT ENCODER")
    print("-" * 55)

    print("\nColumnas categoricas:")
    print(CATEGORICAL_COLUMNS)

    print("\nCategorias originales:")

    for columna in CATEGORICAL_COLUMNS:
        print(
            f"{columna}: "
            f"{X[columna].dropna().unique().tolist()}"
        )

    # Codificar las variables
    X_encoded, transformer = codificar_entrenamiento(X)

    print("\nColumnas despues de la codificacion:")

    nombres_columnas = transformer.get_feature_names_out()

    for nombre in nombres_columnas:
        print("-", nombre)

    # --------------------------------------------------------
    # CONVERSION A MATRIZ NUMPY FLOAT32
    # --------------------------------------------------------

    print("\n" + "-" * 55)
    print("[4] CONVERSION MATRICIAL NUMPY")
    print("-" * 55)

    print("\nMatriz X codificada (primeras 5 filas):")
    print(X_encoded[:5])

    print("\nDimensiones finales de X:")
    print(X_encoded.shape)

    print("\nDimensiones finales de y:")
    print(y.shape)

    print("\nTipo de estructura X:")
    print(type(X_encoded))

    print("\nTipo de datos X:")
    print(X_encoded.dtype)

    print("\nTipo de datos y:")
    print(y.dtype)

    # --------------------------------------------------------
    # VALIDACIONES DEL MODULO
    # --------------------------------------------------------

    print("\n" + "-" * 55)
    print("[5] VALIDACIONES")
    print("-" * 55)

    assert isinstance(X_encoded, np.ndarray), (
        "X no es un arreglo NumPy"
    )

    assert X_encoded.dtype == np.float32, (
        "X no tiene tipo float32"
    )

    assert isinstance(y, np.ndarray), (
        "y no es un arreglo NumPy"
    )

    assert y.dtype == np.float32, (
        "y no tiene tipo float32"
    )

    assert X_encoded.shape[0] == y.shape[0], (
        "La cantidad de registros de X e y no coincide"
    )

    assert np.isfinite(X_encoded).all(), (
        "La matriz X contiene valores no finitos"
    )

    print("\n[OK] X es un arreglo NumPy")
    print("[OK] X utiliza float32")
    print("[OK] y es un arreglo NumPy")
    print("[OK] y utiliza float32")
    print("[OK] X e y tienen la misma cantidad de registros")
    print("[OK] Codificacion categorica completada")

    # --------------------------------------------------------
    # RESULTADO FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 55)
    print("MODULO 2 EJECUTADO CORRECTAMENTE")
    print("DATOS PREPARADOS PARA EL SIGUIENTE MODULO")
    print("=" * 55)
