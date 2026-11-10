"""
Módulo de Preprocesamiento y ETL - SVM Customer Churn
Ingeniería del Conocimiento | TecNM Campus Toluca

Apartado 1: Pipeline ETL, Carga, Limpieza e Imputación Estadística
Rol: Ingeniero de Datos / Ingesta (Ricardo)
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def load_dataset(file_path: str = "data/raw/Churn_Modelling.csv") -> pd.DataFrame:
    """
    Carga el dataset CSV desde la ruta especificada con validación de existencia.
    
    Args:
        file_path: Ruta relativa o absoluta hacia Churn_Modelling.csv.
        
    Returns:
        pd.DataFrame: Dataset cargado en memoria.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[ERROR] No se encontró el archivo en '{file_path}'. "
            "Asegúrate de colocar Churn_Modelling.csv dentro de la carpeta data/raw/."
        )
    
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"[ERROR] El archivo '{file_path}' está vacío.")
        
    print(f"[INFO] Dataset cargado exitosamente desde: {file_path}")
    return df


def perform_eda(df: pd.DataFrame) -> None:
    """
    Realiza un Análisis Exploratorio de Datos (EDA) inicial exhaustivo:
    - Dimensiones y visualización preliminar.
    - Tipos de datos por columna (.info()).
    - Detección y conteo de valores nulos (.isna().sum()).
    - Distribución de la variable objetivo (Exited).
    
    Args:
        df: DataFrame original a inspeccionar.
    """
    print("\n" + "=" * 65)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA BÁSICO)")
    print("=" * 65)
    
    # 1. Dimensiones
    print(f"\n1. Dimensiones del Dataset (Filas, Columnas): {df.shape}")
    
    # 2. Primeras filas
    print("\n2. Primeras 5 filas del dataset:")
    print(df.head())
    
    # 3. Información general y tipos de datos
    print("\n3. Información general de columnas y tipos de datos:")
    df.info()
    
    # 4. Conteo de valores nulos
    print("\n4. Conteo de valores nulos por columna:")
    null_counts = df.isna().sum()
    print(null_counts)
    
    total_nulls = null_counts.sum()
    print(f"\nTotal general de valores nulos detectados: {total_nulls}")
    
    # 5. Balance de la clase objetivo (si está presente)
    if "Exited" in df.columns:
        print("\n5. Distribución de la variable objetivo ('Exited'):")
        counts = df["Exited"].value_counts()
        percentages = df["Exited"].value_counts(normalize=True) * 100
        balance_df = pd.DataFrame({"Casos": counts, "Porcentaje (%)": percentages.round(2)})
        print(balance_df)
    
    print("=" * 65 + "\n")


def clean_and_impute_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean",
    output_processed_path: str = "data/processed/churn_imputed.csv"
) -> Tuple[pd.DataFrame, SimpleImputer]:
    """
    Filtra identificadores irrelevantes, valida tipos, aplica imputación
    estadística sobre variables numéricas y persiste el dataset intermedio.
    
    Args:
        df: DataFrame cargado.
        strategy: Estrategia de imputación para variables numéricas ('mean' o 'median').
        output_processed_path: Ruta destino para persistir el paso intermedio.
        
    Returns:
        Tuple[pd.DataFrame, SimpleImputer]: DataFrame limpio e imputado listo
        para el Apartado 2, y el objeto SimpleImputer ajustado.
    """
    feature_cols = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
    ]
    target_col = "Exited"

    required_cols = feature_cols + [target_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"[ERROR] Columnas requeridas ausentes en el archivo: {missing_cols}")

    # Filtrar solo columnas relevantes para el modelado SVM
    df_clean = df[required_cols].copy()

    # Columnas por naturaleza de datos
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
    ]
    cat_cols = ["Geography", "Gender"]

    # Blindaje 1: Coerción numérica segura
    for col in num_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # Blindaje 2: Imputación numérica estadística
    print(f"[INFO] Aplicando SimpleImputer(strategy='{strategy}') sobre {len(num_cols)} variables numéricas...")
    imputer_num = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df_clean[num_cols] = imputer_num.fit_transform(df_clean[num_cols])

    # Blindaje 3: Validación e imputación preventiva en categóricas (por moda)
    for col in cat_cols:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        if df_clean[col].isna().any() or (df_clean[col] == "").any() or (df_clean[col] == "nan").any():
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].replace(["", "nan", np.nan], mode_val)
            print(f"[WARN] Nulos imputados por moda en variable categórica '{col}': {mode_val}")

    # Blindaje 4: Verificación estricta de ausencia de nulos
    remaining_nulls = df_clean.isna().sum().sum()
    if remaining_nulls != 0:
        raise RuntimeError(f"[ERROR] Falló el blindaje: quedan {remaining_nulls} nulos en los datos.")
    print(f"[INFO] Validación completada. Total de valores nulos restantes: {remaining_nulls}")

    # Blindaje 5: Persistencia en data/processed/ para desacoplamiento modular
    os.makedirs(os.path.dirname(output_processed_path), exist_ok=True)
    df_clean.to_csv(output_processed_path, index=False)
    print(f"[INFO] Dataset intermedio guardado en: '{output_processed_path}'")

    return df_clean, imputer_num


def run_pipeline_step_1(raw_path: str = "data/raw/Churn_Modelling.csv") -> pd.DataFrame:
    """
    Función de enlace para los siguientes integrantes.
    Ejecuta el Apartado 1 de principio a fin y retorna el DataFrame preparado.
    """
    df_raw = load_dataset(raw_path)
    perform_eda(df_raw)
    df_imputed, _ = clean_and_impute_missing_values(df_raw)
    return df_imputed


if __name__ == "__main__":
    # Ejecución directa del Apartado 1
    dataset_final = run_pipeline_step_1()
    print("\n[OK] Apartado 1 concluido exitosamente. Resumen de datos limpios:")
    print(dataset_final.head(3))