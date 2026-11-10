"""
Módulo de Preprocesamiento y ETL - SVM Customer Churn
Ingeniería del Conocimiento | TecNM Campus Toluca

Apartado 1: Pipeline ETL, Carga y Limpieza de Datos
Rol: Ingeniero de Datos / Ingesta (Ricardo)
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


def load_dataset(file_path: str = "data/raw/Churn_Modelling.csv") -> pd.DataFrame:
    """
    Carga el dataset CSV desde la ruta especificada.
    
    Args:
        file_path: Ruta relativa o absoluta hacia Churn_Modelling.csv.
        
    Returns:
        pd.DataFrame: Dataset cargado en memoria.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"No se encontró el archivo en {file_path}. "
            "Asegúrate de colocar Churn_Modelling.csv dentro de data/raw/."
        )
    
    df = pd.read_csv(file_path)
    print(f"[INFO] Dataset cargado exitosamente desde: {file_path}")
    return df


def perform_eda(df: pd.DataFrame) -> None:
    """
    Realiza un Análisis Exploratorio de Datos (EDA) inicial:
    - Dimensiones del dataset.
    - Tipos de datos por columna (.info()).
    - Detección y conteo de valores nulos (.isna().sum()).
    
    Args:
        df: DataFrame original a inspeccionar.
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA BÁSICO)")
    print("=" * 60)
    
    # 1. Dimensiones
    print(f"\n1. Dimensiones (Filas, Columnas): {df.shape}")
    
    # 2. Primeras filas
    print("\n2. Primeras 5 filas del dataset:")
    print(df.head())
    
    # 3. Información general y tipos de datos
    print("\n3. Información general de columnas y tipos de datos:")
    df.info()
    
    # 4. Conteo de valores nulos
    print("\n4. Valores nulos por columna:")
    null_counts = df.isna().sum()
    print(null_counts)
    
    total_nulls = null_counts.sum()
    print(f"\nTotal general de valores nulos detectados: {total_nulls}")
    print("=" * 60 + "\n")


def clean_and_impute_missing_values(
    df: pd.DataFrame,
    strategy: str = "mean"
) -> Tuple[pd.DataFrame, SimpleImputer]:
    """
    Filtra las columnas irrelevantes (RowNumber, CustomerId, Surname),
    separa características de interés y realiza la imputación estadística
    de valores faltantes en variables numéricas usando SimpleImputer.
    
    Args:
        df: DataFrame cargado.
        strategy: Estrategia de imputación para variables numéricas (por defecto 'mean').
        
    Returns:
        Tuple[pd.DataFrame, SimpleImputer]: DataFrame con valores numéricos imputados
        y el estimador SimpleImputer ajustado.
    """
    # Columnas predictoras según la metodología del proyecto
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
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Faltan columnas requeridas en el archivo: {missing_cols}")

    # Trabajamos con una copia limpia
    df_clean = df[required_cols].copy()

    # Columnas numéricas sujetas a imputación estadística
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
    ]

    print(f"[INFO] Aplicando SimpleImputer(strategy='{strategy}') sobre variables numéricas...")
    imputer_num = SimpleImputer(missing_values=np.nan, strategy=strategy)
    df_clean[num_cols] = imputer_num.fit_transform(df_clean[num_cols])

    # Verificación posterior a la imputación
    remaining_num_nulls = df_clean[num_cols].isna().sum().sum()
    print(f"[INFO] Imputación completada. Nulos numéricos restantes: {remaining_num_nulls}")

    return df_clean, imputer_num


if __name__ == "__main__":
    # Prueba de ejecución local del Apartado 1
    raw_path = "data/raw/Churn_Modelling.csv"
    
    # 1. Carga
    df_raw = load_dataset(raw_path)
    
    # 2. EDA
    perform_eda(df_raw)
    
    # 3. Imputación estadística
    df_imputed, fitted_imputer = clean_and_impute_missing_values(df_raw)
    
    print("\nVista previa de los datos limpios e imputados:")
    print(df_imputed.head())