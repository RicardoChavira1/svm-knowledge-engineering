# ==============================================================================
# PROYECTO FINAL: INGENIERÍA DEL CONOCIMIENTO (MÁQUINAS DE SOPORTE VECTORIAL)
# Archivo: src/preprocessing.py
# Módulos Integrados:
#   - Módulo 1 (Ricardo): Carga de datos, selección de features e imputación
#   - Módulo 2 (Deni): Codificación nominal One-Hot y matriz NumPy float32
#   - Módulo 3 (Vania): Partición estratificada 80/20 y estandarización
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------------------------------
# MÓDULO 1 Y 2: INGESTA, LIMPIEZA Y CODIFICACIÓN CATEGÓRICA
# ------------------------------------------------------------------------------
def cargar_y_codificar(filepath="data/raw/Churn_Modelling.csv"):
  dataset = pd.read_csv(filepath)

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
  num_cols = [
      "CreditScore",
      "Age",
      "Tenure",
      "Balance",
      "NumOfProducts",
      "EstimatedSalary",
  ]
  cat_cols = ["Geography", "Gender"]

  X_df = dataset[feature_cols].copy()
  y = dataset["Exited"].values

  # Imputación estadística por media para valores faltantes
  imputer_num = SimpleImputer(missing_values=np.nan, strategy="mean")
  X_df[num_cols] = imputer_num.fit_transform(X_df[num_cols])

  # Codificación categórica nominal con drop='first' para evitar colinealidad
  encoder = ColumnTransformer(
      transformers=[
          ("one_hot", OneHotEncoder(categories="auto", drop="first"), cat_cols)
      ],
      remainder="passthrough",
  )

  X_encoded = np.array(encoder.fit_transform(X_df), dtype=np.float32)
  return X_encoded, y, encoder


# ------------------------------------------------------------------------------
# MÓDULO 3 (VANIA): PARTICIÓN 80/20 Y ESTANDARIZACIÓN DE ESCALA
# ------------------------------------------------------------------------------
def dividir_datos(X, y):
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.20, random_state=42, stratify=y
  )
  print("Datos de entrenamiento:", X_train.shape)
  print("Datos de prueba:", X_test.shape)
  return X_train, X_test, y_train, y_test


def escalar_datos(X_train, X_test):
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)
  return X_train_scaled, X_test_scaled, scaler


# ------------------------------------------------------------------------------
# ORQUESTADOR PRINCIPAL PARA src/train.py
# ------------------------------------------------------------------------------
def prepare_data(filepath="data/raw/Churn_Modelling.csv"):
  X_encoded, y, encoder = cargar_y_codificar(filepath)
  X_train, X_test, y_train, y_test = dividir_datos(X_encoded, y)
  X_train_scaled, X_test_scaled, scaler = escalar_datos(X_train, X_test)
  return X_train_scaled, X_test_scaled, y_train, y_test, scaler, encoder


if __name__ == "__main__":
  X_tr, X_te, y_tr, y_te, sc, enc = prepare_data()
  print("Pipeline de preprocesamiento ejecutado correctamente.")