
# ==========================================
# PRUEBA DEL MODULO 3
# PARTICION Y ESCALADO DE DATOS
# ==========================================

import pandas as pd
import numpy as np

from src.preprocessing import dividir_datos, escalar_datos

# 1. Cargar el dataset
dataset = pd.read_csv("data/raw/Churn_Modelling.csv")

print("Total de clientes:", dataset.shape[0])

# 2. Seleccionar variables numericas
columnas = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "EstimatedSalary"
]

X = dataset[columnas]
y = dataset["Exited"]

# 3. Dividir los datos 80/20
X_train, X_test, y_train, y_test = dividir_datos(X, y)

# 4. Estandarizar los datos
X_train_scaled, X_test_scaled, scaler = escalar_datos(
    X_train, X_test
)

# 5. Comprobar la distribucion de clases
print("\nDistribucion de entrenamiento:")
print(y_train.value_counts(normalize=True))

print("\nDistribucion de prueba:")
print(y_test.value_counts(normalize=True))

# 6. Comprobar estandarizacion
print("\nMedia de entrenamiento:")
print(np.round(X_train_scaled.mean(axis=0), 4))

print("\nDesviacion estandar:")
print(np.round(X_train_scaled.std(axis=0), 4))

# 7. Validar resultados
assert X_train_scaled.shape == (8000, 6)
assert X_test_scaled.shape == (2000, 6)

assert np.allclose(
    X_train_scaled.mean(axis=0), 0, atol=1e-4
)

assert np.allclose(
    X_train_scaled.std(axis=0), 1, atol=1e-4
)

print("\nMODULO 3: TODAS LAS PRUEBAS CORRECTAS")
