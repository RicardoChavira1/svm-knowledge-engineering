
# ==========================================
# MODULO 3: PARTICION Y ESCALADO
# Proyecto: SVM - Prediccion de abandono
# ==========================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# PASO 1: DIVISION DE LOS DATOS 80/20
def dividir_datos(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Datos de entrenamiento:", X_train.shape)
    print("Datos de prueba:", X_test.shape)

    return X_train, X_test, y_train, y_test


# PASO 2: ESTANDARIZACION DE LOS DATOS
def escalar_datos(X_train, X_test):

    scaler = StandardScaler()

    # Ajustar solamente con entrenamiento
    X_train_scaled = scaler.fit_transform(X_train)

    # Transformar prueba sin Data Leakage
    X_test_scaled = scaler.transform(X_test)

    # Conversion a float32
    X_train_scaled = np.asarray(
        X_train_scaled, dtype=np.float32
    )

    X_test_scaled = np.asarray(
        X_test_scaled, dtype=np.float32
    )

    print("Escalado completado correctamente")

    return X_train_scaled, X_test_scaled, scaler
