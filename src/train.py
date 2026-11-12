"""
Módulo de Modelado y Entrenamiento SVM

"""
import os
import sys
import warnings
import joblib
import numpy as np
from sklearn.svm import SVC

# Silencia avisos de futuras versiones para mantener una salida limpia en consola
warnings.filterwarnings("ignore", category=FutureWarning)


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_param: float = 1.0,
    gamma_param: str = "scale",
    random_state: int = 42,
) -> SVC:
    """
    Configura y entrena el clasificador Support Vector Classifier (SVC).

    Parámetros:
        X_train (np.ndarray): Matriz de entrenamiento escalada (StandardScaler).
        y_train (np.ndarray): Vector objetivo binario (0 = retiene, 1 = fuga).
        c_param (float): Penalización por margen blando C (compromiso sesgo-varianza).
        gamma_param (str | float): Coeficiente de curvatura del kernel Gaussiano RBF.
        random_state (int): Semilla para garantizar reproducibilidad.

    Retorna:
        model (SVC): Estimador ajustado con cálculo de probabilidades habilitado.
    """
    print(f"[*] Entrenando SVC (kernel='rbf', C={c_param}, gamma='{gamma_param}')...")

    model = SVC(
        kernel="rbf",
        C=c_param,
        gamma=gamma_param,
        probability=True,
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    print("[+] Entrenamiento SVM finalizado exitosamente.")
    return model


def save_model(model: SVC, output_path: str = "models/svm_model.joblib") -> None:
    """Serializa el modelo entrenado para la evaluación y el dashboard web."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    print(f"[+] Modelo exportado correctamente en: '{output_path}'")


def run_pipeline() -> SVC:
    """Ejecuta el flujo de entrenamiento utilizando los datos reales del pipeline."""
    print("=== APARTADO 4: MODELADO Y ENTRENAMIENTO SVM ===")

    # Asegura visibilidad del directorio src para importaciones modulares
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    try:
        from preprocessing import prepare_data
    except ImportError:
        raise ImportError(
            "No se pudo importar 'prepare_data' desde 'src/preprocessing.py'. "
            "Verifica que los demas archivos hayan definido la función."
        )

    dataset_path = os.path.join("data", "raw", "Churn_Modelling.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"No se encontró el dataset en '{dataset_path}'. "
            "Asegúrate de colocar 'Churn_Modelling.csv' en la carpeta 'data/raw/'."
        )

    print(f"[*] Obteniendo datos preprocesados desde: {dataset_path}")
    X_train, X_test, y_train, y_test, scaler, ct = prepare_data(dataset_path)

    # Entrenamiento y guardado del modelo definitivo
    model = train_svm(X_train, y_train)
    save_model(model)
    return model


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as error:
        print(f"[!] Estado de integración: {error}")