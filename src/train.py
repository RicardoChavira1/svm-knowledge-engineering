"""
Módulo de Modelado, Optimización de Kernels SVM y Persistencia de Artefactos.
"""
import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)


def compare_kernels(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> None:
    """
    Evalúa distintos kernels para justificar la selección del hiperplano óptimo (Fase 3 Rúbrica).
    """
    print("\n[*] Comparativa de optimización de Kernels SVM (Conjunto de Prueba):")
    print("-" * 68)
    print(f"{'Kernel':<12} | {'Accuracy':<12} | {'F1-Score (Fuga)':<18} | {'ROC-AUC':<12}")
    print("-" * 68)

    for kernel_name in ["linear", "poly", "rbf"]:
        clf = SVC(kernel=kernel_name, C=1.0, gamma="scale", probability=True, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        print(f"{kernel_name.upper():<12} | {acc:<12.4f} | {f1:<18.4f} | {auc:<12.4f}")
    print("-" * 68 + "\n")


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    c_param: float = 1.0,
    gamma_param: str = "scale",
    random_state: int = 42,
) -> SVC:
    """Configura y entrena el clasificador definitivo Support Vector Classifier (SVC)."""
    print(f"[*] Entrenando modelo definitivo SVC (kernel='rbf', C={c_param}, gamma='{gamma_param}')...")

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


def save_model(
    model: SVC,
    scaler=None,
    encoder=None,
    output_path: str = "models/svm_model.joblib",
) -> None:
    """Serializa el modelo entrenado y los transformadores para evaluación y dashboard."""
    models_dir = os.path.dirname(output_path) or "models"
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(model, output_path)
    print(f"[+] Modelo exportado correctamente en: '{output_path}'")

    if scaler is not None:
        scaler_path = os.path.join(models_dir, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"[+] Escalador exportado correctamente en: '{scaler_path}'")

    if encoder is not None:
        encoder_path = os.path.join(models_dir, "encoder.joblib")
        joblib.dump(encoder, encoder_path)
        print(f"[+] Codificador exportado correctamente en: '{encoder_path}'")


def export_powerbi_dataset(dataset_path: str, model: SVC, scaler, encoder) -> None:
    """
    Genera un archivo CSV enriquecido con las predicciones del modelo SVM y KPIs
    de negocio homologados a MXN para alimentar el Dashboard de Power BI.
    """
    df = pd.read_csv(dataset_path).drop_duplicates().reset_index(drop=True)
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

    X_enc = np.array(encoder.transform(df[feature_cols]), dtype=np.float32)
    X_sc = scaler.transform(X_enc)

    preds = model.predict(X_sc)
    probas = model.predict_proba(X_sc)[:, 1]

    df_bi = df.copy()
    df_bi["Pais_ES"] = df_bi["Geography"].map(
        {"France": "Francia", "Germany": "Alemania", "Spain": "España"}
    )
    df_bi["Genero_ES"] = df_bi["Gender"].map({"Female": "Femenino", "Male": "Masculino"})
    df_bi["Segmento_Edad"] = pd.cut(
        df_bi["Age"],
        bins=[0, 30, 45, 60, 100],
        labels=["Joven (18-30)", "Adulto (31-45)", "Maduro (46-60)", "Senior (60+)"],
    )
    df_bi["Nivel_CreditScore"] = pd.cut(
        df_bi["CreditScore"],
        bins=[0, 580, 670, 740, 900],
        labels=["Bajo (<580)", "Regular (580-669)", "Bueno (670-739)", "Excelente (740+)"],
    )
    df_bi["Saldo_MXN"] = (df_bi["Balance"] * 20.0).round(2)
    df_bi["Salario_MXN"] = (df_bi["EstimatedSalary"] * 20.0).round(2)
    df_bi["Estado_Real"] = np.where(df_bi["Exited"] == 1, "Abandonó (1)", "Retenido (0)")
    df_bi["Prediccion_SVM"] = preds
    df_bi["Etiqueta_Prediccion"] = np.where(preds == 1, "Riesgo de Fuga", "Cliente Estable")
    df_bi["Probabilidad_Fuga_Pct"] = (probas * 100).round(2)
    df_bi["Nivel_Riesgo_SVM"] = pd.cut(
        df_bi["Probabilidad_Fuga_Pct"],
        bins=[-1, 35, 65, 100],
        labels=["1. Riesgo Bajo", "2. Riesgo Moderado", "3. Riesgo Crítico"],
    )
    df_bi["Capital_En_Riesgo_MXN"] = np.where(preds == 1, df_bi["Saldo_MXN"], 0.0)

    out_file = os.path.join("data", "processed", "churn_powerbi_dashboard.csv")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df_bi.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"[+] Dataset maestro para Power BI generado en: '{out_file}'")


def run_pipeline() -> SVC:
    """Ejecuta el flujo completo de preprocesamiento, optimización, entrenamiento y exportación."""
    print("=== APARTADO 4 Y 7: MODELADO SVM, OPTIMIZACIÓN Y PERSISTENCIA ===")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

    from preprocessing import prepare_data

    dataset_path = os.path.join("data", "raw", "Churn_Modelling.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"No se encontró el dataset en '{dataset_path}'. "
            "Asegúrate de colocar 'Churn_Modelling.csv' en la carpeta 'data/raw/'."
        )

    print(f"[*] Obteniendo datos preprocesados desde: {dataset_path}")
    X_train, X_test, y_train, y_test, scaler, ct = prepare_data(dataset_path)

    # 1. Comparación de kernels para documentar la optimización (Fase 3)
    compare_kernels(X_train, X_test, y_train, y_test)

    # 2. Entrenamiento y guardado de los 3 artefactos .joblib (Fase 4 y 7)
    model = train_svm(X_train, y_train)
    save_model(model, scaler=scaler, encoder=ct)

    # 3. Exportación de base enriquecida para Power BI (Fase 4)
    export_powerbi_dataset(dataset_path, model, scaler, ct)
    return model


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as error:
        print(f"[!] Estado de integración: {error}")