"""
Apartado 7: Serialización, Persistencia y Verificación de Artefactos con Joblib.
Materia: Ingeniería del Conocimiento | TecNM Campus Toluca
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "svm_model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
ENCODER_PATH = MODELS_DIR / "encoder.joblib"
DATASET_PATH = Path("data/raw/Churn_Modelling.csv")


def save_artifacts(model: SVC, scaler: StandardScaler, encoder: ColumnTransformer) -> dict:
    """
    Serializa y guarda en disco el modelo SVM, el escalador y el codificador.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    paths = {
        "model": str(MODEL_PATH),
        "scaler": str(SCALER_PATH),
        "encoder": str(ENCODER_PATH),
    }

    print("[+] Artefactos serializados correctamente con joblib:")
    for name, path in paths.items():
        size_kb = os.path.getsize(path) / 1024
        print(f"    - {name}: {path} ({size_kb:.2f} KB)")

    return paths


def load_artifacts() -> tuple:
    """
    Carga y verifica los artefactos serializados desde la carpeta models/.
    """
    for path in (MODEL_PATH, SCALER_PATH, ENCODER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el artefacto requerido: '{path}'. "
                "Ejecuta primero 'python src/persistence.py'."
            )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)

    print("[+] Verificación exitosa: modelo, scaler y encoder cargados en memoria.")
    return model, scaler, encoder


def build_and_export_artifacts(csv_path: Path = DATASET_PATH) -> None:
    """
    Genera y exporta los artefactos .joblib asegurando compatibilidad
    con los módulos de entrenamiento y el dashboard interactivo.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo '{csv_path}'. "
            "Coloca 'Churn_Modelling.csv' dentro de 'data/raw/'."
        )

    print(f"[*] Leyendo dataset desde: {csv_path}")
    dataset = pd.read_csv(csv_path)
    dataset = dataset.drop_duplicates()

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
    y = dataset[target_col].values

    imputer_num = SimpleImputer(missing_values=np.nan, strategy="mean")
    X_df[num_cols] = imputer_num.fit_transform(X_df[num_cols])

    encoder = ColumnTransformer(
        transformers=[
            ("one_hot", OneHotEncoder(categories="auto", drop="first"), cat_cols)
        ],
        remainder="passthrough",
    )
    X_encoded = np.array(encoder.fit_transform(X_df), dtype=np.float32)

    X_train, X_test, y_train, _ = train_test_split(
        X_encoded, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("[*] Entrenando clasificador SVM (kernel='rbf', C=1.0, gamma='scale')...")
    model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    save_artifacts(model, scaler, encoder)

    loaded_model, _, _ = load_artifacts()
    sample_pred = loaded_model.predict_proba(X_test_scaled[:1])[0][1]
    print(f"[+] Prueba de inferencia post-serialización exitosa (Prob. muestra 1: {sample_pred:.4f})")


if __name__ == "__main__":
    build_and_export_artifacts()