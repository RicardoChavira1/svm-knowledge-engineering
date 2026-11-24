"""
Módulo de Inferencia sobre Clientes Completamente Nuevos (Fuera del Dataset Original).
- Modo normal: genera un lote inicial de 5 clientes nuevos evaluados en data/processed/nuevos_clientes_evaluados.csv
- Modo interactivo (--nuevo): captura un cliente dictado al momento, imprime el veredicto SVM y lo agrega al CSV para Power BI.
"""

import argparse
from datetime import datetime
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MODELS_DIR = Path("models")
OUT_CSV = Path("data/processed/nuevos_clientes_evaluados.csv")
TIPO_CAMBIO_MXN = 20.0

PAISES_MAP = {
    "francia": ("Francia", "France"),
    "france": ("Francia", "France"),
    "alemania": ("Alemania", "Germany"),
    "germany": ("Alemania", "Germany"),
    "españa": ("España", "Spain"),
    "espana": ("España", "Spain"),
    "spain": ("España", "Spain"),
    "mexico": ("México", "Spain"),
    "méxico": ("México", "Spain"),
    "eeuu": ("Estados Unidos", "Germany"),
    "usa": ("Estados Unidos", "Germany"),
}

GENEROS_MAP = {
    "f": ("Femenino", "Female"),
    "femenino": ("Femenino", "Female"),
    "female": ("Femenino", "Female"),
    "mujer": ("Femenino", "Female"),
    "m": ("Masculino", "Male"),
    "masculino": ("Masculino", "Male"),
    "male": ("Masculino", "Male"),
    "hombre": ("Masculino", "Male"),
}


def load_artifacts():
    model = joblib.load(MODELS_DIR / "svm_model.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    encoder = joblib.load(MODELS_DIR / "encoder.joblib")
    return model, scaler, encoder


def evaluate_client_record(model, scaler, encoder, record: dict) -> dict:
    pais_es, pais_model = PAISES_MAP.get(record["Pais"].strip().lower(), ("España", "Spain"))
    gen_es, gen_model = GENEROS_MAP.get(record["Genero"].strip().lower(), ("Femenino", "Female"))

    saldo_mxn = float(record["Saldo_MXN"])
    salario_mxn = float(record["Salario_MXN"])

    input_df = pd.DataFrame([{
        "CreditScore": float(record["CreditScore"]),
        "Geography": pais_model,
        "Gender": gen_model,
        "Age": float(record["Age"]),
        "Tenure": float(record["Tenure"]),
        "Balance": saldo_mxn / TIPO_CAMBIO_MXN,
        "NumOfProducts": float(record["NumOfProducts"]),
        "EstimatedSalary": salario_mxn / TIPO_CAMBIO_MXN,
    }])

    X_enc = np.array(encoder.transform(input_df), dtype=np.float32)
    X_sc = scaler.transform(X_enc)

    pred = int(model.predict(X_sc)[0])
    proba_pct = round(float(model.predict_proba(X_sc)[0][1]) * 100.0, 2)
    dist = round(float(model.decision_function(X_sc)[0]), 4)

    if abs(dist) < 1.0:
        zona = "Zona de Margen (Vector Soporte)"
    elif pred == 1:
        zona = "Riesgo Crítico Confirmado"
    else:
        zona = "Zona Segura (Fidelizado)"

    return {
        "ID_Nuevo": record["ID_Nuevo"],
        "Nombre_Cliente": record["Nombre_Cliente"],
        "Pais_ES": pais_es,
        "Pais_Modelo": pais_model,
        "Genero_ES": gen_es,
        "Age": int(record["Age"]),
        "Tenure": int(record["Tenure"]),
        "CreditScore": int(record["CreditScore"]),
        "Saldo_MXN": round(saldo_mxn, 2),
        "Salario_MXN": round(salario_mxn, 2),
        "NumOfProducts": int(record["NumOfProducts"]),
        "Prediccion_SVM": pred,
        "Dictamen_SVM": "Riesgo de Fuga (1)" if pred == 1 else "Cliente Estable (0)",
        "Probabilidad_Fuga_Pct": proba_pct,
        "Distancia_Hiperplano": dist,
        "Zona_SVM": zona,
        "Capital_En_Riesgo_MXN": round(saldo_mxn if pred == 1 else 0.0, 2),
        "Fecha_Evaluacion": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def seed_initial_new_clients():
    """Genera 5 clientes 100% nuevos (fuera de Churn_Modelling.csv) para poblar Power BI."""
    model, scaler, encoder = load_artifacts()
    sample_clients = [
        {
            "ID_Nuevo": "NEW-001",
            "Nombre_Cliente": "Sofía Mendoza (Demo)",
            "Pais": "México",
            "Genero": "Femenino",
            "Age": 31,
            "Tenure": 6,
            "CreditScore": 730,
            "Saldo_MXN": 450000,
            "Salario_MXN": 980000,
            "NumOfProducts": 2,
        },
        {
            "ID_Nuevo": "NEW-002",
            "Nombre_Cliente": "Hans Weber (Demo)",
            "Pais": "Alemania",
            "Genero": "Masculino",
            "Age": 55,
            "Tenure": 3,
            "CreditScore": 590,
            "Saldo_MXN": 2650000,
            "Salario_MXN": 1450000,
            "NumOfProducts": 1,
        },
        {
            "ID_Nuevo": "NEW-003",
            "Nombre_Cliente": "Laura Castillo (Demo)",
            "Pais": "España",
            "Genero": "Femenino",
            "Age": 49,
            "Tenure": 2,
            "CreditScore": 615,
            "Saldo_MXN": 1950000,
            "Salario_MXN": 1100000,
            "NumOfProducts": 3,
        },
        {
            "ID_Nuevo": "NEW-004",
            "Nombre_Cliente": "Jean Moreau (Demo)",
            "Pais": "Francia",
            "Genero": "Masculino",
            "Age": 42,
            "Tenure": 5,
            "CreditScore": 660,
            "Saldo_MXN": 1600000,
            "Salario_MXN": 1250000,
            "NumOfProducts": 1,
        },
        {
            "ID_Nuevo": "NEW-005",
            "Nombre_Cliente": "Clara Richter (Demo)",
            "Pais": "Alemania",
            "Genero": "Femenino",
            "Age": 58,
            "Tenure": 1,
            "CreditScore": 540,
            "Saldo_MXN": 3100000,
            "Salario_MXN": 1750000,
            "NumOfProducts": 4,
        },
    ]

    evaluated = [evaluate_client_record(model, scaler, encoder, c) for c in sample_clients]
    df_out = pd.DataFrame(evaluated)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n[+] 5 clientes completamente nuevos evaluados y guardados en:", OUT_CSV)
    print(df_out[["ID_Nuevo", "Nombre_Cliente", "Pais_ES", "Age", "NumOfProducts", "Dictamen_SVM", "Probabilidad_Fuga_Pct", "Distancia_Hiperplano"]].to_string(index=False))


def ask_value(prompt: str, default, cast_fn=str):
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return cast_fn(default)
    if cast_fn in (int, float):
        raw = raw.replace(",", "").replace("$", "").strip()
    return cast_fn(raw)


def interactive_new_client():
    """Captura un cliente nuevo en vivo desde consola y lo anexa al CSV de Power BI."""
    model, scaler, encoder = load_artifacts()

    if OUT_CSV.exists():
        df_prev = pd.read_csv(OUT_CSV)
        next_num = len(df_prev) + 1
    else:
        df_prev = pd.DataFrame()
        next_num = 1

    print("\n=== CAPTURA EN VIVO DE CLIENTE COMPLETAMENTE NUEVO ===")
    nombre = ask_value("Nombre del cliente", f"Cliente Nuevo {next_num}", str)
    pais = ask_value("País (Francia / Alemania / España / México)", "Alemania", str)
    genero = ask_value("Género (Femenino / Masculino)", "Femenino", str)
    edad = ask_value("Edad (18-90)", 52, int)
    antiguedad = ask_value("Antigüedad en años (0-10)", 3, int)
    credit = ask_value("Credit Score (350-850)", 620, int)
    saldo = ask_value("Saldo en Cuenta ($ MXN)", 2200000, float)
    salario = ask_value("Salario Anual ($ MXN)", 1300000, float)
    productos = ask_value("Número de productos (1-4)", 1, int)

    record = {
        "ID_Nuevo": f"NEW-{next_num:03d}",
        "Nombre_Cliente": nombre,
        "Pais": pais,
        "Genero": genero,
        "Age": edad,
        "Tenure": antiguedad,
        "CreditScore": credit,
        "Saldo_MXN": saldo,
        "Salario_MXN": salario,
        "NumOfProducts": productos,
    }

    result = evaluate_client_record(model, scaler, encoder, record)
    df_new = pd.concat([df_prev, pd.DataFrame([result])], ignore_index=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 62)
    print(f"DICTAMEN SVM PARA {result['Nombre_Cliente'].upper()} ({result['ID_Nuevo']})")
    print("=" * 62)
    print(f"  Predicción:           {result['Dictamen_SVM']}")
    print(f"  Probabilidad de Fuga: {result['Probabilidad_Fuga_Pct']:.2f}%")
    print(f"  Distancia Hiperplano: {result['Distancia_Hiperplano']:+.4f}  ({result['Zona_SVM']})")
    print(f"  Capital en Riesgo:    ${result['Capital_En_Riesgo_MXN']:,.2f} MXN")
    print("=" * 62)
    print(f"[+] Guardado en '{OUT_CSV}'. Presiona 'Actualizar' en Power BI para verlo al instante.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inferencia SVM para clientes completamente nuevos.")
    parser.add_argument("--nuevo", action="store_true", help="Capturar un cliente nuevo interactivamente.")
    args = parser.parse_args()

    if args.nuevo:
        interactive_new_client()
    else:
        seed_initial_new_clients()