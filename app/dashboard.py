import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configuración de página
st.set_page_config(
    page_title="SVM Customer Churn Predictor | TecNM Toluca",
    page_icon="🏦",
    layout="wide"
)

# Encabezado institucional
st.title("🏦 Sistema Predictivo de Fuga de Clientes (SVM)")
st.markdown("**Ingeniería del Conocimiento** | TecNM Campus Toluca | Docente: Ing. Rafael Hernández López")
st.markdown("---")

# Rutas de los artefactos serializados
MODEL_PATH = "models/svm_model.joblib"
SCALER_PATH = "models/scaler.joblib"
ENCODER_PATH = "models/encoder.joblib"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
    return model, scaler, encoder

# Validación de existencia de modelos
if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(ENCODER_PATH)):
    st.error("⚠️ No se encontraron los archivos `.joblib` en la carpeta `models/`.")
    st.info("Ejecuta primero `python src/train.py` con `Churn_Modelling.csv` en `data/raw/` para generar el modelo entrenado.")
    st.stop()

model, scaler, encoder = load_artifacts()

# Interfaz dividida en dos columnas
col_input, col_pred = st.columns([1, 1.2], gap="large")

with col_input:
    st.header("📋 Perfil del Cliente")
    credit_score = st.slider("Puntaje Crediticio (Credit Score)", 300, 850, 650, step=5)
    geography = st.selectbox("País de Residencia", ["France", "Spain", "Germany"])
    gender = st.selectbox("Género", ["Female", "Male"])
    age = st.slider("Edad (Años)", 18, 90, 40)
    tenure = st.slider("Antigüedad en el Banco (Años)", 0, 10, 5)
    balance = st.number_input("Saldo en Cuenta ($)", 0.0, 300000.0, 50000.0, step=1000.0)
    num_products = st.selectbox("Número de Productos Contratados", [1, 2, 3, 4], index=1)
    salary = st.number_input("Salario Estimado Anual ($)", 0.0, 250000.0, 75000.0, step=1000.0)

    input_df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "EstimatedSalary": salary
    }])

with col_pred:
    st.header("⚡ Inferencia en Tiempo Real (SVM)")
    st.write("Presiona el botón para proyectar las características del cliente y evaluar su posición frente al hiperplano.")

    if st.button("🚀 Ejecutar Predicción con SVM", use_container_width=True):
        try:
            # Preprocesamiento idéntico al entrenamiento
            encoded_input = encoder.transform(input_df)
            scaled_input = scaler.transform(encoded_input)

            # Predicción, probabilidad y distancia al hiperplano
            prediction = model.predict(scaled_input)[0]
            probability = model.predict_proba(scaled_input)[0][1]
            decision_dist = model.decision_function(scaled_input)[0]

            st.markdown("### Diagnóstico del Modelo:")
            if prediction == 1:
                st.error(f"🚨 **ALTO RIESGO DE FUGA (Clase 1)** — Probabilidad de abandono: **{probability * 100:.2f}%**")
            else:
                st.success(f"✅ **CLIENTE RETENIDO / ESTABLE (Clase 0)** — Probabilidad de abandono: **{probability * 100:.2f}%**")

            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    label="Probabilidad de Churn",
                    value=f"{probability * 100:.1f}%",
                    delta="- Riesgo de Abandono" if prediction == 1 else "+ Fidelizado",
                    delta_color="inverse" if prediction == 1 else "normal"
                )
            with m2:
                st.metric(
                    label="Distancia al Hiperplano (wᵀx + b)",
                    value=f"{decision_dist:.4f}",
                    help="Valores > 0 caen en la región de abandono; valores < 0 caen en la región de retención."
                )

            st.info(
                f"**Fundamento Geométrico (Kernel RBF):**  \n"
                f"La función de decisión evaluó el vector escalado obteniendo una distancia orientada de **`{decision_dist:.4f}`**, "
                f"ubicando al cliente {'por encima' if decision_dist > 0 else 'por debajo'} de la frontera óptima ($w^T x + b = 0$)."
            )

        except Exception as err:
            st.error(f"Error al procesar la inferencia: {err}")