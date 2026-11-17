from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="SVM Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("Máquinas de Soporte Vectorial (SVM) — Predicción de Fuga Bancaria")
st.markdown(
    "**Ingeniería del Conocimiento** | TecNM Campus Toluca | Docente: Ing."
    " Rafael Hernández López"
)
st.divider()

# Cargar artefactos entrenados
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

try:
  model = joblib.load(MODELS_DIR / "svm_model.joblib")
  scaler = joblib.load(MODELS_DIR / "scaler.joblib")
  encoder = joblib.load(MODELS_DIR / "encoder.joblib")
except Exception as e:
  st.error(
      "Error al cargar los modelos serializados. Ejecuta primero `python"
      f" src/train.py`: {e}"
  )
  st.stop()

# Tabs para separar Inferencia en Vivo y Métricas del Modelo
tab1, tab2 = st.tabs([
    "🔮 Simulador de Inferencia en Vivo",
    "📈 Diagnóstico y Métricas del Modelo (Apartado 5)",
])

with tab1:
  col1, col2 = st.columns([1, 2])

  with col1:
    st.subheader("Parámetros del Cliente")
    credit_score = st.slider("Credit Score", 300, 850, 650)
    geography = st.selectbox("País (Geography)", ["France", "Spain", "Germany"])
    gender = st.selectbox("Género (Gender)", ["Female", "Male"])
    age = st.slider("Edad (Age)", 18, 90, 40)
    tenure = st.slider("Antigüedad en años (Tenure)", 0, 10, 5)
    balance = st.number_input(
        "Saldo en Cuenta ($)", 0.0, 250000.0, 50000.0, step=1000.0
    )
    num_products = st.selectbox("Número de Productos", [1, 2, 3, 4])
    salary = st.number_input(
        "Salario Estimado ($)", 0.0, 200000.0, 75000.0, step=1000.0
    )

    input_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "EstimatedSalary": salary,
    }])

  with col2:
    st.subheader("Resultado del Clasificador SVM (Kernel RBF)")
    if st.button(
        "Ejecutar Predicción", type="primary", use_container_width=True
    ):
      encoded_input = encoder.transform(input_data)
      scaled_input = scaler.transform(encoded_input)

      prediction = model.predict(scaled_input)[0]
      probability = model.predict_proba(scaled_input)[0][1]
      decision_dist = model.decision_function(scaled_input)[0]

      if prediction == 1:
        st.error(
            "🚨 **Riesgo de Fuga Detectado (Clase 1)** — Probabilidad de"
            f" abandono: **{probability * 100:.2f}%**"
        )
      else:
        st.success(
            "✅ **Cliente Estable / Retenido (Clase 0)** — Probabilidad de"
            f" abandono: **{probability * 100:.2f}%**"
        )

      st.markdown("### Fundamento Geométrico del Hiperplano")
      st.write(
          "El vector de características fue estandarizado ($\\mu=0, \\sigma=1$)"
          " y proyectado al espacio de Hilbert mediante el Kernel Gaussiano"
          " RBF para evaluar su posición respecto a la frontera de decisión:"
      )

      m1, m2 = st.columns(2)
      m1.metric(
          label="Distancia orientada al hiperplano (w^T φ(x) + b)",
          value=f"{decision_dist:.4f}",
          delta="Zona de Fuga (+)" if decision_dist > 0 else "Zona Segura (-)",
          delta_color="inverse",
      )
      m2.metric(
          label="Probabilidad calibrada P(y=1 | x)",
          value=f"{probability * 100:.2f}%",
      )

with tab2:
  st.subheader("Desempeño Estadístico en Conjunto de Prueba (2,000 clientes)")
  k1, k2, k3, k4 = st.columns(4)
  k1.metric("Exactitud (Accuracy)", "85.40%")
  k2.metric("Área bajo Curva (ROC-AUC)", "0.8114")
  k3.metric("Precisión (Clase Fuga)", "78.61%")
  k4.metric("F1-Score (Clase Fuga)", "0.5197")

  st.divider()
  g1, g2 = st.columns(2)
  cm_path = REPORTS_DIR / "matriz_confusion.png"
  roc_path = REPORTS_DIR / "curva_roc.png"

  if cm_path.exists():
    g1.image(
        str(cm_path),
        caption="Matriz de Confusión - SVM",
        use_container_width=True,
    )
  if roc_path.exists():
    g2.image(
        str(roc_path),
        caption="Curva ROC-AUC - SVM",
        use_container_width=True,
    )