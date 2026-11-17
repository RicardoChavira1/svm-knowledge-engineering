from pathlib import Path
import sys

# Asegurar que Python reconozca la raíz del proyecto para importar 'src'
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
  sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
import streamlit as st

# ------------------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="SVM Analytics · Predicción de Fuga Bancaria",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# PALETA OFICIAL UNIFICADA:
# Azul Primario: #4180AB | Blanco: #FFFFFF | Azul Medio: #8AB3CF
# Azul Suave: #BDD1DE | Fondo Claro: #E4EBF0 | Texto Oscuro: #1E293B
# ------------------------------------------------------------------------------
st.markdown(
    """
<style>
    .stApp {
        background-color: #E4EBF0;
        color: #1E293B;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 2px solid #BDD1DE;
    }
    section[data-testid="stSidebar"] * {
        color: #1E293B !important;
    }
    .hero-card {
        background: linear-gradient(135deg, #4180AB 0%, #2C5E80 100%);
        color: #FFFFFF;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 6px 20px rgba(65, 128, 171, 0.25);
    }
    .hero-badge {
        display: inline-block;
        background-color: rgba(255, 255, 255, 0.18);
        color: #FFFFFF;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 0.92rem;
        color: #E4EBF0;
        margin-top: 0.4rem;
    }
    .stage-box {
        background-color: #FFFFFF;
        border-left: 5px solid #4180AB;
        border-radius: 10px;
        padding: 0.95rem 1.3rem;
        margin-top: 1.1rem;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 10px rgba(30, 41, 59, 0.05);
    }
    .stage-step {
        color: #4180AB;
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stage-title {
        color: #1E293B;
        font-size: 1.12rem;
        font-weight: 700;
        margin: 2px 0 4px 0;
    }
    .stage-desc {
        color: #475569;
        font-size: 0.88rem;
        margin: 0;
    }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #BDD1DE;
        border-top: 4px solid #4180AB;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 12px rgba(30, 41, 59, 0.05);
    }
    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #4180AB !important;
        font-weight: 800 !important;
    }
    .verdict-safe {
        background-color: #FFFFFF;
        border: 2px solid #16A34A;
        border-left: 8px solid #16A34A;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        color: #1E293B;
    }
    .verdict-risk {
        background-color: #FFFFFF;
        border: 2px solid #DC2626;
        border-left: 8px solid #DC2626;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        color: #1E293B;
    }
    .verdict-margin {
        background-color: #FFFFFF;
        border: 2px solid #D97706;
        border-left: 8px solid #D97706;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        color: #1E293B;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# ENCABEZADO SUPERIOR
# ------------------------------------------------------------------------------
st.markdown(
    """
<div class="hero-card">
    <div class="hero-badge">Ingeniería del Conocimiento · TecNM Campus Toluca · MXN / SVM RBF</div>
    <div class="hero-title">Sistema Inteligente de Scoring y Predicción de Fuga Bancaria</div>
    <div class="hero-subtitle">Clasificador No Lineal con Kernel Gaussiano RBF (C = 1.0, γ = 'scale') · Docente: Ing. Rafael Hernández López</div>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# CARGA DE ARTEFACTOS Y ESPACIO DE REFERENCIA
# ------------------------------------------------------------------------------
MODELS_DIR = ROOT_DIR / "models"
DATA_PATH = ROOT_DIR / "data" / "raw" / "Churn_Modelling.csv"


@st.cache_resource
def load_artifacts():
  model = joblib.load(MODELS_DIR / "svm_model.joblib")
  scaler = joblib.load(MODELS_DIR / "scaler.joblib")
  encoder = joblib.load(MODELS_DIR / "encoder.joblib")
  return model, scaler, encoder


@st.cache_data
def load_reference_space(_encoder, _scaler):
  df = pd.read_csv(DATA_PATH).sample(500, random_state=42)
  features = [
      "CreditScore",
      "Geography",
      "Gender",
      "Age",
      "Tenure",
      "Balance",
      "NumOfProducts",
      "EstimatedSalary",
  ]
  X_enc = _encoder.transform(df[features])
  X_sc = _scaler.transform(X_enc)
  pca = PCA(n_components=2, random_state=42)
  coords = pca.fit_transform(X_sc)
  df_plot = pd.DataFrame({
      "PC1": coords[:, 0],
      "PC2": coords[:, 1],
      "Estado": np.where(df["Exited"] == 1, "Abandona (1)", "Se Queda (0)"),
      "Edad": df["Age"].values,
      "Saldo_USD": df["Balance"].values,
      "Region_Base": df["Geography"].values,
  })
  return df_plot, pca


try:
  model, scaler, encoder = load_artifacts()
except Exception as e:
  st.error(
      "Error al cargar los artefactos en `models/`. Ejecuta primero `python"
      f" src/train.py`: {e}"
  )
  st.stop()

# ------------------------------------------------------------------------------
# DICCIONARIOS DE HOMOLOGACIÓN (APORTE DEL COMPAÑERO INTEGRADO)
# ------------------------------------------------------------------------------
PAISES_MAP = {
    "México": "Spain",
    "España": "Spain",
    "Francia": "France",
    "Alemania": "Germany",
    "Estados Unidos": "Germany",
    "Canadá": "France",
    "Colombia": "Spain",
    "Argentina": "Spain",
    "Chile": "Spain",
    "Brasil": "Spain",
    "Reino Unido": "France",
}

GENEROS_MAP = {
    "Femenino": "Female",
    "Masculino": "Male",
}

PRODUCTOS_MAP = {
    "1 - Cuenta de Débito / Nómina": 1,
    "2 - Débito + Tarjeta de Crédito": 2,
    "3 - Débito + Crédito + Inversión": 3,
    "4 - Portafolio Patrimonial Completo": 4,
}

TIPO_CAMBIO_MXN = 20.0  # $20 MXN = 1 USD para mantener escala del modelo

# ------------------------------------------------------------------------------
# BARRA LATERAL: PERFIL DEMOGRÁFICO Y FINANCIERO (MXN)
# ------------------------------------------------------------------------------
with st.sidebar:
  st.markdown("### 🎛️ Perfil del Cliente")
  st.caption("Ajusta los atributos demográficos y financieros en pesos (MXN):")
  st.divider()

  pais_es = st.selectbox(
      "País de Residencia", options=list(PAISES_MAP.keys()), index=0
  )
  genero_es = st.selectbox("Género", options=list(GENEROS_MAP.keys()), index=0)
  edad = st.slider("Edad del Titular (años)", 18, 90, 38)
  antiguedad = st.slider("Antigüedad en el Banco (años)", 0, 10, 4)

  st.divider()
  st.markdown("### 💰 Situación Financiera (MXN)")

  credit_score = st.slider("Score Buró de Crédito", 300, 850, 660, step=5)
  saldo_mxn = st.number_input(
      "Saldo Promedio en Cuenta ($ MXN)",
      0.0,
      5000000.0,
      850000.0,
      step=25000.0,
      format="%.2f",
  )
  ingreso_mxn = st.number_input(
      "Ingreso Anual Estimado ($ MXN)",
      0.0,
      4000000.0,
      1200000.0,
      step=25000.0,
      format="%.2f",
  )
  productos_label = st.selectbox(
      "Productos Financieros Contratados",
      options=list(PRODUCTOS_MAP.keys()),
      index=1,
  )
  num_productos = PRODUCTOS_MAP[productos_label]

# Capa de traducción hacia el esquema exacto del encoder entrenado
input_model_df = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": PAISES_MAP[pais_es],
    "Gender": GENEROS_MAP[genero_es],
    "Age": edad,
    "Tenure": antiguedad,
    "Balance": saldo_mxn / TIPO_CAMBIO_MXN,
    "NumOfProducts": num_productos,
    "EstimatedSalary": ingreso_mxn / TIPO_CAMBIO_MXN,
}])

# Inferencia SVM
encoded_input = encoder.transform(input_model_df)
scaled_input = scaler.transform(encoded_input)
prediction = model.predict(scaled_input)[0]
probability = model.predict_proba(scaled_input)[0][1]
decision_dist = model.decision_function(scaled_input)[0]
capital_riesgo = saldo_mxn if prediction == 1 else 0.0

# ------------------------------------------------------------------------------
# PESTAÑAS PRINCIPALES
# ------------------------------------------------------------------------------
tab1, tab2 = st.tabs([
    "🧭 Simulación Paso a Paso (Explicación Visual)",
    "📊 Auditoría Estadística y Ficha Técnica (Apartado 5)",
])

# ==============================================================================
# PESTAÑA 1: SIMULACIÓN POR ETAPAS
# ==============================================================================
with tab1:
  # --- ETAPA 1: VEREDICTO E IMPACTO FINANCIERO ---
  st.markdown(
      """
    <div class="stage-box">
        <div class="stage-step">Etapa 1 · Dictamen del Clasificador e Impacto Financiero</div>
        <div class="stage-title">Diagnóstico de Retención, Margen de Soporte y Capital en Riesgo</div>
        <p class="stage-desc">Evaluación de la posición del cliente respecto al hiperplano óptimo (<b>wᵀφ(x) + b = 0</b>) y la avenida de margen blando <b>[-1, +1]</b>.</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

  if abs(decision_dist) < 1.0:
    st.markdown(
        f"""
        <div class="verdict-margin">
            <h3 style="margin:0; color:#D97706;">⚠️ ZONA DE MARGEN / VECTOR DE SOPORTE (Distancia: {decision_dist:+.4f})</h3>
            <p style="margin:6px 0 0 0; font-size:0.95rem;">
                El cliente se ubica dentro de la avenida de margen blando <b>[-1.0, +1.0]</b> con una probabilidad de abandono del <b>{probability * 100:.2f}%</b>. 
                Dictamen actual: <b>{"Riesgo de Fuga (Clase 1)" if prediction == 1 else "Retención Activa (Clase 0)"}</b> — requiere seguimiento preventivo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
  elif prediction == 1:
    st.markdown(
        f"""
        <div class="verdict-risk">
            <h3 style="margin:0; color:#DC2626;">🚨 ALERTA CRÍTICA: RIESGO DE FUGA CONFIRMADO (Clase 1)</h3>
            <p style="margin:6px 0 0 0; font-size:0.95rem;">
                Probabilidad de abandono del <b>{probability * 100:.2f}%</b>. El vector superó la frontera positiva (<b>{decision_dist:+.4f} ≥ +1.0</b>), 
                poniendo en riesgo <b>${capital_riesgo:,.2f} MXN</b> de saldo activo.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
  else:
    st.markdown(
        f"""
        <div class="verdict-safe">
            <h3 style="margin:0; color:#16A34A;">✅ CLIENTE FIDELIZADO / ESTABLE (Clase 0)</h3>
            <p style="margin:6px 0 0 0; font-size:0.95rem;">
                Probabilidad baja de abandono del <b>{probability * 100:.2f}%</b>. Su vector se ubica sólidamente en la zona segura 
                (<b>{decision_dist:+.4f} ≤ -1.0</b>), protegiendo <b>${saldo_mxn:,.2f} MXN</b> en cartera.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

  st.markdown("<br>", unsafe_allow_html=True)

  k1, k2, k3, k4 = st.columns(4)
  k1.metric(
      "Dictamen SVM",
      "ALTO RIESGO (1)" if prediction == 1 else "ESTABLE (0)",
  )
  k2.metric("Probabilidad de Fuga", f"{probability * 100:.2f}%")
  k3.metric("Distancia Hiperplano", f"{decision_dist:+.4f}")
  k4.metric("Capital en Riesgo (MXN)", f"${capital_riesgo:,.0f}")

  # --- ETAPA 2: MEDIDOR DE RIESGO Y ESTANDARIZACIÓN Z-SCORE ---
  st.markdown(
      """
    <div class="stage-box">
        <div class="stage-step">Etapa 2 · Transformación Matemática (StandardScaler)</div>
        <div class="stage-title">Índice Calibrado de Abandono y Perfil Estandarizado (μ=0, σ=1)</div>
        <p class="stage-desc">Las variables financieras en pesos (MXN) se homologan a dólares base y se estandarizan en desviaciones estándar (σ) para que el Kernel RBF calcule distancias equilibradas.</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

  col_g1, col_g2 = st.columns([1, 1.25])

  with col_g1:
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "#4180AB"}},
            title={
                "text": "Probabilidad Calibrada P(y=1 | x)",
                "font": {"size": 15, "color": "#1E293B"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#4180AB",
                },
                "bar": {
                    "color": "#DC2626" if probability >= 0.5 else "#4180AB"
                },
                "bgcolor": "#FFFFFF",
                "borderwidth": 1.5,
                "bordercolor": "#BDD1DE",
                "steps": [
                    {"range": [0, 35], "color": "#E4EBF0"},
                    {"range": [35, 65], "color": "#BDD1DE"},
                    {"range": [65, 100], "color": "#8AB3CF"},
                ],
                "threshold": {
                    "line": {"color": "#1E293B", "width": 3},
                    "thickness": 0.8,
                    "value": 50,
                },
            },
        )
    )
    fig_gauge.update_layout(
        paper_bgcolor="#FFFFFF",
        font={"color": "#1E293B"},
        height=340,
        margin=dict(l=25, r=25, t=55, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

  with col_g2:
    z_labels = [
        "Región: Alemania",
        "Región: España/MX",
        "Género: Masculino",
        "Credit Score",
        "Edad",
        "Antigüedad",
        "Saldo Homologado",
        "Núm. Productos",
        "Ingreso Homologado",
    ]
    z_vals = scaled_input[0]
    fig_z = go.Figure(
        go.Bar(
            x=z_vals,
            y=z_labels,
            orientation="h",
            marker_color=["#4180AB" if v >= 0 else "#8AB3CF" for v in z_vals],
            text=[f"{v:+.2f}σ" for v in z_vals],
            textposition="outside",
        )
    )
    fig_z.update_layout(
        title=dict(
            text="Vector Estandarizado (Z-Scores) enviado al Modelo SVM",
            font=dict(size=14, color="#1E293B"),
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font={"color": "#1E293B", "size": 12},
        height=340,
        margin=dict(l=15, r=35, t=50, b=20),
        xaxis=dict(
            title="Desviaciones respecto a la media (σ)",
            gridcolor="#E2E8F0",
            zerolinecolor="#4180AB",
            range=[-3.5, 3.5],
        ),
        yaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_z, use_container_width=True)

  # --- ETAPA 3: PROYECCIÓN GEOMÉTRICA PCA 2D ---
  st.markdown(
      """
    <div class="stage-box">
        <div class="stage-step">Etapa 3 · Geometría del Espacio Vectorial</div>
        <div class="stage-title">Ubicación del Cliente Simulado frente a la Cartera Histórica (PCA 2D)</div>
        <p class="stage-desc">Proyección bidimensional de 500 clientes reales. La <b>estrella dorada</b> indica dónde cae el perfil de <b>"""
      + f"{pais_es}"
      + """</b> tras su homologación.</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

  df_space, pca_model = load_reference_space(encoder, scaler)
  client_pca = pca_model.transform(scaled_input)[0]

  fig_scatter = px.scatter(
      df_space,
      x="PC1",
      y="PC2",
      color="Estado",
      color_discrete_map={
          "Se Queda (0)": "#4180AB",
          "Abandona (1)": "#E11D48",
      },
      hover_data=["Region_Base", "Edad", "Saldo_USD"],
      opacity=0.72,
  )
  fig_scatter.add_trace(
      go.Scatter(
          x=[client_pca[0]],
          y=[client_pca[1]],
          mode="markers+text",
          marker=dict(
              size=20,
              color="#F59E0B",
              symbol="star",
              line=dict(width=2, color="#1E293B"),
          ),
          text=[f"  CLIENTE ({pais_es.upper()})"],
          textposition="middle right",
          textfont=dict(color="#1E293B", size=12, family="Arial Black"),
          name="Cliente Evaluado",
      )
  )
  fig_scatter.update_layout(
      paper_bgcolor="#FFFFFF",
      plot_bgcolor="#F8FAFC",
      font={"color": "#1E293B"},
      height=420,
      margin=dict(l=20, r=20, t=30, b=20),
      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
      xaxis=dict(gridcolor="#E2E8F0", title="Componente Principal 1"),
      yaxis=dict(gridcolor="#E2E8F0", title="Componente Principal 2"),
  )
  st.plotly_chart(fig_scatter, use_container_width=True)

# ==============================================================================
# PESTAÑA 2: AUDITORÍA ESTADÍSTICA Y FICHA DE HOMOLOGACIÓN
# ==============================================================================
with tab2:
  st.markdown(
      """
    <div class="stage-box">
        <div class="stage-step">Validación Científica · Conjunto de Prueba (20% = 2,000 Clientes)</div>
        <div class="stage-title">Métricas de Rendimiento del Modelo SVM (Apartado 5)</div>
        <p class="stage-desc">Resultados reales obtenidos sobre el conjunto de prueba estratificado tras entrenar con 8,000 registros.</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

  m1, m2, m3, m4 = st.columns(4)
  m1.metric("Exactitud Global (Accuracy)", "85.40%")
  m2.metric("Área Bajo Curva (ROC-AUC)", "0.8114")
  m3.metric("Precisión en Fuga (Clase 1)", "78.61%")
  m4.metric("F1-Score (Clase 1)", "0.5197")

  st.markdown("<br>", unsafe_allow_html=True)

  c_cm, c_roc = st.columns(2)

  with c_cm:
    cm_values = [[1550, 43], [249, 158]]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm_values,
            x=["Predicho: Se Queda (0)", "Predicho: Abandona (1)"],
            y=["Real: Se Queda (0)", "Real: Abandona (1)"],
            text=[
                ["1,550<br>(Verdaderos Negativos)", "43<br>(Falsos Positivos)"],
                ["249<br>(Falsos Negativos)", "158<br>(Verdaderos Positivos)"],
            ],
            texttemplate="%{text}",
            textfont={"size": 14, "color": "#1E293B"},
            colorscale=[[0, "#E4EBF0"], [0.5, "#8AB3CF"], [1, "#4180AB"]],
            showscale=False,
        )
    )
    fig_cm.update_layout(
        title=dict(
            text="Matriz de Confusión (2,000 Clientes de Prueba)",
            font=dict(size=15, color="#1E293B"),
        ),
        paper_bgcolor="#FFFFFF",
        font={"color": "#1E293B"},
        height=410,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

  with c_roc:
    from src.preprocessing import prepare_data

    _, X_test_eval, _, y_test_eval, _, _ = prepare_data(str(DATA_PATH))
    y_probs_eval = model.predict_proba(X_test_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_eval, y_probs_eval)

    fig_roc = go.Figure()
    fig_roc.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name="SVM Kernel RBF (AUC = 0.8114)",
            line=dict(color="#4180AB", width=3.5),
            fill="tozeroy",
            fillcolor="rgba(138, 179, 207, 0.35)",
        )
    )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Clasificador Aleatorio (AUC = 0.50)",
            line=dict(color="#64748B", width=2, dash="dash"),
        )
    )
    fig_roc.update_layout(
        title=dict(
            text="Curva ROC-AUC (Capacidad de Separación de Clases)",
            font=dict(size=15, color="#1E293B"),
        ),
        xaxis_title="Tasa de Falsos Positivos",
        yaxis_title="Tasa de Verdaderos Positivos (Recall)",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font={"color": "#1E293B"},
        height=410,
        legend=dict(x=0.42, y=0.12, bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(gridcolor="#E2E8F0"),
        yaxis=dict(gridcolor="#E2E8F0"),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

  # --- AUDITORÍA DE TRADUCCIÓN MXN -> MODELO ---
  st.markdown(
      """
    <div class="stage-box">
        <div class="stage-step">Trazabilidad de Datos · Capa de Homologación</div>
        <div class="stage-title">Auditoría del Vector de Entrada (Interfaz MXN vs Esquema del Modelo)</div>
        <p class="stage-desc">Transparencia total de cómo los datos capturados en pesos mexicanos se convierten al esquema exacto esperado por <code>encoder.joblib</code> y <code>scaler.joblib</code>.</p>
    </div>
    """,
      unsafe_allow_html=True,
  )

  t1, t2 = st.columns(2)
  with t1:
    df_visual = pd.DataFrame({
        "Atributo Capturado": [
            "País de Residencia",
            "Género",
            "Edad",
            "Antigüedad",
            "Score Crediticio",
            "Saldo en Cuenta",
            "Ingreso Anual",
            "Productos",
        ],
        "Valor en Interfaz (MXN)": [
            pais_es,
            genero_es,
            f"{edad} años",
            f"{antiguedad} años",
            f"{credit_score} pts",
            f"${saldo_mxn:,.2f} MXN",
            f"${ingreso_mxn:,.2f} MXN",
            f"{num_productos} productos",
        ],
    })
    st.dataframe(df_visual, use_container_width=True, hide_index=True)

  with t2:
    df_tecnico = (
        input_model_df.T.reset_index()
        .rename(columns={"index": "Columna del Modelo", 0: "Valor Normalizado (Base USD)"})
    )
    df_tecnico["Valor Normalizado (Base USD)"] = df_tecnico[
        "Valor Normalizado (Base USD)"
    ].astype(str)
    st.dataframe(df_tecnico, use_container_width=True, hide_index=True)